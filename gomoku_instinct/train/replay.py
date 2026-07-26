"""Replay buffer：滑动窗口 + 分片落盘。

设计取舍：

* **常驻内存的环形缓冲**。开发机内存足够放下整个窗口，磁盘只作容灾。
* **辅助标签在入库时算一次**。棋型与禁手点都是棋盘的确定性函数，
  而每条样本会被训练很多次，因此在入库时算一遍存下来，比每个 batch 现算便宜得多。
  标注在 C++ 里做且释放 GIL，用线程池摊到多核上。
* **分片落盘**。样本按固定条数攒成分片写盘（先写 .tmp 再 rename，保证原子性），
  续训时按时间倒序装回最近的若干分片。中断不会留下半截数据。
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from ..rules.constants import BLACK, WHITE
from ..rules.symmetry import NUM_SYMMETRIES, index_map

# 落盘分片里保存的字段。policy 用 float16：它是访问计数的归一化结果，
# 半精度足够，能省一半内存。
_FIELDS = {
    "boards": np.uint8,
    "policy": np.float16,
    "to_move": np.uint8,
    "history": np.int32,
    "move_number": np.int32,
    "value": np.float32,
    "plies_remaining": np.int32,
    "next_move": np.int32,
    "root_value": np.float32,
    "blunder_gap": np.float32,
    "threat_self": np.uint8,
    "threat_opp": np.uint8,
    "forbidden": np.uint8,
}

HISTORY = 4

_FIELDS_ORDER = list(_FIELDS)
_PER_CELL_FIELDS = {"boards", "policy", "threat_self", "threat_opp", "forbidden"}


@dataclass
class Batch:
    """一个训练批次，全部张量已在目标设备上。"""

    boards: torch.Tensor  # (B, N) uint8
    to_move: torch.Tensor  # (B,) uint8
    history: torch.Tensor  # (B, 4) int64
    move_number: torch.Tensor  # (B,) int64
    policy: torch.Tensor  # (B, N) float32，MCTS 访问分布
    value: torch.Tensor  # (B,) float32，行棋方视角的对局结果
    plies_remaining: torch.Tensor  # (B,) int64
    next_move: torch.Tensor  # (B,) int64，-1 表示本局最后一手
    threat_self: torch.Tensor  # (B, N) int64，行棋方的棋型等级
    threat_opp: torch.Tensor  # (B, N) int64，对方的棋型等级
    forbidden: torch.Tensor  # (B, N) float32，黑方禁手点
    root_value: torch.Tensor  # (B,) float32，搜索给出的根节点评估
    # (B,) float32，零搜索选点与搜索最优手的价值差。给默认值是为了让
    # 手工构造 Batch 的调用方（测试、旧 checkpoint 的数据）不必都改一遍。
    blunder_gap: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.blunder_gap is None:
            self.blunder_gap = torch.zeros(
                self.boards.shape[0], device=self.boards.device
            )

    def __len__(self) -> int:
        return self.boards.shape[0]


def compute_labels(
    boards: np.ndarray,
    to_move: np.ndarray,
    board_size: int,
    rules,
    workers: int = 8,
) -> dict[str, np.ndarray]:
    """算出棋型与禁手点标签。

    全部由规则导出，不含任何棋谱知识。C++ 侧会释放 GIL，所以线程池是有效的。
    """
    n = board_size * board_size
    count = boards.shape[0]
    threat_self = np.zeros((count, n), dtype=np.uint8)
    threat_opp = np.zeros((count, n), dtype=np.uint8)
    forbidden = np.zeros((count, n), dtype=np.uint8)

    def one(i: int) -> None:
        grid = boards[i]
        me = int(to_move[i])
        them = WHITE if me == BLACK else BLACK
        threat_self[i] = np.frombuffer(
            rules.pattern_map(grid, board_size, me), dtype=np.uint8
        )
        threat_opp[i] = np.frombuffer(
            rules.pattern_map(grid, board_size, them), dtype=np.uint8
        )
        forbidden[i] = np.frombuffer(
            rules.forbidden_map(grid, board_size), dtype=np.uint8
        )

    if workers > 1 and count > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, range(count)))
    else:
        for i in range(count):
            one(i)

    return {
        "threat_self": threat_self,
        "threat_opp": threat_opp,
        "forbidden": forbidden,
    }



def _read_shard(path: str, n: int) -> dict[str, np.ndarray] | None:
    """读一个分片，缺失字段补零。

    字段是会新增的（例如后来加的 blunder_gap）。若按 _FIELDS 逐字段硬取，
    旧分片会抛 KeyError；再被 except 吞掉的话，磁盘上积累的全部样本会**静默作废**，
    表现为续训后 buffer=0 而不报任何错。所以这里对缺失字段补零，
    真正读不出来才返回 None，并由调用方出声。
    """
    try:
        with np.load(path) as data:
            keys = set(data.files)
            count = data[_FIELDS_ORDER[0]].shape[0]
            chunk = {}
            for key, dtype in _FIELDS.items():
                if key in keys:
                    chunk[key] = data[key]
                else:
                    shape = (count, n) if key in _PER_CELL_FIELDS else (count,)
                    chunk[key] = np.zeros(shape, dtype=dtype)
            return chunk
    except (OSError, ValueError, KeyError, EOFError):
        return None


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        board_size: int,
        *,
        shard_dir: str | None = None,
        shard_size: int = 65536,
        keep_shards: int = 400,
        label_workers: int = 8,
        shard_prefix: str = "shard",
        blunder_threshold: float = 0.0,
        blunder_fraction: float = 0.0,
    ) -> None:
        self.capacity = capacity
        self.board_size = board_size
        self.n = board_size * board_size
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.keep_shards = keep_shards
        self.label_workers = label_workers
        # 多个 actor 进程共写一个目录时，各自用不同前缀避免撞名
        self.shard_prefix = shard_prefix
        self._seen_shards: set[str] = set()
        # 失误挖掘：把「零搜索会走错、且错得厉害」的局面按比例塞进每个批次。
        # 这类局面正是策略缺前瞻的地方，均匀采样会把它们淹没在海量平凡局面里。
        self.blunder_threshold = blunder_threshold
        self.blunder_fraction = blunder_fraction

        self.size = 0
        self.cursor = 0
        self.total_added = 0
        self._shard_index = 0
        self._pending: list[dict[str, np.ndarray]] = []
        self._pending_count = 0

        n = self.n
        self._data = {
            "boards": np.zeros((capacity, n), np.uint8),
            "policy": np.zeros((capacity, n), np.float16),
            "to_move": np.zeros(capacity, np.uint8),
            "history": np.zeros((capacity, HISTORY), np.int32),
            "move_number": np.zeros(capacity, np.int32),
            "value": np.zeros(capacity, np.float32),
            "plies_remaining": np.zeros(capacity, np.int32),
            "next_move": np.zeros(capacity, np.int32),
            "root_value": np.zeros(capacity, np.float32),
            "blunder_gap": np.zeros(capacity, np.float32),
            "threat_self": np.zeros((capacity, n), np.uint8),
            "threat_opp": np.zeros((capacity, n), np.uint8),
            "forbidden": np.zeros((capacity, n), np.uint8),
        }

        # 八重对称的下标映射，预先算好；增强时直接 gather。
        self._sym = np.stack(
            [np.asarray(index_map(board_size, t), dtype=np.int64)
             for t in range(NUM_SYMMETRIES)]
        )

        if shard_dir:
            os.makedirs(shard_dir, exist_ok=True)

    def __len__(self) -> int:
        return self.size

    # ── 入库 ────────────────────────────────────────────────────────────────
    def add_from_drain(self, drained: dict, rules) -> int:
        """把 SelfPlayActor.drain() 的产物入库，顺带算好辅助标签。"""
        count = int(drained["count"])
        if count == 0:
            return 0

        chunk = {
            "boards": np.ascontiguousarray(drained["boards"][:count]),
            "policy": drained["policy"][:count].astype(np.float16),
            "to_move": np.ascontiguousarray(drained["to_move"][:count]),
            "history": np.ascontiguousarray(drained["history"][:count]),
            "move_number": np.ascontiguousarray(drained["move_number"][:count]),
            "value": np.ascontiguousarray(drained["value"][:count]),
            "plies_remaining": np.ascontiguousarray(drained["plies_remaining"][:count]),
            "next_move": np.ascontiguousarray(drained["next_move"][:count]),
            "root_value": np.ascontiguousarray(drained["root_value"][:count]),
            "blunder_gap": np.ascontiguousarray(
                drained.get(
                    "blunder_gap", np.zeros(count, np.float32)
                )[:count]
            ),
        }
        chunk.update(
            compute_labels(
                chunk["boards"],
                chunk["to_move"],
                self.board_size,
                rules,
                self.label_workers,
            )
        )
        self.add(chunk)
        return count

    def add(self, chunk: dict[str, np.ndarray]) -> None:
        count = chunk["boards"].shape[0]
        if count == 0:
            return

        # 环形写入，跨越末端时分两段。
        start = self.cursor
        first = min(count, self.capacity - start)
        for key, array in self._data.items():
            src = chunk[key]
            array[start : start + first] = src[:first]
            if first < count:
                array[: count - first] = src[first:]

        self.cursor = (start + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        self.total_added += count

        if self.shard_dir:
            self._pending.append({k: np.array(v, copy=True) for k, v in chunk.items()})
            self._pending_count += count
            if self._pending_count >= self.shard_size:
                self._flush_shard()

    # ── 采样 ────────────────────────────────────────────────────────────────
    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
        rng: np.random.Generator,
        augment: bool = True,
    ) -> Batch:
        if self.size == 0:
            raise RuntimeError("replay buffer 为空")

        idx = rng.integers(0, self.size, size=batch_size)
        # 失误挖掘：定向过采样「零搜索会走错」的局面。
        # 这类局面在全体样本里占比很小，均匀采样几乎碰不到，
        # 而它们恰恰是策略缺前瞻能力的地方。
        n_blunder = int(batch_size * self.blunder_fraction)
        if n_blunder > 0 and self.blunder_threshold > 0:
            gaps = self._data["blunder_gap"][: self.size]
            pool = np.flatnonzero(gaps > self.blunder_threshold)
            if pool.size:
                picked = rng.choice(pool, size=min(n_blunder, batch_size))
                idx[: picked.size] = picked

        boards = self._data["boards"][idx]
        policy = self._data["policy"][idx].astype(np.float32)
        history = self._data["history"][idx]
        next_move = self._data["next_move"][idx]
        threat_self = self._data["threat_self"][idx]
        threat_opp = self._data["threat_opp"][idx]
        forbidden = self._data["forbidden"][idx]

        if augment:
            # 八重二面体对称是纯规则导出的免费增强：局面变换后规则完全不变。
            t = rng.integers(0, NUM_SYMMETRIES, size=batch_size)
            perm = self._sym[t]  # (B, N)：perm[i][j] 是原下标 j 变换后的位置
            inv = np.argsort(perm, axis=1)  # gather 用逆映射

            boards = np.take_along_axis(boards, inv, axis=1)
            policy = np.take_along_axis(policy, inv, axis=1)
            threat_self = np.take_along_axis(threat_self, inv, axis=1)
            threat_opp = np.take_along_axis(threat_opp, inv, axis=1)
            forbidden = np.take_along_axis(forbidden, inv, axis=1)

            # 落点下标也要跟着变换；-1 保持不变
            hist_valid = history >= 0
            history = np.where(
                hist_valid, np.take_along_axis(perm, np.maximum(history, 0), axis=1), -1
            )
            nm_valid = next_move >= 0
            next_move = np.where(
                nm_valid,
                perm[np.arange(batch_size), np.maximum(next_move, 0)],
                -1,
            )

        dev = torch.device(device)

        def to(array, dtype):
            return torch.from_numpy(np.ascontiguousarray(array)).to(dev).to(dtype)

        return Batch(
            boards=to(boards, torch.uint8),
            to_move=to(self._data["to_move"][idx], torch.uint8),
            history=to(history, torch.int64),
            move_number=to(self._data["move_number"][idx], torch.int64),
            policy=to(policy, torch.float32),
            value=to(self._data["value"][idx], torch.float32),
            plies_remaining=to(self._data["plies_remaining"][idx], torch.int64),
            next_move=to(next_move, torch.int64),
            threat_self=to(threat_self, torch.int64),
            threat_opp=to(threat_opp, torch.int64),
            forbidden=to(forbidden, torch.float32),
            root_value=to(self._data["root_value"][idx], torch.float32),
            blunder_gap=to(self._data["blunder_gap"][idx], torch.float32),
        )

    # ── 落盘与恢复 ──────────────────────────────────────────────────────────
    def _flush_shard(self) -> None:
        if not self._pending or not self.shard_dir:
            return
        merged = {
            key: np.concatenate([p[key] for p in self._pending], axis=0)
            for key in _FIELDS
        }
        name = f"{self.shard_prefix}_{self._shard_index:08d}.npz"
        path = os.path.join(self.shard_dir, name)
        # 先写临时文件再 rename：中断时不会留下半截分片被续训读到。
        tmp = path + ".tmp.npz"
        np.savez(tmp, **merged)
        os.replace(tmp, path)

        self._shard_index += 1
        self._pending.clear()
        self._pending_count = 0
        self._prune_shards()

    def flush(self) -> None:
        """把未满的分片也写出去 —— checkpoint 前调用，避免丢样本。"""
        self._flush_shard()

    def _prune_shards(self) -> None:
        if not self.shard_dir or self.keep_shards <= 0:
            return
        shards = sorted(
            f
            for f in os.listdir(self.shard_dir)
            if f.startswith(self.shard_prefix + "_")
        )
        for name in shards[: max(0, len(shards) - self.keep_shards)]:
            try:
                os.remove(os.path.join(self.shard_dir, name))
            except OSError:
                pass

    def resume_shard_index(self) -> int:
        """从磁盘上已有的同前缀分片接着编号。

        actor 进程每次重启都会新建一个 sink，编号若从 0 重来，写出的分片会与上一次
        运行的同名：既覆盖了旧数据，又因为 trainer 的「已见分片」集合里早有这个名字
        而被整批跳过 —— 新样本被静默丢弃，训练看似在跑却一步不动。
        """
        if not self.shard_dir or not os.path.isdir(self.shard_dir):
            return 0
        prefix = self.shard_prefix + "_"
        existing = [
            f
            for f in os.listdir(self.shard_dir)
            if f.startswith(prefix) and f.endswith(".npz") and not f.endswith(".tmp.npz")
        ]
        if existing:
            self._shard_index = max(
                int(f[len(prefix) : len(prefix) + 8]) for f in existing
            ) + 1
        return self._shard_index

    def ingest_new_shards(self, prefixes: tuple[str, ...] = ("actor",)) -> int:
        """装入外部 actor 进程新写出来的分片。

        多卡编排下 actor 各自独占一张 GPU 跑自博弈、把样本写成分片；
        trainer 只管扫描新分片并入库。分片是原子 rename 出来的，
        因此扫到的一定是完整文件，不需要额外的握手协议。
        """
        if not self.shard_dir or not os.path.isdir(self.shard_dir):
            return 0

        candidates = sorted(
            f
            for f in os.listdir(self.shard_dir)
            if f.endswith(".npz")
            and not f.endswith(".tmp.npz")
            # 分片名形如 actor0_00000123.npz，前缀里带 actor 编号，
            # 所以要用 startswith 匹配，不能拿 split("_")[0] 去比对
            and any(f.startswith(p) for p in prefixes)
            and f not in self._seen_shards
        )

        loaded = 0
        for name in candidates:
            path = os.path.join(self.shard_dir, name)
            chunk = _read_shard(path, self.n)
            if chunk is None:
                continue  # 还没写完或已被回收，下轮再看
            self._seen_shards.add(name)
            # 入库但不再重复落盘 —— 分片已经在磁盘上了
            shard_dir, self.shard_dir = self.shard_dir, None
            self.add(chunk)
            self.shard_dir = shard_dir
            loaded += chunk["boards"].shape[0]
        return loaded

    def restore_from_shards(self) -> int:
        """续训时按时间倒序装回最近的分片，直到填满窗口。"""
        if not self.shard_dir or not os.path.isdir(self.shard_dir):
            return 0
        # 续训时把所有分片（自身的与各 actor 的）都算上
        shards = sorted(
            f
            for f in os.listdir(self.shard_dir)
            if f.endswith(".npz") and not f.endswith(".tmp.npz")
        )
        if not shards:
            return 0
        own = [f for f in shards if f.startswith(self.shard_prefix + "_")]
        if own:
            self._shard_index = int(own[-1].rsplit("_", 1)[1][:8]) + 1

        # total_added 统计的是「历史上一共产出过多少样本」，用于算样本复用率；
        # 从磁盘装回来不是新产出，不能重复计数。
        produced = self.total_added

        loaded = 0
        skipped = 0
        for name in reversed(shards):
            if loaded >= self.capacity:
                break
            chunk = _read_shard(os.path.join(self.shard_dir, name), self.n)
            if chunk is None:
                skipped += 1
                continue
            # 直接写入环形缓冲，但不再重复落盘
            shard_dir, self.shard_dir = self.shard_dir, None
            self.add(chunk)
            self.shard_dir = shard_dir
            loaded += chunk["boards"].shape[0]

        # 已经装回来的分片不该再被 ingest_new_shards 重复吃一遍
        self._seen_shards.update(shards)
        self.total_added = produced
        if skipped:
            print(f"[replay] 有 {skipped}/{len(shards)} 个分片读不出来，已跳过", flush=True)
        return min(loaded, self.capacity)

    def state_dict(self) -> dict:
        return {
            "size": self.size,
            "cursor": self.cursor,
            "total_added": self.total_added,
            "shard_index": self._shard_index,
            "capacity": self.capacity,
            "board_size": self.board_size,
        }

    def load_state_dict(self, state: dict) -> None:
        self.total_added = state.get("total_added", 0)
        self._shard_index = max(self._shard_index, state.get("shard_index", 0))

    def write_manifest(self, path: str) -> None:
        with open(path + ".tmp", "w") as fh:
            json.dump(self.state_dict(), fh, indent=2)
        os.replace(path + ".tmp", path)
