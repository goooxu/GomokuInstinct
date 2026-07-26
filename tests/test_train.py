"""Replay buffer、损失与续训的测试。

重点在**容灾**：开发机随时可能换机，训练必须能原地续训。
所以这里不只查「保存/加载没报错」，而是逐项比对恢复后的状态是否与保存时完全一致 ——
包括最容易被忽略的 Kahan 补偿缓冲区和 replay buffer 的分片游标。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from gomoku_instinct.core import load_core, make_rules
from gomoku_instinct.model import ModelConfig
from gomoku_instinct.rules import BLACK, EMPTY, WHITE
from gomoku_instinct.rules.constants import Level
from gomoku_instinct.rules.symmetry import NUM_SYMMETRIES
from gomoku_instinct.selfplay import RandomEvaluator, SelfPlayActor
from gomoku_instinct.train import (
    LossWeights,
    ReplayBuffer,
    Trainer,
    TrainerConfig,
    compute_labels,
    compute_losses,
    lr_at,
)

SIZE = 9
N = SIZE * SIZE


@pytest.fixture(scope="module")
def rules():
    return make_rules(None)


def _fake_drain(count: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    boards = np.zeros((count, N), dtype=np.uint8)
    for i in range(count):
        k = int(rng.integers(0, 20))
        cells = rng.choice(N, size=k, replace=False)
        for j, cell in enumerate(cells):
            boards[i, cell] = BLACK if j % 2 == 0 else WHITE
    policy = rng.random((count, N)).astype(np.float32)
    policy *= boards == EMPTY
    policy /= np.maximum(policy.sum(axis=1, keepdims=True), 1e-9)

    # 对手的应手必须落在空点上（真实数据即如此），否则测的就不是增强本身了
    next_move = np.empty(count, dtype=np.int32)
    for i in range(count):
        empties = np.flatnonzero(boards[i] == EMPTY)
        next_move[i] = -1 if rng.random() < 0.1 else int(rng.choice(empties))

    return {
        "count": count,
        "boards": boards,
        "policy": policy,
        "to_move": np.where(
            (boards != EMPTY).sum(axis=1) % 2 == 0, BLACK, WHITE
        ).astype(np.uint8),
        "history": np.full((count, 4), -1, dtype=np.int32),
        "move_number": (boards != EMPTY).sum(axis=1).astype(np.int32),
        "value": rng.choice([-1.0, 0.0, 1.0], size=count).astype(np.float32),
        "plies_remaining": rng.integers(1, 40, size=count).astype(np.int32),
        "next_move": next_move,
        "root_value": rng.uniform(-1, 1, size=count).astype(np.float32),
        "searched": np.ones(count, dtype=np.uint8),
    }


# ── 辅助标签 ────────────────────────────────────────────────────────────────


def test_compute_labels_shapes_and_range(rules):
    drained = _fake_drain(16, seed=1)
    labels = compute_labels(drained["boards"], drained["to_move"], SIZE, rules, 4)
    for key in ("threat_self", "threat_opp", "forbidden"):
        assert labels[key].shape == (16, N)
    assert labels["threat_self"].max() <= int(Level.OVERLINE)
    assert set(np.unique(labels["forbidden"]).tolist()) <= {0, 1}


def test_labels_are_zero_on_occupied_points(rules):
    drained = _fake_drain(8, seed=2)
    labels = compute_labels(drained["boards"], drained["to_move"], SIZE, rules, 2)
    occupied = drained["boards"] != EMPTY
    assert labels["threat_self"][occupied].max() == 0
    assert labels["forbidden"][occupied].max() == 0


# ── Replay buffer ───────────────────────────────────────────────────────────


def test_buffer_add_and_sample(rules, tmp_path):
    buf = ReplayBuffer(1000, SIZE, shard_dir=str(tmp_path / "replay"), shard_size=64)
    added = buf.add_from_drain(_fake_drain(200, seed=3), rules)
    assert added == 200
    assert len(buf) == 200

    batch = buf.sample(32, "cpu", np.random.default_rng(0))
    assert len(batch) == 32
    assert batch.boards.shape == (32, N)
    assert batch.policy.shape == (32, N)
    assert torch.allclose(batch.policy.sum(-1), torch.ones(32), atol=1e-3)


def test_buffer_is_a_sliding_window(rules):
    buf = ReplayBuffer(100, SIZE)
    buf.add_from_drain(_fake_drain(80, seed=4), rules)
    buf.add_from_drain(_fake_drain(80, seed=5), rules)
    assert len(buf) == 100  # 容量封顶
    assert buf.total_added == 160


def test_symmetry_augmentation_preserves_policy_mass(rules):
    """八重对称增强必须保持「目标分布只落在空点上」这条性质。"""
    buf = ReplayBuffer(500, SIZE)
    buf.add_from_drain(_fake_drain(200, seed=6), rules)
    batch = buf.sample(128, "cpu", np.random.default_rng(1), augment=True)

    occupied = batch.boards != EMPTY
    assert batch.policy[occupied].max().item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(batch.policy.sum(-1), torch.ones(128), atol=1e-3)


def test_augmentation_keeps_next_move_on_empty_cells(rules):
    buf = ReplayBuffer(500, SIZE)
    buf.add_from_drain(_fake_drain(200, seed=7), rules)
    batch = buf.sample(128, "cpu", np.random.default_rng(2), augment=True)

    valid = batch.next_move >= 0
    if valid.any():
        idx = torch.nonzero(valid).squeeze(-1)
        cells = batch.boards[idx, batch.next_move[idx]]
        # 变换后的落点仍必须指向空点；否则说明下标映射写反了
        assert torch.all(cells == EMPTY)


def test_augmentation_actually_changes_boards(rules):
    """增强要真的在变换棋盘，否则这条「免费数据增强」是假的。"""
    buf = ReplayBuffer(500, SIZE)
    buf.add_from_drain(_fake_drain(200, seed=8), rules)
    plain = buf.sample(64, "cpu", np.random.default_rng(3), augment=False)
    augmented = buf.sample(64, "cpu", np.random.default_rng(3), augment=True)
    assert not torch.equal(plain.boards, augmented.boards)


def test_shards_are_written_and_restorable(rules, tmp_path):
    shard_dir = str(tmp_path / "replay")
    buf = ReplayBuffer(5000, SIZE, shard_dir=shard_dir, shard_size=100)
    buf.add_from_drain(_fake_drain(450, seed=9), rules)
    buf.flush()

    shards = [f for f in os.listdir(shard_dir) if f.startswith("shard_")]
    assert shards, "没有写出任何分片"
    assert not any(f.endswith(".tmp.npz") for f in os.listdir(shard_dir)), (
        "留下了临时文件，原子写有问题"
    )

    restored = ReplayBuffer(5000, SIZE, shard_dir=shard_dir, shard_size=100)
    count = restored.restore_from_shards()
    assert count == 450
    assert len(restored) == 450


def test_trainer_ingests_actor_shards(rules, tmp_path):
    """多卡编排下 trainer 必须能扫到 actor 写出的分片。

    分片名形如 actor0_00000123.npz —— 前缀带 actor 编号。这条曾经漏掉：
    匹配写成了拿下划线切分后的首段去比对集合，`"actor0" in ("actor",)` 永远为假，
    于是 trainer 扫了几小时一个分片都没吃进去，自博弈白跑。
    这类 bug 不会报错，只会让训练步数一直停在 0。
    """
    shard_dir = str(tmp_path / "replay")

    # 模拟两个 actor 各写出一批分片
    for actor_id in (0, 1):
        sink = ReplayBuffer(
            5000, SIZE, shard_dir=shard_dir, shard_size=50,
            shard_prefix=f"actor{actor_id}",
        )
        sink.add_from_drain(_fake_drain(100, seed=40 + actor_id), rules)
        sink.flush()

    trainer_buf = ReplayBuffer(5000, SIZE, shard_dir=shard_dir, shard_size=50)
    ingested = trainer_buf.ingest_new_shards()
    assert ingested == 200, f"只吃到 {ingested} 条，actor 分片没被扫到"
    assert len(trainer_buf) == 200

    # 同一个分片不能被重复吃进来
    assert trainer_buf.ingest_new_shards() == 0


def test_ingest_ignores_trainers_own_shards(rules, tmp_path):
    """trainer 自己写的分片不该被再吃一遍，否则样本会翻倍。"""
    shard_dir = str(tmp_path / "replay")
    own = ReplayBuffer(5000, SIZE, shard_dir=shard_dir, shard_size=50)
    own.add_from_drain(_fake_drain(100, seed=44), rules)
    own.flush()

    assert own.ingest_new_shards() == 0


# ── 损失 ────────────────────────────────────────────────────────────────────


def test_losses_are_finite_and_metrics_present(rules):
    from gomoku_instinct.model import InstinctNet, encode

    buf = ReplayBuffer(500, SIZE)
    buf.add_from_drain(_fake_drain(200, seed=10), rules)
    batch = buf.sample(16, "cpu", np.random.default_rng(4))

    cfg = ModelConfig(size=SIZE, channels=16, blocks=2, attn_every=2)
    net = InstinctNet(cfg)
    planes = encode(
        batch.boards, batch.to_move, batch.history, batch.move_number,
        SIZE, dtype=torch.float32,
    )
    out = net(planes)
    loss, metrics = compute_losses(
        out, batch, LossWeights(), step=0, num_levels=cfg.threat_levels
    )
    assert torch.isfinite(loss)
    for key in ("loss/policy", "loss/value", "loss/threat", "loss/forbidden"):
        assert key in metrics and np.isfinite(metrics[key])
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())


def _forbidden_batch(device="cpu"):
    """造一个黑方待走、且确实存在禁手点的局面。

    (7,5)(7,6) 与 (5,7)(6,7) 四子在手，(7,7) 是三三禁手 —— 与规则测试里
    那条用例同形，这里复用它来构造带禁手标签的训练批次。
    """
    from gomoku_instinct.core import make_rules

    rules_core = make_rules(None)
    size = 15
    n = size * size
    grid = bytearray(n)
    for r, c in [(7, 5), (7, 6), (5, 7), (6, 7)]:
        grid[r * size + c] = BLACK
    forbidden = np.frombuffer(rules_core.forbidden_map(bytes(grid), size), np.uint8)
    assert forbidden.sum() > 0, "构造的局面里没有禁手点，测试前提不成立"

    boards = torch.tensor([list(grid)], dtype=torch.uint8, device=device)
    policy = torch.zeros(1, n, device=device)
    policy[0, 0] = 1.0  # 目标分布随便给个合法点
    return (
        boards,
        torch.tensor(forbidden.copy(), dtype=torch.float32, device=device).unsqueeze(0),
        policy,
        size,
        n,
    )


def test_forbidden_mass_penalty_pushes_policy_off_forbidden_points():
    """策略把概率压在禁手点上时，必须被直接惩罚。

    这是零搜索部署下最致命的失误：走上去直接判负，而对战时没有搜索兜底。
    单靠策略交叉熵推不动 —— 禁手点在 225 个点里只占极小的概率质量。
    """
    from gomoku_instinct.model import InstinctNet, ModelConfig, encode
    from gomoku_instinct.train.replay import Batch

    boards, forbidden, policy, size, n = _forbidden_batch()
    forbidden_idx = int(forbidden[0].argmax().item())

    batch = Batch(
        boards=boards,
        to_move=torch.tensor([BLACK], dtype=torch.uint8),
        history=torch.full((1, 4), -1, dtype=torch.int64),
        move_number=torch.tensor([4], dtype=torch.int64),
        policy=policy,
        value=torch.zeros(1),
        plies_remaining=torch.tensor([5], dtype=torch.int64),
        next_move=torch.tensor([-1], dtype=torch.int64),
        threat_self=torch.zeros(1, n, dtype=torch.int64),
        threat_opp=torch.zeros(1, n, dtype=torch.int64),
        forbidden=forbidden,
        root_value=torch.zeros(1),
    )

    cfg = ModelConfig(size=size, channels=16, blocks=2, attn_every=2)
    net = InstinctNet(cfg)
    planes = encode(
        batch.boards, batch.to_move, batch.history, batch.move_number,
        size, dtype=torch.float32,
    )

    # 人为把禁手点的 logit 抬到最高，模拟「策略学会了走最凶的一手」
    out = net(planes)
    out.policy = out.policy.clone()
    out.policy[0, forbidden_idx] += 20.0

    off = LossWeights(policy_forbidden=0.0)
    on = LossWeights(policy_forbidden=2.0)
    loss_off, m_off = compute_losses(out, batch, off, 0, cfg.threat_levels)
    loss_on, m_on = compute_losses(out, batch, on, 0, cfg.threat_levels)

    assert m_off["policy/forbidden_mass"] > 0.9, "构造失败：策略没压在禁手点上"
    assert m_off["policy/forbidden_argmax_rate"] == 1.0
    assert loss_on.item() > loss_off.item() + 1.0, "惩罚项没有生效"

    # 梯度必须把那个 logit 往下压
    out2 = net(planes)
    logit = out2.policy[0, forbidden_idx]
    out2.policy = out2.policy.clone()
    out2.policy[0, forbidden_idx] = logit + 20.0
    loss, _ = compute_losses(out2, batch, on, 0, cfg.threat_levels)
    grad = torch.autograd.grad(loss, out2.policy, retain_graph=True)[0]
    assert grad[0, forbidden_idx] > 0, "禁手点 logit 的梯度方向不对"


def test_forbidden_metrics_absent_when_white_to_move():
    """禁手只约束黑方，白方待走时不该产出这些指标。"""
    from gomoku_instinct.model import InstinctNet, ModelConfig, encode
    from gomoku_instinct.train.replay import Batch

    boards, forbidden, policy, size, n = _forbidden_batch()
    batch = Batch(
        boards=boards,
        to_move=torch.tensor([WHITE], dtype=torch.uint8),
        history=torch.full((1, 4), -1, dtype=torch.int64),
        move_number=torch.tensor([5], dtype=torch.int64),
        policy=policy,
        value=torch.zeros(1),
        plies_remaining=torch.tensor([5], dtype=torch.int64),
        next_move=torch.tensor([-1], dtype=torch.int64),
        threat_self=torch.zeros(1, n, dtype=torch.int64),
        threat_opp=torch.zeros(1, n, dtype=torch.int64),
        forbidden=forbidden,
        root_value=torch.zeros(1),
    )
    cfg = ModelConfig(size=size, channels=16, blocks=2, attn_every=2)
    net = InstinctNet(cfg)
    planes = encode(
        batch.boards, batch.to_move, batch.history, batch.move_number,
        size, dtype=torch.float32,
    )
    _, metrics = compute_losses(net(planes), batch, LossWeights(), 0, cfg.threat_levels)
    assert "policy/forbidden_mass" not in metrics


def test_aux_weights_decay_over_training():
    w = LossWeights(decay_start_step=100, decay_end_step=200, decay_final_scale=0.1)
    assert w.aux_scale(0) == 1.0
    assert w.aux_scale(150) == pytest.approx(0.55)
    assert w.aux_scale(1000) == pytest.approx(0.1)


def test_train_steps_follow_target_reuse(tmp_path):
    """训练步数应当自适应地追着目标复用率走。

    固定配比很容易失衡：自博弈一手棋要几百次网络评估，而一个训练步只吃一个 batch，
    实测按固定配比跑复用率会飙到 20 以上。
    """
    trainer = _tiny_trainer(str(tmp_path / "reuse"))
    trainer.cfg.batch_size = 100
    trainer.cfg.target_sample_reuse = 4.0
    trainer.cfg.max_train_steps_per_cycle = 1000

    trainer.buffer.total_added = 1000
    trainer.samples_seen = 0
    assert trainer.train_steps_for_cycle() == 40  # 4000 次使用 / batch 100

    trainer.samples_seen = 3900
    assert trainer.train_steps_for_cycle() == 1

    # 已经用够了就不再训练，等自博弈补新样本
    trainer.samples_seen = 4000
    assert trainer.train_steps_for_cycle() == 0
    trainer.samples_seen = 99999
    assert trainer.train_steps_for_cycle() == 0


def test_lr_schedule_warms_up_then_decays():
    cfg = TrainerConfig(lr=1e-3, warmup_steps=100, max_steps=1000, min_lr_scale=0.05)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(1000, cfg) == pytest.approx(1e-3 * 0.05, rel=1e-3)
    assert lr_at(500, cfg) < lr_at(100, cfg)


# ── 续训 ────────────────────────────────────────────────────────────────────


def _tiny_trainer(run_dir: str) -> Trainer:
    cfg = TrainerConfig(
        board_size=SIZE,
        num_games=8,
        selfplay_threads=2,
        sims=4,
        fast_sims=2,
        full_search_prob=1.0,
        capacity=5000,
        min_positions_to_start=1,
        shard_size=64,
        batch_size=8,
        max_steps=1_000_000,
        compile=False,
        selfplay_steps_per_cycle=120,
        max_train_steps_per_cycle=3,
        target_sample_reuse=1000.0,  # 测试里要确保每周期都真的训练几步
        label_workers=2,
        checkpoint_every_seconds=1e9,
        checkpoint_every_steps=1_000_000,
        resign_enabled=False,
        seed=4321,
    )
    model_cfg = ModelConfig(size=SIZE, channels=16, blocks=2, attn_every=2)
    return Trainer(cfg, model_cfg, LossWeights(), run_dir, device="cpu")


@pytest.mark.slow
def test_checkpoint_restores_full_training_state(tmp_path):
    run_dir = str(tmp_path / "run")
    trainer = _tiny_trainer(run_dir)

    for _ in range(6):
        trainer.selfplay_cycle()
    assert len(trainer.buffer) > 0
    for _ in range(5):
        trainer.train_step()

    path = trainer.save_checkpoint()
    assert os.path.exists(path)

    before_step = trainer.step
    before_buffer = len(trainer.buffer)
    before_weights = {
        k: v.clone() for k, v in trainer.model.state_dict().items()
    }
    before_comp = trainer.optimizer.compensation_norm()

    # 模拟换机：全新进程重新构造 trainer，只有 run_dir 是共享的
    revived = _tiny_trainer(run_dir)
    assert revived.resume_or_start()

    assert revived.step == before_step
    assert revived.buffer.total_added == trainer.buffer.total_added
    assert len(revived.buffer) == before_buffer, "replay buffer 没有从分片恢复回来"

    for key, value in revived.model.state_dict().items():
        assert torch.equal(value, before_weights[key]), f"权重 {key} 没恢复"

    assert revived.optimizer.compensation_norm() == pytest.approx(
        before_comp, rel=1e-6
    ), "Kahan 补偿缓冲区没恢复 —— 会丢掉已经攒下的低位更新"

    # 恢复后还能继续训练
    revived.train_step()
    assert revived.step == before_step + 1


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_checkpoint_roundtrip_on_gpu(tmp_path):
    """GPU 上的存取续训。

    这条曾经漏掉：CPU 上的续训测试全过，但在 GPU 上 torch.load 会把随机数状态
    ByteTensor 一并搬到显存里，set_rng_state 直接拒收，续训在启动时就崩。
    正式训练跑在 GPU 上，这条路径必须单独覆盖。
    """
    run_dir = str(tmp_path / "gpu_run")
    cfg = TrainerConfig(
        board_size=SIZE, num_games=4, selfplay_threads=2, sims=4, fast_sims=2,
        full_search_prob=1.0, capacity=2000, min_positions_to_start=1,
        shard_size=32, batch_size=8, compile=False,
        selfplay_steps_per_cycle=120, max_train_steps_per_cycle=2,
        target_sample_reuse=1000.0, label_workers=2,
        checkpoint_every_seconds=1e9, checkpoint_every_steps=1_000_000,
        resign_enabled=False, seed=99,
    )
    model_cfg = ModelConfig(size=SIZE, channels=16, blocks=2, attn_every=2)

    trainer = Trainer(cfg, model_cfg, LossWeights(), run_dir, device="cuda")
    for _ in range(6):
        trainer.selfplay_cycle()
    assert len(trainer.buffer) > 0
    trainer.train_step()
    trainer.save_checkpoint()

    revived = Trainer(cfg, model_cfg, LossWeights(), run_dir, device="cuda")
    assert revived.resume_or_start()
    assert revived.step == trainer.step
    revived.train_step()  # 恢复后还能继续训练


@pytest.mark.slow
def test_latest_pointer_survives_multiple_checkpoints(tmp_path):
    run_dir = str(tmp_path / "run2")
    trainer = _tiny_trainer(run_dir)
    for _ in range(6):  # 攒到有完整对局产出样本为止
        trainer.selfplay_cycle()
    assert len(trainer.buffer) > 0

    paths = []
    for _ in range(3):
        trainer.train_step()
        paths.append(trainer.save_checkpoint())

    assert trainer.latest_checkpoint() == paths[-1]
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    assert not any(f.endswith(".tmp") for f in os.listdir(ckpt_dir)), (
        "留下了临时文件，checkpoint 的原子写有问题"
    )
