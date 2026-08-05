#include "search.h"

#include <algorithm>
#include <cstring>

#include "thread_pool.h"

namespace gi {

BatchSearcher::BatchSearcher(int board_size, int sims, const MctsConfig& mcts,
                             const RuleConfig& rules,
                             ForbiddenSemantics semantics, int num_slots,
                             int num_threads, int leaves_per_slot)
    : size_(board_size), sims_(sims), leaves_(std::max(1, leaves_per_slot)) {
  rules_ = std::make_unique<Rules>(rules);

  MctsConfig cfg = mcts;
  // 每次搜索重建一棵树，节点池只需覆盖单次搜索。批量收集时一轮会多开几个节点，
  // 按 leaves 留出余量 —— 池子耗尽本身有兜底，但那条路径会让搜索白跑。
  cfg.max_nodes = sims + 8 * std::max(1, leaves_per_slot) + 8;

  const int n = num_cells();
  slots_.resize(num_slots);
  for (Slot& slot : slots_) {
    slot.pos = std::make_unique<Position>(board_size, rules_.get(), semantics);
    slot.tree = std::make_unique<MctsTree>(n, cfg);
    slot.visits.resize(n);
    slot.pending.resize(leaves_);
    for (Pending& p : slot.pending) p.moves.resize(n);
  }
  pool_ = std::make_unique<ThreadPool>(std::max(0, num_threads));
}

BatchSearcher::~BatchSearcher() = default;

void BatchSearcher::set_positions(const int32_t* moves, const int32_t* counts,
                                  int count) {
  active_count_ = std::min(count, capacity());

  int offset = 0;
  for (int i = 0; i < capacity(); ++i) {
    Slot& slot = slots_[i];
    slot.pos->reset();
    slot.tree->clear();
    slot.sims_done = 0;
    slot.pending_count = 0;
    slot.leaf = -1;
    slot.active = i < active_count_;

    if (!slot.active) continue;
    // 按完整着法序列重放：网络输入含最近数手的落点平面，
    // 只摆棋盘的话那几个平面会全空，测的就不是同一个输入下的表现了。
    for (int k = 0; k < counts[i]; ++k) {
      slot.pos->play(moves[offset + k]);
    }
    slot.root_ply = slot.pos->ply();
    offset += counts[i];
  }
}

void BatchSearcher::collect(uint8_t* boards, uint8_t* to_move, int32_t* history,
                            int32_t* move_number, uint8_t* active) {
  const int n = num_cells();
  auto work = [&](int i) {
    Slot& slot = slots_[i];

    // 这一轮该槽位最多再跑多少次模拟
    const int budget = slot.active ? std::min(leaves_, sims_ - slot.sims_done) : 0;
    slot.pending_count = 0;

    for (int j = 0; j < budget; ++j) {
      const size_t row = static_cast<size_t>(i) * leaves_ + j;
      const size_t rbase = row * n;

      // leaves_ == 1 时走原来那条精确路径，**一个字节都不碰 virtual loss**。
      const Descent d = leaves_ == 1
                            ? slot.tree->descend(*slot.pos)
                            : slot.tree->descend_with_virtual_loss(*slot.pos);

      // 先判重再记录。树还没长开时（例如根节点尚未展开）连续下潜会拿到
      // 同一个叶子 —— 那次模拟对搜索没有任何贡献，**不能计数**，
      // 否则同样的 sims 下批量会比逐叶少算几次（实测少 1 次，被测试抓到）。
      bool repeated = false;
      for (int k = 0; k < slot.pending_count; ++k) {
        if (slot.pending[k].leaf == d.leaf) { repeated = true; break; }
      }
      const int depth_now = slot.pos->ply() - slot.root_ply;
      if (repeated) {
        for (int k = 0; k < depth_now; ++k) slot.pos->undo();
        slot.tree->undo_virtual_loss(d.leaf);   // 不用它，就得把虚拟败绩撤掉
        break;
      }

      Pending& p = slot.pending[slot.pending_count];
      p.leaf = d.leaf;
      p.needs_eval = d.needs_eval;
      p.terminal_value = d.terminal_value;
      // 展开要用的合法着法必须当场取 —— 下面就把局面回退了。
      p.move_count = d.needs_eval ? slot.pos->legal_moves(p.moves.data()) : 0;

      slot.pos->copy_grid_to(boards + rbase);
      to_move[row] = slot.pos->to_move();
      slot.pos->history(history + row * HISTORY_PLANES);
      move_number[row] = slot.pos->ply();
      active[row] = 1;
      ++slot.pending_count;

      // 回退到根，下一次下潜重新从根出发
      const int depth = slot.pos->ply() - slot.root_ply;
      for (int k = 0; k < depth; ++k) slot.pos->undo();

      // 终局叶子不会被展开，再下潜还是它 —— 这一轮到此为止。
      if (!d.needs_eval) break;
    }

    // 没用上的行仍要填出合法输入，保持批形状不变（CUDA Graph 那条设计约束）
    for (int j = slot.pending_count; j < leaves_; ++j) {
      const size_t row = static_cast<size_t>(i) * leaves_ + j;
      std::memset(boards + row * n, EMPTY, static_cast<size_t>(n));
      to_move[row] = BLACK;
      for (int k = 0; k < HISTORY_PLANES; ++k) {
        history[row * HISTORY_PLANES + k] = -1;
      }
      move_number[row] = 0;
      active[row] = 0;
    }
  };
  pool_->run(work, capacity());
}

void BatchSearcher::apply(const float* policy, const float* value) {
  const int n = num_cells();
  auto work = [&](int i) {
    Slot& slot = slots_[i];
    if (!slot.active) return;
    const bool clear_virtual = leaves_ > 1;

    for (int j = 0; j < slot.pending_count; ++j) {
      const Pending& p = slot.pending[j];
      const size_t row = static_cast<size_t>(i) * leaves_ + j;

      if (p.needs_eval && !slot.tree->expanded(p.leaf)) {
        slot.tree->expand_and_backup_moves(p.leaf, policy + row * n, value[row],
                                           p.moves.data(), p.move_count,
                                           clear_virtual);
      } else {
        // 两种情况走这里：终局叶子；以及**这一轮里已经被展开过的同一个叶子**。
        // 后者重复展开会把子节点池写坏，所以只回传。
        const float v = p.needs_eval ? value[row] : p.terminal_value;
        slot.tree->backup_terminal(p.leaf, v, clear_virtual);
      }
      ++slot.sims_done;
    }
    slot.pending_count = 0;
    // 局面在 collect 里就已经回退到根了，这里不需要再 undo。

    if (slot.sims_done >= sims_) {
      slot.tree->root_visit_counts(slot.visits.data());
    }
  };
  pool_->run(work, capacity());
}

int64_t BatchSearcher::virtual_outstanding() const {
  int64_t total = 0;
  for (const Slot& slot : slots_) {
    if (slot.tree) total += slot.tree->virtual_outstanding();
  }
  return total;
}

bool BatchSearcher::done() const {
  for (const Slot& slot : slots_) {
    if (slot.active && slot.sims_done < sims_) return false;
  }
  return true;
}

void BatchSearcher::visit_counts(int32_t* out) const {
  const int n = num_cells();
  for (int i = 0; i < capacity(); ++i) {
    int32_t* row = out + static_cast<size_t>(i) * n;
    const Slot& slot = slots_[i];
    if (!slot.active) {
      std::fill(row, row + n, 0);
      continue;
    }
    // 直接从树里现取，不读 slot.visits —— 后者只在跑满 sims 的那一刻回填，
    // 中途调用会读到上一次搜索留下的旧值，而那种错不报任何异常。
    slot.tree->root_visit_counts(row);
  }
}

void BatchSearcher::root_values(float* out) const {
  for (int i = 0; i < capacity(); ++i) {
    const Slot& slot = slots_[i];
    out[i] = slot.active ? slot.tree->root_value() : 0.0f;
  }
}

void BatchSearcher::best_moves(int32_t* out) const {
  const int n = num_cells();
  for (int i = 0; i < capacity(); ++i) {
    const Slot& slot = slots_[i];
    if (!slot.active) {
      out[i] = -1;
      continue;
    }
    // 评测要的是确定性的最强手：按访问数取 argmax，不做温度采样
    int best = -1;
    int best_visits = -1;
    for (int m = 0; m < n; ++m) {
      if (slot.visits[m] > best_visits) {
        best_visits = slot.visits[m];
        best = m;
      }
    }
    // 搜索一次都没展开过（例如根节点即终局）时退回第一个合法点
    if (best_visits <= 0) {
      for (int m = 0; m < n; ++m) {
        if (slot.pos->is_legal(m)) {
          best = m;
          break;
        }
      }
    }
    out[i] = best;
  }
}

}  // namespace gi
