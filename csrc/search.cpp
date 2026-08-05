#include "search.h"

#include <algorithm>
#include <cstring>

#include "thread_pool.h"

namespace gi {

BatchSearcher::BatchSearcher(int board_size, int sims, const MctsConfig& mcts,
                             const RuleConfig& rules,
                             ForbiddenSemantics semantics, int num_slots,
                             int num_threads)
    : size_(board_size), sims_(sims) {
  rules_ = std::make_unique<Rules>(rules);

  MctsConfig cfg = mcts;
  cfg.max_nodes = sims + 8;  // 每次搜索重建一棵树，节点池只需覆盖单次搜索

  const int n = num_cells();
  slots_.resize(num_slots);
  for (Slot& slot : slots_) {
    slot.pos = std::make_unique<Position>(board_size, rules_.get(), semantics);
    slot.tree = std::make_unique<MctsTree>(n, cfg);
    slot.visits.resize(n);
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
    slot.descent_depth = 0;
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
    const size_t base = static_cast<size_t>(i) * n;

    if (!slot.active || slot.sims_done >= sims_) {
      // 非活跃槽位仍要填出合法输入，保持批形状不变
      std::memset(boards + base, EMPTY, static_cast<size_t>(n));
      to_move[i] = BLACK;
      for (int k = 0; k < HISTORY_PLANES; ++k) {
        history[static_cast<size_t>(i) * HISTORY_PLANES + k] = -1;
      }
      move_number[i] = 0;
      active[i] = 0;
      return;
    }

    const Descent d = slot.tree->descend(*slot.pos);
    slot.leaf = d.leaf;
    slot.leaf_needs_eval = d.needs_eval;
    slot.leaf_terminal_value = d.terminal_value;
    slot.descent_depth = slot.pos->ply() - slot.root_ply;

    slot.pos->copy_grid_to(boards + base);
    to_move[i] = slot.pos->to_move();
    slot.pos->history(history + static_cast<size_t>(i) * HISTORY_PLANES);
    move_number[i] = slot.pos->ply();
    active[i] = 1;
  };
  pool_->run(work, capacity());
}

void BatchSearcher::apply(const float* policy, const float* value) {
  const int n = num_cells();
  auto work = [&](int i) {
    Slot& slot = slots_[i];
    if (!slot.active || slot.sims_done >= sims_) return;

    if (slot.leaf_needs_eval) {
      slot.tree->expand_and_backup(
          slot.leaf, policy + static_cast<size_t>(i) * n, value[i], *slot.pos);
    } else {
      slot.tree->backup_terminal(slot.leaf, slot.leaf_terminal_value);
    }

    for (int k = 0; k < slot.descent_depth; ++k) slot.pos->undo();
    slot.descent_depth = 0;
    ++slot.sims_done;

    if (slot.sims_done >= sims_) {
      slot.tree->root_visit_counts(slot.visits.data());
    }
  };
  pool_->run(work, capacity());
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
