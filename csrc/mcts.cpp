#include "mcts.h"

#include <algorithm>
#include <cmath>

namespace gi {

MctsTree::MctsTree(int num_cells, const MctsConfig& cfg)
    : num_cells_(num_cells), cfg_(cfg) {
  const size_t nodes = static_cast<size_t>(cfg.max_nodes);
  node_visits_.resize(nodes);
  node_virtual_.resize(nodes);
  node_value_sum_.resize(nodes);
  node_expanded_.resize(nodes);
  node_child_start_.resize(nodes);
  node_child_count_.resize(nodes);
  node_parent_.resize(nodes);
  node_parent_slot_.resize(nodes);

  // 一次性按最坏情况预留子节点数组，之后不再扩容 ——
  // 搜索过程中数组不会重新分配，指针和下标都保持有效。
  const size_t children = nodes * static_cast<size_t>(num_cells);
  child_move_.resize(children);
  child_prior_.resize(children);
  child_visits_.resize(children);
  child_virtual_.resize(children);
  child_value_sum_.resize(children);
  child_node_.resize(children);

  scratch_moves_.resize(num_cells);
  clear();
}

void MctsTree::clear() {
  node_count_ = 1;
  child_count_ = 0;
  root_ = 0;
  node_visits_[0] = 0;
  node_virtual_[0] = 0;
  node_value_sum_[0] = 0.0f;
  node_expanded_[0] = 0;
  node_child_start_[0] = 0;
  node_child_count_[0] = 0;
  node_parent_[0] = -1;
  node_parent_slot_[0] = -1;
}

float MctsTree::root_value() const {
  const int visits = node_visits_[root_];
  return visits > 0 ? node_value_sum_[root_] / static_cast<float>(visits) : 0.0f;
}

int MctsTree::new_node(int parent, int parent_child_slot) {
  if (node_count_ >= cfg_.max_nodes) return -1;
  const int idx = node_count_++;
  node_visits_[idx] = 0;
  node_virtual_[idx] = 0;
  node_value_sum_[idx] = 0.0f;
  node_expanded_[idx] = 0;
  node_child_start_[idx] = 0;
  node_child_count_[idx] = 0;
  node_parent_[idx] = parent;
  node_parent_slot_[idx] = parent_child_slot;
  return idx;
}

int MctsTree::select_child(int node) const {
  const int start = node_child_start_[node];
  const int count = node_child_count_[node];
  if (count == 0) return -1;

  // 虚拟访问一并算进来。**全为 0 时下面每一步的算术与不带 virtual loss 时
  // 逐位相同**（整数加 0、浮点减 0.0f 都是精确的），自博弈因此完全不受影响。
  const int parent_n = node_visits_[node] + node_virtual_[node];
  const float parent_visits = static_cast<float>(parent_n);
  const float sqrt_parent = std::sqrt(std::max(parent_visits, 1.0f));

  // 随访问数增长的 c_puct，让搜索后期更偏向利用。
  const float c_puct =
      cfg_.c_puct +
      std::log((parent_visits + cfg_.c_puct_base + 1.0f) / cfg_.c_puct_base);

  // FPU：未访问过的子节点用「父节点当前评估 - 已探索先验占比的折减」作初值，
  // 避免一上来就把先验小的着法全部试一遍。
  float explored_prior = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (child_visits_[start + i] + child_virtual_[start + i] > 0) {
      explored_prior += child_prior_[start + i];
    }
  }
  // 每个虚拟访问按一次失败计（价值 -1），下一次下潜就会绕开这条路径。
  const float parent_q =
      parent_n > 0 ? (node_value_sum_[node] -
                      static_cast<float>(node_virtual_[node])) /
                         static_cast<float>(parent_n)
                   : 0.0f;
  const float fpu = parent_q - cfg_.fpu_reduction * std::sqrt(explored_prior);

  int best = -1;
  float best_score = -1e30f;
  for (int i = 0; i < count; ++i) {
    const int slot = start + i;
    const int visits = child_visits_[slot] + child_virtual_[slot];
    const float q =
        visits > 0 ? (child_value_sum_[slot] -
                      static_cast<float>(child_virtual_[slot])) /
                         static_cast<float>(visits)
                   : fpu;
    const float u = c_puct * child_prior_[slot] * sqrt_parent /
                    (1.0f + static_cast<float>(visits));
    const float score = q + u;
    if (score > best_score) {
      best_score = score;
      best = slot;
    }
  }
  return best;
}

Descent MctsTree::descend(Position& pos) { return descend_impl(pos, false); }

Descent MctsTree::descend_with_virtual_loss(Position& pos) {
  return descend_impl(pos, true);
}

// 两条路径共用同一份选点逻辑。**不复制一份出来改** —— 复制品会随时间漂移，
// 而"对战时的搜索和自博弈时的搜索悄悄不一样了"是查不出来的那种错。
Descent MctsTree::descend_impl(Position& pos, bool virtual_loss) {
  Descent d;
  int node = root_;
  if (virtual_loss) node_virtual_[node] += 1;

  while (true) {
    if (pos.terminal()) {
      d.leaf = node;
      d.needs_eval = false;
      // to_move() 语义是「下一手轮到谁」，正好就是该节点的价值视角。
      d.terminal_value = pos.result_for(pos.to_move());
      return d;
    }
    if (node_expanded_[node] == 0) {
      d.leaf = node;
      d.needs_eval = true;
      return d;
    }

    const int slot = select_child(node);
    if (slot < 0) {  // 已展开但无合法着法，按和棋处理
      d.leaf = node;
      d.needs_eval = false;
      d.terminal_value = 0.0f;
      return d;
    }

    pos.play(child_move_[slot]);

    int child = child_node_[slot];
    if (child < 0) {
      child = new_node(node, slot);
      if (child < 0) {  // 节点池耗尽，就地回传，不再深入
        d.leaf = node;
        d.needs_eval = false;
        d.terminal_value = node_visits_[node] > 0
                               ? node_value_sum_[node] / node_visits_[node]
                               : 0.0f;
        // 这条边没走成，**不能**给它记虚拟访问 —— 回传是从 node 往上走的，
        // 不会经过这条边，记了就永远减不回来。
        return d;
      }
      child_node_[slot] = child;
    }
    if (virtual_loss) {
      child_virtual_[slot] += 1;
      node_virtual_[child] += 1;
    }
    node = child;
  }
}

void MctsTree::expand_and_backup(int leaf, const float* priors, float value,
                                 const Position& pos_at_leaf,
                                 bool clear_virtual) {
  const int count = pos_at_leaf.legal_moves(scratch_moves_.data());
  expand_and_backup_moves(leaf, priors, value, scratch_moves_.data(), count,
                          clear_virtual);
}

void MctsTree::expand_and_backup_moves(int leaf, const float* priors,
                                       float value, const int32_t* moves,
                                       int count, bool clear_virtual) {
  node_child_start_[leaf] = child_count_;
  node_child_count_[leaf] = static_cast<int16_t>(count);

  // 在真正的合法着法上重新归一化。调用方只按空点做了屏蔽，
  // 在 ILLEGAL 语义下禁手点也不合法，落到那些点上的概率质量要摊回来。
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += priors[moves[i]];
  const float scale = total > 1e-12f ? 1.0f / total : 0.0f;
  const float uniform = count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;

  for (int i = 0; i < count; ++i) {
    const int move = moves[i];
    const int slot = child_count_++;
    child_move_[slot] = static_cast<int16_t>(move);
    child_prior_[slot] = scale > 0.0f ? priors[move] * scale : uniform;
    child_visits_[slot] = 0;
    child_virtual_[slot] = 0;
    child_value_sum_[slot] = 0.0f;
    child_node_[slot] = -1;
  }
  node_expanded_[leaf] = 1;
  backup_terminal(leaf, value, clear_virtual);
}

void MctsTree::backup_terminal(int leaf, float value, bool clear_virtual) {
  int node = leaf;
  float v = value;  // 当前节点行棋方视角
  while (true) {
    node_visits_[node] += 1;
    node_value_sum_[node] += v;
    // 下潜时沿这条路径记的虚拟败绩，在这里原路减回去。
    if (clear_virtual && node_virtual_[node] > 0) node_virtual_[node] -= 1;

    const int parent = node_parent_[node];
    if (parent < 0) break;

    // 子节点的统计量挂在父节点上，视角要翻到父节点行棋方。
    const int slot = node_parent_slot_[node];
    child_visits_[slot] += 1;
    child_value_sum_[slot] += -v;
    if (clear_virtual && child_virtual_[slot] > 0) child_virtual_[slot] -= 1;

    v = -v;
    node = parent;
  }
}

void MctsTree::undo_virtual_loss(int leaf) {
  int node = leaf;
  while (true) {
    if (node_virtual_[node] > 0) node_virtual_[node] -= 1;
    const int parent = node_parent_[node];
    if (parent < 0) break;
    const int slot = node_parent_slot_[node];
    if (child_virtual_[slot] > 0) child_virtual_[slot] -= 1;
    node = parent;
  }
}

int64_t MctsTree::virtual_outstanding() const {
  int64_t total = 0;
  for (int i = 0; i < node_count_; ++i) total += node_virtual_[i];
  for (int i = 0; i < child_count_; ++i) total += child_virtual_[i];
  return total;
}

void MctsTree::root_visit_counts(int32_t* out) const {
  std::fill(out, out + num_cells_, 0);
  const int start = node_child_start_[root_];
  const int count = node_child_count_[root_];
  for (int i = 0; i < count; ++i) {
    out[child_move_[start + i]] = child_visits_[start + i];
  }
}

void MctsTree::root_child_values(float* out) const {
  std::fill(out, out + num_cells_, -1.0f);
  const int start = node_child_start_[root_];
  const int count = node_child_count_[root_];
  for (int i = 0; i < count; ++i) {
    const int slot = start + i;
    if (child_visits_[slot] > 0) {
      out[child_move_[slot]] =
          child_value_sum_[slot] / static_cast<float>(child_visits_[slot]);
    }
  }
}

void MctsTree::add_root_noise(const float* noise, float eps) {
  const int start = node_child_start_[root_];
  const int count = node_child_count_[root_];
  for (int i = 0; i < count; ++i) {
    const int slot = start + i;
    child_prior_[slot] = (1.0f - eps) * child_prior_[slot] + eps * noise[i];
  }
}

void MctsTree::advance_root(int move) {
  // 目前每手重建一棵树：不做子树复用，换来的是节点池占用恒定、实现简单。
  // 子树复用是后续可选的优化项。
  (void)move;
  clear();
}

}  // namespace gi
