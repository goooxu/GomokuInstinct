#include "mcts.h"

#include <algorithm>
#include <cmath>

namespace gi {

MctsTree::MctsTree(int num_cells, const MctsConfig& cfg)
    : num_cells_(num_cells), cfg_(cfg) {
  const size_t nodes = static_cast<size_t>(cfg.max_nodes);
  node_visits_.resize(nodes);
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

  const float parent_visits = static_cast<float>(node_visits_[node]);
  const float sqrt_parent = std::sqrt(std::max(parent_visits, 1.0f));

  // 随访问数增长的 c_puct，让搜索后期更偏向利用。
  const float c_puct =
      cfg_.c_puct +
      std::log((parent_visits + cfg_.c_puct_base + 1.0f) / cfg_.c_puct_base);

  // FPU：未访问过的子节点用「父节点当前评估 - 已探索先验占比的折减」作初值，
  // 避免一上来就把先验小的着法全部试一遍。
  float explored_prior = 0.0f;
  for (int i = 0; i < count; ++i) {
    if (child_visits_[start + i] > 0) explored_prior += child_prior_[start + i];
  }
  const float parent_q =
      node_visits_[node] > 0
          ? node_value_sum_[node] / static_cast<float>(node_visits_[node])
          : 0.0f;
  const float fpu = parent_q - cfg_.fpu_reduction * std::sqrt(explored_prior);

  int best = -1;
  float best_score = -1e30f;
  for (int i = 0; i < count; ++i) {
    const int slot = start + i;
    const int visits = child_visits_[slot];
    const float q =
        visits > 0 ? child_value_sum_[slot] / static_cast<float>(visits) : fpu;
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

Descent MctsTree::descend(Position& pos) {
  Descent d;
  int node = root_;

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
        return d;
      }
      child_node_[slot] = child;
    }
    node = child;
  }
}

void MctsTree::expand_and_backup(int leaf, const float* priors, float value,
                                 const Position& pos_at_leaf) {
  const int count = pos_at_leaf.legal_moves(scratch_moves_.data());
  node_child_start_[leaf] = child_count_;
  node_child_count_[leaf] = static_cast<int16_t>(count);

  // 在真正的合法着法上重新归一化。调用方只按空点做了屏蔽，
  // 在 ILLEGAL 语义下禁手点也不合法，落到那些点上的概率质量要摊回来。
  float total = 0.0f;
  for (int i = 0; i < count; ++i) total += priors[scratch_moves_[i]];
  const float scale = total > 1e-12f ? 1.0f / total : 0.0f;
  const float uniform = count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;

  for (int i = 0; i < count; ++i) {
    const int move = scratch_moves_[i];
    const int slot = child_count_++;
    child_move_[slot] = static_cast<int16_t>(move);
    child_prior_[slot] = scale > 0.0f ? priors[move] * scale : uniform;
    child_visits_[slot] = 0;
    child_value_sum_[slot] = 0.0f;
    child_node_[slot] = -1;
  }
  node_expanded_[leaf] = 1;
  backup_terminal(leaf, value);
}

void MctsTree::backup_terminal(int leaf, float value) {
  int node = leaf;
  float v = value;  // 当前节点行棋方视角
  while (true) {
    node_visits_[node] += 1;
    node_value_sum_[node] += v;

    const int parent = node_parent_[node];
    if (parent < 0) break;

    // 子节点的统计量挂在父节点上，视角要翻到父节点行棋方。
    const int slot = node_parent_slot_[node];
    child_visits_[slot] += 1;
    child_value_sum_[slot] += -v;

    v = -v;
    node = parent;
  }
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
