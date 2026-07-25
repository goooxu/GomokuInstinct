// 单局的 PUCT 搜索树。
//
// 树节点用扁平数组存，子节点的统计量（先验、访问数、价值和）挂在父节点上 ——
// 这是 AlphaZero 类实现的常规布局，选点时能顺序扫过一段连续内存。
//
// 这里**不需要 virtual loss**：向量化自博弈里每局每轮只下潜一次，
// 同一棵树上不存在并发下潜，因此可以做精确的 PUCT 而不引入任何近似。
#pragma once

#include <cstdint>
#include <vector>

#include "constants.h"
#include "position.h"

namespace gi {

struct MctsConfig {
  float c_puct = 1.6f;
  float c_puct_base = 19652.0f;
  float fpu_reduction = 0.25f;  // 未访问子节点的先验价值折减
  int max_nodes = 8192;
};

// 一次下潜的结果。
struct Descent {
  int leaf = -1;          // 叶节点编号
  bool needs_eval = false;  // 是否需要网络评估（终局叶子不需要）
  float terminal_value = 0.0f;  // 终局叶子的价值（叶节点行棋方视角）
};

class MctsTree {
 public:
  MctsTree(int num_cells, const MctsConfig& cfg);

  void clear();
  bool empty() const { return node_count_ == 0; }
  int root() const { return root_; }
  int root_visits() const { return node_visits_[root_]; }
  float root_value() const;

  // 从根下潜到一个叶子；沿途落子写进 pos，调用方负责在 apply 之后回退。
  Descent descend(Position& pos);

  // 用网络给出的先验展开叶子，并把价值沿路径回传。
  // priors 长度为 num_cells，调用方已做过合法性屏蔽与归一化。
  void expand_and_backup(int leaf, const float* priors, float value,
                         const Position& pos_at_leaf);
  // 终局叶子：不展开，只回传。
  void backup_terminal(int leaf, float value);

  // 根节点各着法的访问数，写入长度 num_cells 的数组。
  void root_visit_counts(int32_t* out) const;
  // 给根节点先验混入 Dirichlet 噪声。
  void add_root_noise(const float* noise, float eps);
  bool root_expanded() const { return node_expanded_[root_] != 0; }
  int root_child_count() const { return node_child_count_[root_]; }

  // 走一步之后把对应子树提为新根（子树复用）。找不到就重置。
  void advance_root(int move);

  int node_count() const { return node_count_; }

 private:
  int select_child(int node) const;
  int new_node(int parent, int parent_child_slot);

  int num_cells_;
  MctsConfig cfg_;
  int root_ = 0;
  int node_count_ = 0;

  // 节点级
  std::vector<int32_t> node_visits_;
  std::vector<float> node_value_sum_;
  std::vector<uint8_t> node_expanded_;
  std::vector<int32_t> node_child_start_;
  std::vector<int16_t> node_child_count_;
  std::vector<int32_t> node_parent_;
  std::vector<int32_t> node_parent_slot_;

  // 子节点级（按 node_child_start_ 分段）
  std::vector<int16_t> child_move_;
  std::vector<float> child_prior_;
  std::vector<int32_t> child_visits_;
  std::vector<float> child_value_sum_;
  std::vector<int32_t> child_node_;
  int child_count_ = 0;

  std::vector<int32_t> scratch_moves_;
};

}  // namespace gi
