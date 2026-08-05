// 单局的 PUCT 搜索树。
//
// 树节点用扁平数组存，子节点的统计量（先验、访问数、价值和）挂在父节点上 ——
// 这是 AlphaZero 类实现的常规布局，选点时能顺序扫过一段连续内存。
//
// **自博弈不需要 virtual loss**：那里每局每轮只下潜一次，同一棵树上不存在并发下潜，
// 因此可以做精确的 PUCT 而不引入任何近似。这条路径（`descend` / 不带 clear_virtual 的
// `backup_terminal`）**一行都没有为批量搜索改动过**。
//
// 但**对战时只有一个局面**，"靠同时跑上千局来填批"这条前提不成立：一轮只产一个叶子，
// GPU 每次只算一个 15×15 的输入，利用率实测约 1.7%，其余全耗在 kernel 启动开销上。
// 所以另外提供一条 opt-in 的路径（`descend_with_virtual_loss`）：沿途记一笔虚拟败绩，
// 逼下一次下潜岔开，一轮凑出多个叶子一起送 GPU。
//
// **虚拟访问全为 0 时，select_child 的算术与改动前逐位相同**（整数加 0、浮点减 0.0f
// 都是精确的），自博弈因此完全不受影响 —— 这一点有快照测试钉着。
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

  // 从根下潜，沿途给经过的节点与子节点各记一笔**虚拟败绩**。
  // 与 `descend` 的选点逻辑完全相同，区别只在于它会改变后续下潜看到的统计量。
  // 每次调用都必须有一次对应的 `backup_terminal(..., clear_virtual=true)`
  // 或 `expand_and_backup(..., clear_virtual=true)` 把它减回去，否则会泄漏。
  Descent descend_with_virtual_loss(Position& pos);

  // 用网络给出的先验展开叶子，并把价值沿路径回传。
  // priors 长度为 num_cells，调用方已做过合法性屏蔽与归一化。
  void expand_and_backup(int leaf, const float* priors, float value,
                         const Position& pos_at_leaf, bool clear_virtual = false);
  // 同上，但合法着法由调用方给出。批量收集时局面已经回退了，拿不到 pos_at_leaf。
  void expand_and_backup_moves(int leaf, const float* priors, float value,
                               const int32_t* moves, int count,
                               bool clear_virtual = false);
  // 终局叶子：不展开，只回传。
  void backup_terminal(int leaf, float value, bool clear_virtual = false);

  // 叶子是否已经展开过。一轮里两次下潜可能落到同一个叶子，
  // 重复展开会把子节点池写坏，调用方据此跳过。
  bool expanded(int node) const { return node_expanded_[node] != 0; }

  // 撤掉一次下潜留下的虚拟败绩，但**不计入任何统计量**。
  // 用于"这次下潜取到的叶子本轮已经有了、不打算用它"的情形 ——
  // 不撤就会泄漏，泄漏了搜索会一直绕开那条路径而且不报任何错。
  void undo_virtual_loss(int leaf);

  // 审计：还有多少虚拟访问没有被减回去。一轮结束后必须为 0。
  // 这类"悄悄不归零"的错不报异常，只会让搜索慢慢变差 —— 正是第 11 章那一类。
  int64_t virtual_outstanding() const;

  // 根节点各着法的访问数，写入长度 num_cells 的数组。
  void root_visit_counts(int32_t* out) const;
  // 根节点各着法的 Q（父节点视角）。未访问过的着法写 -1，按最坏情况算。
  void root_child_values(float* out) const;
  // 给根节点先验混入 Dirichlet 噪声。
  void add_root_noise(const float* noise, float eps);
  bool root_expanded() const { return node_expanded_[root_] != 0; }
  int root_child_count() const { return node_child_count_[root_]; }

  // 走一步之后把对应子树提为新根（子树复用）。找不到就重置。
  void advance_root(int move);

  int node_count() const { return node_count_; }

 private:
  Descent descend_impl(Position& pos, bool virtual_loss);
  int select_child(int node) const;
  int new_node(int parent, int parent_child_slot);

  int num_cells_;
  MctsConfig cfg_;
  int root_ = 0;
  int node_count_ = 0;

  // 节点级
  std::vector<int32_t> node_visits_;
  std::vector<int32_t> node_virtual_;
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
  std::vector<int32_t> child_virtual_;
  std::vector<float> child_value_sum_;
  std::vector<int32_t> child_node_;
  int child_count_ = 0;

  std::vector<int32_t> scratch_moves_;
};

}  // namespace gi
