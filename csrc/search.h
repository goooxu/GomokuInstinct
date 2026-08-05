// 对任意给定局面做批量 MCTS 搜索。
//
// 与 SelfPlayRunner 的区别：那个从空盘自己往下走，这个由调用方指定要搜哪些局面。
// 用途是测量本项目的头号指标 —— 同一份权重下，零搜索策略与它自己的 MCTS 版本
// 的 Elo 差。整个方案赌的就是"能把搜索压进权重里"，这个差值是唯一能直接量化它的数。
//
// 局面按**完整着法序列**载入而非直接摆棋盘：网络输入含最近数手的落点平面，
// 只给棋盘状态的话这几个平面会全空，测出来的就不是同一个网络在同一个输入上的表现。
//
// 评测场景不加 Dirichlet 噪声、按访问数取 argmax —— 要的是确定性的最强手。
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "constants.h"
#include "mcts.h"
#include "position.h"
#include "renju.h"
#include "thread_pool.h"

namespace gi {

class BatchSearcher {
 public:
  // leaves_per_slot：一轮从**同一棵树**里最多取几个叶子凑成一批。
  // 1 = 逐个叶子的精确搜索（评测与竞技场的口径，报告里所有数字都是这个）。
  // 大于 1 时用 virtual loss 把后续下潜逼开 —— 对战时只有一个局面，
  // 不这样做 GPU 每次只能算一个输入，利用率实测约 1.7%。
  BatchSearcher(int board_size, int sims, const MctsConfig& mcts,
                const RuleConfig& rules, ForbiddenSemantics semantics,
                int num_slots, int num_threads, int leaves_per_slot = 1);
  ~BatchSearcher();

  int capacity() const { return static_cast<int>(slots_.size()); }
  int leaves_per_slot() const { return leaves_; }
  // 收集/回填缓冲区的行数。leaves_per_slot = 1 时就等于 capacity()。
  int batch_rows() const { return capacity() * leaves_; }
  // 审计：还有多少虚拟访问没被减回去。**一轮结束后必须为 0。**
  int64_t virtual_outstanding() const;
  int num_cells() const { return size_ * size_; }
  int sims() const { return sims_; }

  // 载入一批局面。moves 是各局着法序列首尾相接的扁平数组，counts 给出各自长度。
  void set_positions(const int32_t* moves, const int32_t* counts, int count);

  // 收集/回填协议。缓冲区按 **batch_rows() 行**给，第 i 个槽位的第 j 个叶子
  // 落在第 i * leaves_per_slot + j 行。active 标记哪些行真的有叶子 ——
  // 树还没长开时一轮可能只产得出一个叶子，其余行是空的。
  void collect(uint8_t* boards, uint8_t* to_move, int32_t* history,
               int32_t* move_number, uint8_t* active);
  void apply(const float* policy, const float* value);

  // 全部活跃槽位都跑够 sims 了吗
  bool done() const;

  // 各槽位按根节点访问数取的最佳着法；非活跃槽位写 -1
  void best_moves(int32_t* out) const;

  // 根节点各着法的访问数，按 capacity × num_cells 行主序写出；非活跃槽位整行写 0。
  // 部署端要用它显示"搜索到底看了哪些点、各看了多少次" —— 那是搜索模式下
  // 与零搜索的策略概率对应的东西。
  void visit_counts(int32_t* out) const;

  // 各槽位根节点的价值（行棋方视角）；非活跃槽位写 0。
  void root_values(float* out) const;

 private:
  // 一轮里从某个槽位收集到的一个叶子。局面在收集时就回退了，
  // 所以展开需要的合法着法必须当场存下来 —— 回填时已经取不到那个局面了。
  struct Pending {
    int leaf = -1;
    bool needs_eval = false;
    float terminal_value = 0.0f;
    int move_count = 0;
    std::vector<int32_t> moves;
  };

  struct Slot {
    std::unique_ptr<Position> pos;
    std::unique_ptr<MctsTree> tree;
    std::vector<int32_t> visits;
    std::vector<Pending> pending;   // 长度 leaves_
    int pending_count = 0;
    int root_ply = 0;
    int sims_done = 0;
    int leaf = -1;
    bool active = false;
  };

  int size_;
  int sims_;
  int leaves_ = 1;
  int active_count_ = 0;
  std::unique_ptr<Rules> rules_;
  std::vector<Slot> slots_;
  std::unique_ptr<ThreadPool> pool_;
};

}  // namespace gi
