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
  BatchSearcher(int board_size, int sims, const MctsConfig& mcts,
                const RuleConfig& rules, ForbiddenSemantics semantics,
                int num_slots, int num_threads);
  ~BatchSearcher();

  int capacity() const { return static_cast<int>(slots_.size()); }
  int num_cells() const { return size_ * size_; }
  int sims() const { return sims_; }

  // 载入一批局面。moves 是各局着法序列首尾相接的扁平数组，counts 给出各自长度。
  void set_positions(const int32_t* moves, const int32_t* counts, int count);

  // 与 SelfPlayRunner 相同的收集/回填协议。active 标记哪些槽位真的在搜。
  void collect(uint8_t* boards, uint8_t* to_move, int32_t* history,
               int32_t* move_number, uint8_t* active);
  void apply(const float* policy, const float* value);

  // 全部活跃槽位都跑够 sims 了吗
  bool done() const;

  // 各槽位按根节点访问数取的最佳着法；非活跃槽位写 -1
  void best_moves(int32_t* out) const;

 private:
  struct Slot {
    std::unique_ptr<Position> pos;
    std::unique_ptr<MctsTree> tree;
    std::vector<int32_t> visits;
    int root_ply = 0;
    int sims_done = 0;
    int descent_depth = 0;
    int leaf = -1;
    bool active = false;
    bool leaf_needs_eval = false;
    float leaf_terminal_value = 0.0f;
  };

  int size_;
  int sims_;
  int active_count_ = 0;
  std::unique_ptr<Rules> rules_;
  std::vector<Slot> slots_;
  std::unique_ptr<ThreadPool> pool_;
};

}  // namespace gi
