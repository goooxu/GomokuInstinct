// 连珠规则判定（高性能实现）。
//
// 与 gomoku_instinct/rules/ 下的 Python 参考实现同规则、不同做法：
// 这里把每条直线打包成位掩码，用窗口位运算与 popcount 完成棋型判定；
// Python 那份则是抽出定长列表逐格扫描。两者的一致性由差分测试锁定。
#pragma once

#include <atomic>
#include <cstdint>

#include "constants.h"

namespace gi {

class Rules {
 public:
  explicit Rules(const RuleConfig& cfg = RuleConfig()) : cfg_(cfg) {}

  // 判定在空点 move 落 color 子的结果。grid 在调用前后保持不变。
  Judgment judge(const uint8_t* grid, int size, int move, uint8_t color) const;

  // 全盘黑方禁手点标记；out 长度 size*size，非空点写 0。
  void forbidden_map(const uint8_t* grid, int size, uint8_t* out) const;

  // 逐点棋型标注：out[i] 为 color 在空点 i 落子后能形成的最高棋型等级（Level）。
  // 非空点写 0。这是辅助监督头的标签来源，全部由规则导出，不含任何棋谱知识。
  void pattern_map(const uint8_t* grid, int size, uint8_t color,
                   uint8_t* out) const;

  // 逐点判定；out 长度 5*size*size，每格依次为
  // [outcome, forbidden, fours, open_threes, longest_run]，非空点全部填 255。
  void judge_all(const uint8_t* grid, int size, uint8_t color, uint8_t* out) const;

  const RuleConfig& config() const { return cfg_; }

  // 递归封顶发生的次数。应恒为 0，非零说明深度上限设小了。
  int64_t depth_exceeded() const {
    return depth_exceeded_.load(std::memory_order_relaxed);
  }
  // 实际达到过的最大递归深度，用来验证上限是否合理。
  int max_depth() const { return max_depth_.load(std::memory_order_relaxed); }
  void reset_counters() {
    depth_exceeded_.store(0, std::memory_order_relaxed);
    max_depth_.store(0, std::memory_order_relaxed);
  }

 private:
  // grid 中 move 处已经放上了 color 子。
  Judgment judge_placed(uint8_t* grid, int size, int move, uint8_t color,
                        int depth) const;
  bool has_open_three(uint8_t* grid, int size, int move, int dir,
                      int depth) const;

  RuleConfig cfg_;
  mutable std::atomic<int64_t> depth_exceeded_{0};
  mutable std::atomic<int> max_depth_{0};
};

}  // namespace gi
