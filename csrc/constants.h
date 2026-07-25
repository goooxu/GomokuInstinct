// 规则层基础常量。取值必须与 gomoku_instinct/rules/constants.py 完全一致，
// 差分测试会连带校验这一点。
#pragma once

#include <cstdint>

namespace gi {

constexpr uint8_t EMPTY = 0;
constexpr uint8_t BLACK = 1;
constexpr uint8_t WHITE = 2;
constexpr uint8_t WALL = 3;

enum class Outcome : uint8_t {
  ONGOING = 0,
  BLACK_WIN = 1,
  WHITE_WIN = 2,
  DRAW = 3,
};

enum class Forbidden : uint8_t {
  NONE = 0,
  OVERLINE = 1,
  DOUBLE_FOUR = 2,
  DOUBLE_THREE = 3,
};

// 四个方向：横、竖、主对角、副对角。反向是同一条线，不重复计。
constexpr int NUM_DIRS = 4;
constexpr int DR[NUM_DIRS] = {0, 1, 1, 1};
constexpr int DC[NUM_DIRS] = {1, 0, 1, -1};

// 棋盘边长上限。一条线最长等于边长，用 uint32_t 打包位掩码，因此上限 32。
constexpr int MAX_SIZE = 32;

struct RuleConfig {
  bool forbidden_enabled = true;
  bool overline = true;
  bool double_four = true;
  bool double_three = true;
  bool five_overrides_forbidden = true;
  bool white_overline_wins = true;
  // 三三判定的递归深度上限。递归每深一层都会往盘上多放一颗子，所以必然终止；
  // 这个上限只是安全网，取大一些，真正的深度由 Rules::max_depth() 观测。
  int recursion_depth = 64;
};

struct Judgment {
  Outcome outcome = Outcome::ONGOING;
  Forbidden forbidden = Forbidden::NONE;
  uint8_t fours = 0;
  uint8_t open_threes = 0;
  uint8_t longest_run = 0;

  bool is_forbidden() const { return forbidden != Forbidden::NONE; }
};

}  // namespace gi
