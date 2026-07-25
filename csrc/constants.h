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

// 棋型等级：某个空点落子后能形成的最高棋型，用作辅助监督标签。
// 取值必须与 gomoku_instinct/rules/constants.py 的 Level 一致。
//
// 注意这里的「活三」用的是**非递归**判定（只看能否一手成活四），
// 与三三禁手里那个递归定义不同 —— 它只是个特征标签，不参与规则判定；
// 精确的禁手规则由单独的 forbidden 头去学。
enum class Level : uint8_t {
  NONE = 0,
  CLOSED_THREE = 1,  // 眠三：再一手可成四，但成不了活四
  OPEN_THREE = 2,    // 活三：再一手可成活四
  FOUR = 3,          // 冲四：只有一个成五点
  OPEN_FOUR = 4,     // 活四：两个成五点
  FIVE = 5,          // 五连
  OVERLINE = 6,      // 长连（黑方为禁手；白方按成五算）
};
constexpr int NUM_LEVELS = 7;

struct Judgment {
  Outcome outcome = Outcome::ONGOING;
  Forbidden forbidden = Forbidden::NONE;
  uint8_t fours = 0;
  uint8_t open_threes = 0;
  uint8_t longest_run = 0;

  bool is_forbidden() const { return forbidden != Forbidden::NONE; }
};

}  // namespace gi
