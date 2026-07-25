#include "renju.h"

#include <cstring>
#include <stdexcept>

#include "geometry.h"

namespace gi {

namespace {

// 一条直线在某个颜色视角下的位掩码视图。
// 线外的格子不存在于掩码中 —— 窗口必须整体落在 [0, len) 内，越界因此自动排除。
struct LineView {
  uint32_t own = 0;      // 己方子
  uint32_t blocked = 0;  // 对方子（线外不参与）
  int len = 0;
  int pos = 0;   // 当前落点在线内的下标
  int line = 0;  // 线号，用于映射回格子编号
};

inline LineView make_line(const uint8_t* grid, const Geometry& g, int cell,
                          int dir, uint8_t color) {
  LineView lv;
  lv.line = g.line_id[dir][cell];
  lv.pos = g.line_pos[dir][cell];
  const std::vector<int16_t>& cells = g.lines[dir][lv.line];
  lv.len = static_cast<int>(cells.size());
  for (int j = 0; j < lv.len; ++j) {
    const uint8_t v = grid[cells[j]];
    if (v == color) {
      lv.own |= (1u << j);
    } else if (v != EMPTY) {
      lv.blocked |= (1u << j);
    }
  }
  return lv;
}

// 经过 pos 的同色连子长度。
inline int run_length(uint32_t own, int len, int pos) {
  int l = pos;
  while (l > 0 && ((own >> (l - 1)) & 1u)) --l;
  int r = pos;
  while (r + 1 < len && ((own >> (r + 1)) & 1u)) ++r;
  return r - l + 1;
}

// 在空点 gap 补一子是否成五。exact=true（黑方）要求恰好五连：
// 长连是禁手，既不算成五，也就不构成「四」。
inline bool completes_five(uint32_t own, int len, int gap, bool exact) {
  const uint32_t filled = own | (1u << gap);
  const int run = run_length(filled, len, gap);
  return exact ? (run == 5) : (run >= 5);
}

// 统计 pos 这一手在本方向造成的「四」的个数，**按四颗子的位掩码去重**。
//
// 去重是关键：活四 `.XXXX.` 的两个成五点共用同一组四颗子，只算一个四（合法）；
// 而 `X.XXX.X` 的两个四分别由 {0,2,3,4} 与 {2,3,4,6} 构成，是两个独立的四，
// 构成四四禁手。照窗口数或照成五点数都会把这两种情况混为一谈。
inline int count_fours(uint32_t own, uint32_t blocked, int len, int pos,
                       bool exact, bool* open_four) {
  uint32_t keys[5];
  int nkeys = 0;
  bool open = false;

  const int lo = pos - 4 < 0 ? 0 : pos - 4;
  for (int s = lo; s <= pos; ++s) {
    if (s + 4 >= len) break;  // s 递增，之后的窗口只会更靠右
    const uint32_t win = 0x1Fu << s;
    if (blocked & win) continue;
    const uint32_t mine = own & win;
    if (__builtin_popcount(mine) != 4) continue;

    // 窗口内既非己方子也非对方子的那一格必然为空。
    const uint32_t gapmask = win & ~mine;
    const int gap = __builtin_ctz(gapmask);
    if (!completes_five(own, len, gap, exact)) continue;

    int k = -1;
    for (int i = 0; i < nkeys; ++i) {
      if (keys[i] == mine) {
        k = i;
        break;
      }
    }
    if (k < 0) {
      keys[nkeys++] = mine;
    } else {
      // 同一组子落在两个窗口里，说明有两个不同的成五点 —— 这就是活四。
      open = true;
    }
  }
  if (open_four != nullptr) *open_four = open;
  return nkeys;
}

// 某个方向上、落子之后形成的最高棋型等级。
//
// 「活三」在这里用非递归判定（只看本方向能否一手成活四）：它是特征标签而非规则判定，
// 精确的三三禁手仍由 has_open_three 的递归版本负责。
uint8_t level_in_direction(uint32_t own, uint32_t blocked, int len, int pos,
                           bool black) {
  const int run = run_length(own, len, pos);
  if (run >= 6) {
    return static_cast<uint8_t>(black ? Level::OVERLINE : Level::FIVE);
  }
  if (run == 5) return static_cast<uint8_t>(Level::FIVE);

  const bool exact = black;  // 黑方必须恰好五连
  bool open_four = false;
  const int fours = count_fours(own, blocked, len, pos, exact, &open_four);
  if (fours > 0) {
    return static_cast<uint8_t>(open_four ? Level::OPEN_FOUR : Level::FOUR);
  }

  // 试探邻近空点：能一手成活四即为活三；只能成四则为眠三。
  bool can_make_four = false;
  const int lo = pos - 4 < 0 ? 0 : pos - 4;
  const int hi = pos + 4 >= len ? len - 1 : pos + 4;
  for (int p = lo; p <= hi; ++p) {
    const uint32_t bit = 1u << p;
    if ((own & bit) || (blocked & bit)) continue;
    const uint32_t own2 = own | bit;
    if (run_length(own2, len, p) >= 5) continue;  // 那是四，不是三
    bool o4 = false;
    const int f = count_fours(own2, blocked, len, p, exact, &o4);
    if (o4) return static_cast<uint8_t>(Level::OPEN_THREE);
    if (f > 0) can_make_four = true;
  }
  return static_cast<uint8_t>(can_make_four ? Level::CLOSED_THREE : Level::NONE);
}

}  // namespace

void Rules::pattern_map(const uint8_t* grid, int size, uint8_t color,
                        uint8_t* out) const {
  const int n = size * size;
  std::memset(out, 0, static_cast<size_t>(n));

  uint8_t buf[MAX_SIZE * MAX_SIZE];
  std::memcpy(buf, grid, static_cast<size_t>(n));
  const Geometry& g = geometry(size);
  const bool black = (color == BLACK);

  for (int i = 0; i < n; ++i) {
    if (buf[i] != EMPTY) continue;
    buf[i] = color;
    uint8_t best = 0;
    for (int d = 0; d < NUM_DIRS; ++d) {
      const LineView lv = make_line(buf, g, i, d, color);
      const uint8_t level =
          level_in_direction(lv.own, lv.blocked, lv.len, lv.pos, black);
      if (level > best) best = level;
    }
    buf[i] = EMPTY;
    out[i] = best;
  }
}

Judgment Rules::judge_placed(uint8_t* grid, int size, int move, uint8_t color,
                             int depth) const {
  // 记录实际达到的最大递归深度（松散更新即可，只用于审计）。
  int seen = max_depth_.load(std::memory_order_relaxed);
  while (depth > seen &&
         !max_depth_.compare_exchange_weak(seen, depth,
                                           std::memory_order_relaxed)) {
  }

  const Geometry& g = geometry(size);

  LineView lv[NUM_DIRS];
  int runs[NUM_DIRS];
  int longest = 0;
  for (int d = 0; d < NUM_DIRS; ++d) {
    lv[d] = make_line(grid, g, move, d, color);
    runs[d] = run_length(lv[d].own, lv[d].len, lv[d].pos);
    if (runs[d] > longest) longest = runs[d];
  }

  Judgment j;
  j.longest_run = static_cast<uint8_t>(longest);

  if (color == WHITE) {
    const bool wins = cfg_.white_overline_wins ? (longest >= 5) : (longest == 5);
    j.outcome = wins ? Outcome::WHITE_WIN : Outcome::ONGOING;
    return j;
  }

  // 关闭禁手即退化为自由五子棋，黑白规则对称。
  if (!cfg_.forbidden_enabled) {
    const bool wins = cfg_.white_overline_wins ? (longest >= 5) : (longest == 5);
    j.outcome = wins ? Outcome::BLACK_WIN : Outcome::ONGOING;
    return j;
  }

  bool has_five = false;
  for (int d = 0; d < NUM_DIRS; ++d) {
    if (runs[d] == 5) has_five = true;
  }
  const bool has_overline = longest >= 6;

  // 五连优先于禁手。
  if (has_five && cfg_.five_overrides_forbidden) {
    j.outcome = Outcome::BLACK_WIN;
    return j;
  }
  if (cfg_.overline && has_overline) {
    j.outcome = Outcome::WHITE_WIN;
    j.forbidden = Forbidden::OVERLINE;
    return j;
  }

  int total_fours = 0;
  bool dir_has_four[NUM_DIRS];
  for (int d = 0; d < NUM_DIRS; ++d) {
    bool open_four = false;
    const int cnt = count_fours(lv[d].own, lv[d].blocked, lv[d].len, lv[d].pos,
                                /*exact=*/true, &open_four);
    dir_has_four[d] = cnt > 0;
    total_fours += cnt;
  }
  j.fours = static_cast<uint8_t>(total_fours);
  if (cfg_.double_four && total_fours >= 2) {
    j.outcome = Outcome::WHITE_WIN;
    j.forbidden = Forbidden::DOUBLE_FOUR;
    return j;
  }

  int open_threes = 0;
  if (cfg_.double_three) {
    for (int d = 0; d < NUM_DIRS; ++d) {
      // 按最高等级归类：已经成四的方向不再计作三。
      if (dir_has_four[d]) continue;
      if (has_open_three(grid, size, move, d, depth)) ++open_threes;
    }
    j.open_threes = static_cast<uint8_t>(open_threes);
    if (open_threes >= 2) {
      j.outcome = Outcome::WHITE_WIN;
      j.forbidden = Forbidden::DOUBLE_THREE;
      return j;
    }
  }

  if (has_five) {  // five_overrides_forbidden 关闭时走到这里
    j.outcome = Outcome::BLACK_WIN;
    return j;
  }
  j.outcome = Outcome::ONGOING;
  return j;
}

bool Rules::has_open_three(uint8_t* grid, int size, int move, int dir,
                           int depth) const {
  const Geometry& g = geometry(size);
  const LineView lv = make_line(grid, g, move, dir, BLACK);
  const std::vector<int16_t>& cells = g.lines[dir][lv.line];

  // 能把三变成活四的那一手，必定与本手同处一个 4 连之内，距离不超过 3；
  // 取 ±4 是留一格余量。
  const int lo = lv.pos - 4 < 0 ? 0 : lv.pos - 4;
  const int hi = lv.pos + 4 >= lv.len ? lv.len - 1 : lv.pos + 4;

  for (int p = lo; p <= hi; ++p) {
    const uint32_t bit = 1u << p;
    if ((lv.own & bit) || (lv.blocked & bit)) continue;

    const uint32_t own2 = lv.own | bit;
    // 该点直接成五或长连，说明原形已经是四，不是三。
    if (run_length(own2, lv.len, p) >= 5) continue;

    bool open_four = false;
    count_fours(own2, lv.blocked, lv.len, p, /*exact=*/true, &open_four);
    if (!open_four) continue;

    if (depth >= cfg_.recursion_depth) {
      depth_exceeded_.fetch_add(1, std::memory_order_relaxed);
      return true;  // 深度封顶：按「那一手可下」处理
    }

    // 递归：只有当「造活四的那一手本身不是禁手」时，这才是真活三。
    const int cell = cells[p];
    grid[cell] = BLACK;
    const Judgment sub = judge_placed(grid, size, cell, BLACK, depth + 1);
    grid[cell] = EMPTY;
    if (!sub.is_forbidden()) return true;
  }
  return false;
}

Judgment Rules::judge(const uint8_t* grid, int size, int move,
                      uint8_t color) const {
  const int n = size * size;
  if (move < 0 || move >= n) {
    throw std::invalid_argument("move out of range");
  }
  if (grid[move] != EMPTY) {
    throw std::invalid_argument("move is not an empty point");
  }
  uint8_t buf[MAX_SIZE * MAX_SIZE];
  std::memcpy(buf, grid, static_cast<size_t>(n));
  buf[move] = color;
  return judge_placed(buf, size, move, color, 0);
}

void Rules::forbidden_map(const uint8_t* grid, int size, uint8_t* out) const {
  const int n = size * size;
  std::memset(out, 0, static_cast<size_t>(n));
  if (!cfg_.forbidden_enabled) return;

  uint8_t buf[MAX_SIZE * MAX_SIZE];
  std::memcpy(buf, grid, static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (buf[i] != EMPTY) continue;
    buf[i] = BLACK;
    const Judgment j = judge_placed(buf, size, i, BLACK, 0);
    buf[i] = EMPTY;
    out[i] = j.is_forbidden() ? 1 : 0;
  }
}

void Rules::judge_all(const uint8_t* grid, int size, uint8_t color,
                      uint8_t* out) const {
  const int n = size * size;
  std::memset(out, 255, static_cast<size_t>(5 * n));

  uint8_t buf[MAX_SIZE * MAX_SIZE];
  std::memcpy(buf, grid, static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (buf[i] != EMPTY) continue;
    buf[i] = color;
    const Judgment j = judge_placed(buf, size, i, color, 0);
    buf[i] = EMPTY;
    uint8_t* slot = out + 5 * i;
    slot[0] = static_cast<uint8_t>(j.outcome);
    slot[1] = static_cast<uint8_t>(j.forbidden);
    slot[2] = j.fours;
    slot[3] = j.open_threes;
    slot[4] = j.longest_run;
  }
}

}  // namespace gi
