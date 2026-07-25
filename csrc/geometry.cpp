#include "geometry.h"

#include <map>
#include <mutex>
#include <stdexcept>

namespace gi {

namespace {

Geometry build(int size) {
  if (size < 5 || size > MAX_SIZE) {
    throw std::invalid_argument("board size out of range");
  }
  Geometry g;
  g.size = size;
  const int n = size * size;

  for (int d = 0; d < NUM_DIRS; ++d) {
    g.line_id[d].assign(n, -1);
    g.line_pos[d].assign(n, -1);

    const int dr = DR[d];
    const int dc = DC[d];

    for (int r = 0; r < size; ++r) {
      for (int c = 0; c < size; ++c) {
        // 一条线的起点：沿反方向再走一步就出界。
        const int pr = r - dr;
        const int pc = c - dc;
        const bool is_start = pr < 0 || pr >= size || pc < 0 || pc >= size;
        if (!is_start) continue;

        std::vector<int16_t> cells;
        int cr = r;
        int cc = c;
        while (cr >= 0 && cr < size && cc >= 0 && cc < size) {
          const int cell = cr * size + cc;
          g.line_id[d][cell] = static_cast<int16_t>(g.lines[d].size());
          g.line_pos[d][cell] = static_cast<int16_t>(cells.size());
          cells.push_back(static_cast<int16_t>(cell));
          cr += dr;
          cc += dc;
        }
        g.lines[d].push_back(std::move(cells));
      }
    }
  }
  return g;
}

std::mutex g_mutex;
std::map<int, Geometry> g_cache;

}  // namespace

const Geometry& geometry(int size) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_cache.find(size);
  if (it == g_cache.end()) {
    it = g_cache.emplace(size, build(size)).first;
  }
  return it->second;
}

}  // namespace gi
