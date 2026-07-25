// 棋盘的直线几何：把二维坐标一次性预计算成「第几条线、线内第几格」。
//
// 判定逻辑因此完全运行在一维位掩码上，不再出现任何边界判断 ——
// 越界不是靠 WALL 填充，而是靠「线本来就只有那么长」自然排除。
#pragma once

#include <cstdint>
#include <vector>

#include "constants.h"

namespace gi {

struct Geometry {
  int size = 0;

  // line_id[d][cell]：cell 在方向 d 上属于第几条线
  // line_pos[d][cell]：cell 在那条线上的下标
  std::vector<int16_t> line_id[NUM_DIRS];
  std::vector<int16_t> line_pos[NUM_DIRS];

  // lines[d][line]：该线按顺序经过的格子编号
  std::vector<std::vector<int16_t>> lines[NUM_DIRS];
};

// 按边长缓存，进程内只构建一次。
const Geometry& geometry(int size);

}  // namespace gi
