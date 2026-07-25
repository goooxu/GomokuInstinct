// pybind11 绑定。pybind11 头文件由 PyTorch 自带，因此不引入额外依赖。
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.h"
#include "renju.h"

namespace py = pybind11;

namespace {

// 接受任何实现了缓冲协议的一维字节容器（bytes / bytearray / numpy uint8）。
struct GridBuffer {
  py::buffer_info info;
  const uint8_t* ptr;

  GridBuffer(const py::buffer& buf, int expected)
      : info(buf.request()), ptr(static_cast<const uint8_t*>(info.ptr)) {
    if (info.ndim != 1) {
      throw std::invalid_argument("grid must be one-dimensional");
    }
    if (info.itemsize != 1) {
      throw std::invalid_argument("grid must be a byte buffer");
    }
    if (info.size < expected) {
      throw std::invalid_argument("grid is shorter than size*size");
    }
  }
};

void check_size(int size) {
  if (size < 5 || size > gi::MAX_SIZE) {
    throw std::invalid_argument("board size out of range");
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "gomoku-instinct 规则内核（连珠禁手判定）";

  // 供 Python 侧核对编码一致性 —— 差分测试会断言这些值与
  // gomoku_instinct/rules/constants.py 完全相同。
  m.attr("EMPTY") = py::int_(gi::EMPTY);
  m.attr("BLACK") = py::int_(gi::BLACK);
  m.attr("WHITE") = py::int_(gi::WHITE);
  m.attr("WALL") = py::int_(gi::WALL);
  m.attr("MAX_SIZE") = py::int_(gi::MAX_SIZE);
  m.attr("NUM_DIRS") = py::int_(gi::NUM_DIRS);
  m.attr("DIRECTIONS") = py::make_tuple(
      py::make_tuple(gi::DR[0], gi::DC[0]), py::make_tuple(gi::DR[1], gi::DC[1]),
      py::make_tuple(gi::DR[2], gi::DC[2]), py::make_tuple(gi::DR[3], gi::DC[3]));

  py::class_<gi::Rules>(m, "Rules")
      .def(py::init([](bool forbidden_enabled, bool overline, bool double_four,
                       bool double_three, bool five_overrides_forbidden,
                       bool white_overline_wins, int recursion_depth) {
             gi::RuleConfig cfg;
             cfg.forbidden_enabled = forbidden_enabled;
             cfg.overline = overline;
             cfg.double_four = double_four;
             cfg.double_three = double_three;
             cfg.five_overrides_forbidden = five_overrides_forbidden;
             cfg.white_overline_wins = white_overline_wins;
             cfg.recursion_depth = recursion_depth;
             return new gi::Rules(cfg);
           }),
           py::arg("forbidden_enabled") = true, py::arg("overline") = true,
           py::arg("double_four") = true, py::arg("double_three") = true,
           py::arg("five_overrides_forbidden") = true,
           py::arg("white_overline_wins") = true,
           py::arg("recursion_depth") = 64)

      .def(
          "judge",
          [](const gi::Rules& self, const py::buffer& grid, int size, int move,
             int color) {
            check_size(size);
            GridBuffer buf(grid, size * size);
            gi::Judgment j;
            {
              py::gil_scoped_release release;
              j = self.judge(buf.ptr, size, move, static_cast<uint8_t>(color));
            }
            return py::make_tuple(static_cast<int>(j.outcome),
                                  static_cast<int>(j.forbidden),
                                  static_cast<int>(j.fours),
                                  static_cast<int>(j.open_threes),
                                  static_cast<int>(j.longest_run));
          },
          py::arg("grid"), py::arg("size"), py::arg("move"), py::arg("color"),
          "判定在空点 move 落子的结果，返回 "
          "(outcome, forbidden, fours, open_threes, longest_run)")

      .def(
          "forbidden_map",
          [](const gi::Rules& self, const py::buffer& grid, int size) {
            check_size(size);
            const int n = size * size;
            GridBuffer buf(grid, n);
            std::vector<uint8_t> out(static_cast<size_t>(n));
            {
              py::gil_scoped_release release;
              self.forbidden_map(buf.ptr, size, out.data());
            }
            return py::bytes(reinterpret_cast<const char*>(out.data()),
                             static_cast<size_t>(n));
          },
          py::arg("grid"), py::arg("size"),
          "全盘黑方禁手点标记，返回长度 size*size 的字节串")

      .def(
          "judge_all",
          [](const gi::Rules& self, const py::buffer& grid, int size, int color) {
            check_size(size);
            const int n = size * size;
            GridBuffer buf(grid, n);
            std::vector<uint8_t> out(static_cast<size_t>(5 * n));
            {
              py::gil_scoped_release release;
              self.judge_all(buf.ptr, size, static_cast<uint8_t>(color),
                             out.data());
            }
            return py::bytes(reinterpret_cast<const char*>(out.data()),
                             static_cast<size_t>(5 * n));
          },
          py::arg("grid"), py::arg("size"), py::arg("color"),
          "对每个空点做一次完整判定，返回长度 5*size*size 的字节串；"
          "每格依次为 [outcome, forbidden, fours, open_threes, longest_run]，"
          "非空点填 255")

      .def_property_readonly("depth_exceeded", &gi::Rules::depth_exceeded)
      .def_property_readonly("max_depth", &gi::Rules::max_depth)
      .def("reset_counters", &gi::Rules::reset_counters);
}
