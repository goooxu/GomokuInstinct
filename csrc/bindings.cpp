// pybind11 绑定。pybind11 头文件由 PyTorch 自带，因此不引入额外依赖。
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.h"
#include "position.h"
#include "renju.h"
#include "selfplay.h"

namespace py = pybind11;

template <typename T>
using Array = py::array_t<T, py::array::c_style | py::array::forcecast>;

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
          "pattern_map",
          [](const gi::Rules& self, const py::buffer& grid, int size, int color) {
            check_size(size);
            const int n = size * size;
            GridBuffer buf(grid, n);
            std::vector<uint8_t> out(static_cast<size_t>(n));
            {
              py::gil_scoped_release release;
              self.pattern_map(buf.ptr, size, static_cast<uint8_t>(color),
                               out.data());
            }
            return py::bytes(reinterpret_cast<const char*>(out.data()),
                             static_cast<size_t>(n));
          },
          py::arg("grid"), py::arg("size"), py::arg("color"),
          "逐点棋型标注，返回长度 size*size 的字节串，取值为 Level")

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

  // ── 向量化自博弈 ─────────────────────────────────────────────────────────
  py::class_<gi::SelfPlayRunner>(m, "SelfPlayRunner")
      .def(py::init([](int board_size, int num_games, int sims, int fast_sims,
                       float full_search_prob, float c_puct, float fpu_reduction,
                       float dirichlet_alpha, float dirichlet_eps,
                       float temperature, int temperature_moves,
                       float raw_policy_fraction, bool resign_enabled,
                       float resign_threshold, float resign_audit_fraction,
                       int num_threads, uint64_t seed, int forbidden_semantics,
                       bool forbidden_enabled, int recursion_depth) {
             gi::SelfPlayConfig cfg;
             cfg.board_size = board_size;
             cfg.num_games = num_games;
             cfg.sims = sims;
             cfg.fast_sims = fast_sims;
             cfg.full_search_prob = full_search_prob;
             cfg.dirichlet_alpha = dirichlet_alpha;
             cfg.dirichlet_eps = dirichlet_eps;
             cfg.temperature = temperature;
             cfg.temperature_moves = temperature_moves;
             cfg.raw_policy_fraction = raw_policy_fraction;
             cfg.resign_enabled = resign_enabled;
             cfg.resign_threshold = resign_threshold;
             cfg.resign_audit_fraction = resign_audit_fraction;
             cfg.num_threads = num_threads;
             cfg.seed = seed;
             cfg.semantics =
                 static_cast<gi::ForbiddenSemantics>(forbidden_semantics);
             cfg.mcts.c_puct = c_puct;
             cfg.mcts.fpu_reduction = fpu_reduction;
             cfg.rules.forbidden_enabled = forbidden_enabled;
             cfg.rules.recursion_depth = recursion_depth;
             return new gi::SelfPlayRunner(cfg);
           }),
           py::arg("board_size") = 15, py::arg("num_games") = 1024,
           py::arg("sims") = 400, py::arg("fast_sims") = 100,
           py::arg("full_search_prob") = 0.25f, py::arg("c_puct") = 1.6f,
           py::arg("fpu_reduction") = 0.25f, py::arg("dirichlet_alpha") = 0.15f,
           py::arg("dirichlet_eps") = 0.25f, py::arg("temperature") = 1.0f,
           py::arg("temperature_moves") = 16,
           py::arg("raw_policy_fraction") = 0.0f,
           py::arg("resign_enabled") = true,
           py::arg("resign_threshold") = -0.92f,
           py::arg("resign_audit_fraction") = 0.05f,
           py::arg("num_threads") = 8, py::arg("seed") = 20260725ULL,
           py::arg("forbidden_semantics") = 0,
           py::arg("forbidden_enabled") = true,
           py::arg("recursion_depth") = 64)

      .def_property_readonly("num_games", &gi::SelfPlayRunner::num_games)
      .def_property_readonly("num_cells", &gi::SelfPlayRunner::num_cells)

      .def(
          "collect",
          [](gi::SelfPlayRunner& self, Array<uint8_t> boards,
             Array<uint8_t> to_move, Array<int32_t> history,
             Array<int32_t> move_number, Array<uint8_t> needs_eval) {
            const int g = self.num_games();
            const int n = self.num_cells();
            if (boards.size() < static_cast<py::ssize_t>(g) * n ||
                to_move.size() < g || history.size() < g * gi::HISTORY_PLANES ||
                move_number.size() < g || needs_eval.size() < g) {
              throw std::invalid_argument("collect 的输出缓冲区尺寸不足");
            }
            uint8_t* pb = boards.mutable_data();
            uint8_t* pt = to_move.mutable_data();
            int32_t* ph = history.mutable_data();
            int32_t* pm = move_number.mutable_data();
            uint8_t* pn = needs_eval.mutable_data();
            py::gil_scoped_release release;
            self.collect(pb, pt, ph, pm, pn);
          },
          py::arg("boards"), py::arg("to_move"), py::arg("history"),
          py::arg("move_number"), py::arg("needs_eval"),
          "每局从根下潜一次，把待评估局面写进给定缓冲区")

      .def(
          "apply",
          [](gi::SelfPlayRunner& self, Array<float> policy, Array<float> value) {
            const int g = self.num_games();
            const int n = self.num_cells();
            if (policy.size() < static_cast<py::ssize_t>(g) * n ||
                value.size() < g) {
              throw std::invalid_argument("apply 的输入缓冲区尺寸不足");
            }
            const float* pp = policy.data();
            const float* pv = value.data();
            py::gil_scoped_release release;
            self.apply(pp, pv);
          },
          py::arg("policy"), py::arg("value"),
          "回填评估结果：展开叶子、回传价值，搜索次数够了就落子")

      .def_property_readonly("pending_samples",
                             &gi::SelfPlayRunner::pending_samples)

      .def(
          "drain",
          [](gi::SelfPlayRunner& self, int max_samples) {
            const int n = self.num_cells();
            const int count = std::min(max_samples, self.pending_samples());

            Array<uint8_t> boards({count, n});
            Array<float> policy({count, n});
            Array<uint8_t> to_move({count});
            Array<int32_t> history({count, gi::HISTORY_PLANES});
            Array<int32_t> move_number({count});
            Array<float> value({count});
            Array<int32_t> plies({count});
            Array<int32_t> next_move({count});
            Array<float> root_value({count});
            Array<uint8_t> searched({count});

            int written = 0;
            if (count > 0) {
              uint8_t* pb = boards.mutable_data();
              float* pp = policy.mutable_data();
              uint8_t* pt = to_move.mutable_data();
              int32_t* ph = history.mutable_data();
              int32_t* pm = move_number.mutable_data();
              float* pv = value.mutable_data();
              int32_t* pl = plies.mutable_data();
              int32_t* pnm = next_move.mutable_data();
              float* prv = root_value.mutable_data();
              uint8_t* ps = searched.mutable_data();
              py::gil_scoped_release release;
              written = self.drain(count, pb, pt, ph, pm, pp, pv, pl, pnm, prv, ps);
            }

            py::dict out;
            out["count"] = written;
            out["boards"] = boards;
            out["policy"] = policy;
            out["to_move"] = to_move;
            out["history"] = history;
            out["move_number"] = move_number;
            out["value"] = value;
            out["plies_remaining"] = plies;
            out["next_move"] = next_move;
            out["root_value"] = root_value;
            out["searched"] = searched;
            return out;
          },
          py::arg("max_samples"), "取出已完成对局的样本并清空队列")

      .def_property_readonly(
          "stats",
          [](const gi::SelfPlayRunner& self) {
            const gi::Stats s = self.stats();
            py::dict out;
            out["games"] = s.games;
            out["moves"] = s.moves;
            out["completed_plies"] = s.completed_plies;
            out["samples"] = s.samples;
            out["black_wins"] = s.black_wins;
            out["white_wins"] = s.white_wins;
            out["draws"] = s.draws;
            out["forbidden_losses"] = s.forbidden_losses;
            out["resigns"] = s.resigns;
            out["resign_audits"] = s.resign_audits;
            out["resign_false_positives"] = s.resign_false_positives;
            out["raw_policy_games"] = s.raw_policy_games;
            return out;
          })
      .def("reset_stats", &gi::SelfPlayRunner::reset_stats)
      .def("rng_state", &gi::SelfPlayRunner::rng_state)
      .def("set_rng_state", &gi::SelfPlayRunner::set_rng_state);
}
