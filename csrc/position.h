// 对局状态：棋盘 + 行棋方 + 历史 + 终局判定。
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "constants.h"
#include "renju.h"

namespace gi {

// 禁手点的语义。默认 LOSE：严格 RIF，禁手点仍是合法落子，黑方落上去立即判负 ——
// 避开禁手因此是模型必须自己学会的能力，而不是由引擎代劳。
enum class ForbiddenSemantics : uint8_t { LOSE = 0, ILLEGAL = 1 };

constexpr int HISTORY_PLANES = 4;

class Position {
 public:
  Position(int size, const Rules* rules, ForbiddenSemantics semantics)
      : size_(size),
        cells_(static_cast<size_t>(size) * size, EMPTY),
        rules_(rules),
        semantics_(semantics) {}

  void reset() {
    std::fill(cells_.begin(), cells_.end(), EMPTY);
    to_move_ = BLACK;
    outcome_ = Outcome::ONGOING;
    moves_.clear();
  }

  int size() const { return size_; }
  int num_cells() const { return static_cast<int>(cells_.size()); }
  const uint8_t* grid() const { return cells_.data(); }
  uint8_t to_move() const { return to_move_; }
  Outcome outcome() const { return outcome_; }
  bool terminal() const { return outcome_ != Outcome::ONGOING; }
  int ply() const { return static_cast<int>(moves_.size()); }
  const std::vector<int32_t>& moves() const { return moves_; }

  bool is_legal(int move) const {
    if (move < 0 || move >= num_cells() || cells_[move] != EMPTY) return false;
    if (semantics_ == ForbiddenSemantics::ILLEGAL && to_move_ == BLACK) {
      return !rules_->judge(cells_.data(), size_, move, BLACK).is_forbidden();
    }
    return true;
  }

  // 合法落子写入 out，返回个数。
  int legal_moves(int32_t* out) const {
    int count = 0;
    for (int i = 0; i < num_cells(); ++i) {
      if (cells_[i] != EMPTY) continue;
      if (semantics_ == ForbiddenSemantics::ILLEGAL && to_move_ == BLACK &&
          rules_->judge(cells_.data(), size_, i, BLACK).is_forbidden()) {
        continue;
      }
      out[count++] = i;
    }
    return count;
  }

  void play(int move) {
    const Judgment j = rules_->judge(cells_.data(), size_, move, to_move_);
    cells_[move] = to_move_;
    moves_.push_back(move);

    // 无论是否终局都切换行棋方：to_move_ 的语义统一为「下一手轮到谁」。
    // 搜索树用它来确定叶节点的价值视角，含糊不得。
    to_move_ = (to_move_ == BLACK) ? WHITE : BLACK;

    if (j.outcome != Outcome::ONGOING) {
      outcome_ = j.outcome;
      return;
    }
    if (static_cast<int>(moves_.size()) >= num_cells()) {
      outcome_ = Outcome::DRAW;
      return;
    }

    // ILLEGAL 语义下黑方可能被逼到无处可下。
    if (semantics_ == ForbiddenSemantics::ILLEGAL && to_move_ == BLACK) {
      bool any = false;
      for (int i = 0; i < num_cells() && !any; ++i) {
        if (cells_[i] == EMPTY &&
            !rules_->judge(cells_.data(), size_, i, BLACK).is_forbidden()) {
          any = true;
        }
      }
      if (!any) outcome_ = Outcome::WHITE_WIN;
    }
  }

  void undo() {
    if (moves_.empty()) return;
    const int move = moves_.back();
    moves_.pop_back();
    to_move_ = cells_[move];
    cells_[move] = EMPTY;
    outcome_ = Outcome::ONGOING;
  }

  // 最近 HISTORY_PLANES 手的落点，不足处填 -1（下标 0 为最近一手）。
  void history(int32_t* out) const {
    const int n = static_cast<int>(moves_.size());
    for (int k = 0; k < HISTORY_PLANES; ++k) {
      out[k] = (k < n) ? moves_[n - 1 - k] : -1;
    }
  }

  // 从 color 视角看的终局结果：胜 1 / 和 0 / 负 -1。
  float result_for(uint8_t color) const {
    if (outcome_ == Outcome::DRAW) return 0.0f;
    if (outcome_ == Outcome::BLACK_WIN) return color == BLACK ? 1.0f : -1.0f;
    if (outcome_ == Outcome::WHITE_WIN) return color == WHITE ? 1.0f : -1.0f;
    return 0.0f;
  }

  void copy_grid_to(uint8_t* out) const {
    std::memcpy(out, cells_.data(), cells_.size());
  }

 private:
  int size_;
  std::vector<uint8_t> cells_;
  const Rules* rules_;
  ForbiddenSemantics semantics_;
  uint8_t to_move_ = BLACK;
  Outcome outcome_ = Outcome::ONGOING;
  std::vector<int32_t> moves_;
};

}  // namespace gi
