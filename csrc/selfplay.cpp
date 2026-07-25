#include "selfplay.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>

namespace gi {

// ── 常驻线程池 ──────────────────────────────────────────────────────────────
// collect/apply 每手要被调用几百次，每次都新建线程的开销不可忽略，
// 所以线程只创建一次，之后靠代次号唤醒。
class ThreadPool {
 public:
  explicit ThreadPool(int n) {
    for (int i = 0; i < n; ++i) {
      threads_.emplace_back([this] { worker(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : threads_) t.join();
  }

  void run(const std::function<void(int)>& fn, int count) {
    if (threads_.empty()) {
      for (int i = 0; i < count; ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fn_ = &fn;
      count_ = count;
      next_.store(0, std::memory_order_relaxed);
      active_ = static_cast<int>(threads_.size());
      ++generation_;
    }
    cv_start_.notify_all();
    std::unique_lock<std::mutex> lock(mutex_);
    cv_done_.wait(lock, [this] { return active_ == 0; });
  }

 private:
  void worker() {
    uint64_t seen = 0;
    while (true) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      const std::function<void(int)>* fn = fn_;
      const int count = count_;
      lock.unlock();

      int i;
      while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < count) {
        (*fn)(i);
      }

      lock.lock();
      if (--active_ == 0) cv_done_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  const std::function<void(int)>* fn_ = nullptr;
  int count_ = 0;
  std::atomic<int> next_{0};
  int active_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

namespace {

float result_from(Outcome outcome, uint8_t color) {
  if (outcome == Outcome::DRAW) return 0.0f;
  if (outcome == Outcome::BLACK_WIN) return color == BLACK ? 1.0f : -1.0f;
  if (outcome == Outcome::WHITE_WIN) return color == WHITE ? 1.0f : -1.0f;
  return 0.0f;
}

}  // namespace

// ── 构造 ────────────────────────────────────────────────────────────────────
SelfPlayRunner::SelfPlayRunner(const SelfPlayConfig& cfg) : cfg_(cfg) {
  rules_ = std::make_unique<Rules>(cfg.rules);

  MctsConfig mcts = cfg.mcts;
  // 每手重建一棵树，因此节点池只需覆盖单手的模拟次数。
  mcts.max_nodes = std::max(cfg.sims, cfg.fast_sims) + 8;

  const int n = num_cells();
  slots_.resize(cfg.num_games);
  for (int i = 0; i < cfg.num_games; ++i) {
    Slot& slot = slots_[i];
    slot.pos =
        std::make_unique<Position>(cfg.board_size, rules_.get(), cfg.semantics);
    slot.tree = std::make_unique<MctsTree>(n, mcts);
    // 每局独立的随机源：换机续训时可以精确恢复。
    slot.rng.seed(cfg.seed + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(i) + 1));
    slot.visit_counts.resize(n);
    slot.root_policy.resize(n);
    slot.noise.resize(n);
    start_game(slot);
  }

  pool_ = std::make_unique<ThreadPool>(std::max(0, cfg.num_threads));
}

SelfPlayRunner::~SelfPlayRunner() = default;

// ── 对局与单手的起止 ────────────────────────────────────────────────────────
void SelfPlayRunner::start_game(Slot& slot) {
  slot.pos->reset();
  slot.pending.clear();
  slot.would_resign_side = 0;

  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  slot.raw_policy_game = unit(slot.rng) < cfg_.raw_policy_fraction;
  slot.disable_resign =
      cfg_.resign_enabled && unit(slot.rng) < cfg_.resign_audit_fraction;
  if (slot.raw_policy_game) slot.stats.raw_policy_games += 1;
  if (slot.disable_resign) slot.stats.resign_audits += 1;

  start_move(slot);
}

void SelfPlayRunner::start_move(Slot& slot) {
  slot.tree->clear();
  slot.root_ply = slot.pos->ply();
  slot.sims_done = 0;

  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  slot.full_search = unit(slot.rng) < cfg_.full_search_prob;
  slot.sims_target = slot.full_search ? cfg_.sims : cfg_.fast_sims;
}

// ── 一轮：收集 ──────────────────────────────────────────────────────────────
void SelfPlayRunner::collect(uint8_t* boards, uint8_t* to_move,
                             int32_t* history, int32_t* move_number,
                             uint8_t* needs_eval) {
  const int n = num_cells();
  auto work = [&](int i) {
    Slot& slot = slots_[i];
    const Descent d = slot.tree->descend(*slot.pos);
    slot.leaf = d.leaf;
    slot.leaf_needs_eval = d.needs_eval;
    slot.leaf_terminal_value = d.terminal_value;
    slot.descent_depth = slot.pos->ply() - slot.root_ply;

    slot.pos->copy_grid_to(boards + static_cast<size_t>(i) * n);
    to_move[i] = slot.pos->to_move();
    slot.pos->history(history + static_cast<size_t>(i) * HISTORY_PLANES);
    move_number[i] = slot.pos->ply();
    needs_eval[i] = d.needs_eval ? 1 : 0;
  };
  pool_->run(work, static_cast<int>(slots_.size()));
}

// ── 一轮：回填 ──────────────────────────────────────────────────────────────
void SelfPlayRunner::apply(const float* policy, const float* value) {
  const int n = num_cells();
  auto work = [&](int i) {
    Slot& slot = slots_[i];
    const float* priors = policy + static_cast<size_t>(i) * n;

    if (slot.leaf_needs_eval) {
      const bool is_root = (slot.leaf == slot.tree->root());
      slot.tree->expand_and_backup(slot.leaf, priors, value[i], *slot.pos);

      if (is_root) {
        // 根节点刚展开：先留一份加噪前的先验（零搜索落子要用），再混入 Dirichlet 噪声。
        std::memcpy(slot.root_policy.data(), priors,
                    static_cast<size_t>(n) * sizeof(float));

        const int count = slot.tree->root_child_count();
        if (count > 0 && cfg_.dirichlet_eps > 0.0f) {
          std::gamma_distribution<float> gamma(cfg_.dirichlet_alpha, 1.0f);
          float sum = 0.0f;
          for (int k = 0; k < count; ++k) {
            slot.noise[k] = gamma(slot.rng);
            sum += slot.noise[k];
          }
          if (sum > 0.0f) {
            for (int k = 0; k < count; ++k) slot.noise[k] /= sum;
            slot.tree->add_root_noise(slot.noise.data(), cfg_.dirichlet_eps);
          }
        }
      }
    } else {
      slot.tree->backup_terminal(slot.leaf, slot.leaf_terminal_value);
    }

    // 退回根部
    for (int k = 0; k < slot.descent_depth; ++k) slot.pos->undo();
    slot.descent_depth = 0;

    if (++slot.sims_done >= slot.sims_target) finish_move(slot);
  };
  pool_->run(work, static_cast<int>(slots_.size()));
}

// ── 落子 ────────────────────────────────────────────────────────────────────
int SelfPlayRunner::choose_move(Slot& slot) {
  const int n = num_cells();

  if (slot.raw_policy_game) {
    // 部署分布自博弈：按零搜索策略落子（与实际对战时完全一致的选点方式）。
    int best = -1;
    float best_p = -1.0f;
    for (int m = 0; m < n; ++m) {
      if (!slot.pos->is_legal(m)) continue;
      if (slot.root_policy[m] > best_p) {
        best_p = slot.root_policy[m];
        best = m;
      }
    }
    if (best >= 0) return best;
  }

  const float temperature =
      slot.pos->ply() < cfg_.temperature_moves ? cfg_.temperature : 0.0f;

  if (temperature <= 1e-6f) {
    int best = -1;
    int best_visits = -1;
    for (int m = 0; m < n; ++m) {
      if (slot.visit_counts[m] > best_visits) {
        best_visits = slot.visit_counts[m];
        best = m;
      }
    }
    return best;
  }

  double total = 0.0;
  std::vector<double> weights(n, 0.0);
  const double inv_t = 1.0 / temperature;
  for (int m = 0; m < n; ++m) {
    if (slot.visit_counts[m] <= 0) continue;
    weights[m] = std::pow(static_cast<double>(slot.visit_counts[m]), inv_t);
    total += weights[m];
  }
  if (total <= 0.0) {
    for (int m = 0; m < n; ++m) {
      if (slot.pos->is_legal(m)) return m;
    }
    return -1;
  }

  std::uniform_real_distribution<double> pick(0.0, total);
  double target = pick(slot.rng);
  for (int m = 0; m < n; ++m) {
    target -= weights[m];
    if (target <= 0.0 && weights[m] > 0.0) return m;
  }
  for (int m = n - 1; m >= 0; --m) {
    if (weights[m] > 0.0) return m;
  }
  return -1;
}

void SelfPlayRunner::finish_move(Slot& slot) {
  const int n = num_cells();
  slot.tree->root_visit_counts(slot.visit_counts.data());
  const float root_value = slot.tree->root_value();
  const uint8_t mover = slot.pos->to_move();

  // 认输。审计局照常走完，只记下「本该在这里认输」，用来算误判率。
  if (cfg_.resign_enabled && root_value < cfg_.resign_threshold &&
      slot.pos->ply() >= 10) {
    if (slot.disable_resign) {
      if (slot.would_resign_side == 0) slot.would_resign_side = mover;
    } else {
      const Outcome outcome =
          mover == BLACK ? Outcome::WHITE_WIN : Outcome::BLACK_WIN;
      finish_game(slot, outcome, /*by_resign=*/true);
      return;
    }
  }

  // 只有完整搜索的手才产出训练目标。
  if (slot.full_search) {
    int total_visits = 0;
    for (int m = 0; m < n; ++m) total_visits += slot.visit_counts[m];
    if (total_visits > 0) {
      Sample sample;
      sample.board.resize(n);
      slot.pos->copy_grid_to(sample.board.data());
      sample.policy.resize(n);
      const float inv = 1.0f / static_cast<float>(total_visits);
      for (int m = 0; m < n; ++m) {
        sample.policy[m] = static_cast<float>(slot.visit_counts[m]) * inv;
      }
      sample.to_move = mover;
      slot.pos->history(sample.history);
      sample.move_number = slot.pos->ply();
      sample.ply = slot.pos->ply();
      sample.root_value = root_value;
      sample.searched = 1;
      slot.pending.push_back(std::move(sample));
    }
  }

  const int move = choose_move(slot);
  if (move < 0) {  // 无处可下
    finish_game(slot, Outcome::DRAW, false);
    return;
  }

  slot.pos->play(move);
  slot.stats.moves += 1;

  if (slot.pos->terminal()) {
    // 黑方落子却判白胜，只能是走出了禁手。
    if (slot.pos->outcome() == Outcome::WHITE_WIN && mover == BLACK) {
      slot.stats.forbidden_losses += 1;
    }
    finish_game(slot, slot.pos->outcome(), false);
    return;
  }
  start_move(slot);
}

void SelfPlayRunner::finish_game(Slot& slot, Outcome outcome, bool by_resign) {
  const std::vector<int32_t>& moves = slot.pos->moves();
  const int total_plies = static_cast<int>(moves.size());

  for (Sample& sample : slot.pending) {
    sample.value = result_from(outcome, sample.to_move);
    sample.plies_remaining = total_plies - sample.ply;
    // 对手的实际应手 —— 逼网络内化一层前瞻。
    sample.next_move =
        (sample.ply + 1 < total_plies) ? moves[sample.ply + 1] : -1;
  }
  slot.stats.samples += static_cast<int64_t>(slot.pending.size());
  slot.finished.insert(slot.finished.end(),
                       std::make_move_iterator(slot.pending.begin()),
                       std::make_move_iterator(slot.pending.end()));
  slot.pending.clear();

  slot.stats.games += 1;
  slot.stats.completed_plies += total_plies;
  if (outcome == Outcome::BLACK_WIN) slot.stats.black_wins += 1;
  else if (outcome == Outcome::WHITE_WIN) slot.stats.white_wins += 1;
  else slot.stats.draws += 1;

  if (by_resign) slot.stats.resigns += 1;

  // 审计：认输方最终反而赢了，说明这条阈值会误杀。
  if (slot.would_resign_side != 0 &&
      result_from(outcome, slot.would_resign_side) > 0.0f) {
    slot.stats.resign_false_positives += 1;
  }

  start_game(slot);
}

// ── 取样本与统计 ────────────────────────────────────────────────────────────
int SelfPlayRunner::pending_samples() const {
  int total = 0;
  for (const Slot& slot : slots_) total += static_cast<int>(slot.finished.size());
  return total;
}

int SelfPlayRunner::drain(int max_samples, uint8_t* boards, uint8_t* to_move,
                          int32_t* history, int32_t* move_number, float* policy,
                          float* value, int32_t* plies, int32_t* next_move,
                          float* root_value, uint8_t* searched) {
  const int n = num_cells();
  int written = 0;

  for (Slot& slot : slots_) {
    size_t taken = 0;
    for (Sample& sample : slot.finished) {
      if (written >= max_samples) break;
      const size_t offset = static_cast<size_t>(written) * n;
      std::memcpy(boards + offset, sample.board.data(), n);
      std::memcpy(policy + offset, sample.policy.data(), n * sizeof(float));
      to_move[written] = sample.to_move;
      std::memcpy(history + static_cast<size_t>(written) * HISTORY_PLANES,
                  sample.history, HISTORY_PLANES * sizeof(int32_t));
      move_number[written] = sample.move_number;
      value[written] = sample.value;
      plies[written] = sample.plies_remaining;
      next_move[written] = sample.next_move;
      root_value[written] = sample.root_value;
      searched[written] = sample.searched;
      ++written;
      ++taken;
    }
    slot.finished.erase(slot.finished.begin(),
                        slot.finished.begin() + static_cast<long>(taken));
    if (written >= max_samples) break;
  }
  return written;
}

Stats SelfPlayRunner::stats() const {
  Stats total;
  for (const Slot& slot : slots_) {
    total.games += slot.stats.games;
    total.moves += slot.stats.moves;
    total.completed_plies += slot.stats.completed_plies;
    total.samples += slot.stats.samples;
    total.black_wins += slot.stats.black_wins;
    total.white_wins += slot.stats.white_wins;
    total.draws += slot.stats.draws;
    total.forbidden_losses += slot.stats.forbidden_losses;
    total.resigns += slot.stats.resigns;
    total.resign_false_positives += slot.stats.resign_false_positives;
    total.resign_audits += slot.stats.resign_audits;
    total.raw_policy_games += slot.stats.raw_policy_games;
  }
  return total;
}

void SelfPlayRunner::reset_stats() {
  for (Slot& slot : slots_) slot.stats = Stats();
}

std::vector<std::string> SelfPlayRunner::rng_state() const {
  std::vector<std::string> out;
  out.reserve(slots_.size());
  for (const Slot& slot : slots_) {
    std::ostringstream ss;
    ss << slot.rng;
    out.push_back(ss.str());
  }
  return out;
}

void SelfPlayRunner::set_rng_state(const std::vector<std::string>& state) {
  for (size_t i = 0; i < slots_.size() && i < state.size(); ++i) {
    std::istringstream ss(state[i]);
    ss >> slots_[i].rng;
  }
}

}  // namespace gi
