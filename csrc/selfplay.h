// 向量化自博弈：同时推进上千局，把 MCTS 的叶子评估攒成一个大批次交给 GPU。
//
// 一轮的节奏是：
//   collect()  每局从根下潜一次，停在叶子上，把待评估局面写进输出缓冲区
//   （Python 侧编码特征、跑一次网络前向）
//   apply()    用先验展开叶子、回传价值、退回根部；某局搜索次数够了就落子
//
// 每局每轮只下潜一次，同一棵树上不存在并发下潜，因此**不需要 virtual loss**，
// PUCT 是精确的。批大小恒等于对局数，便于 CUDA Graph 捕获。
//
// 特征编码刻意留在 Python 侧：编码规范只有一份（gomoku_instinct/model/features.py），
// 不在 C++ 里重复实现，免得两边悄悄走样。
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "constants.h"
#include "mcts.h"
#include "position.h"
#include "renju.h"
#include "thread_pool.h"

namespace gi {

struct SelfPlayConfig {
  int board_size = 15;
  int num_games = 1024;

  int sims = 400;
  int fast_sims = 100;
  // playout cap randomization：少数手用满 sims 并产出训练目标，
  // 多数手用低 sims 只负责推进对局 —— 同样算力下目标产量高得多。
  float full_search_prob = 0.25f;

  float dirichlet_alpha = 0.15f;
  float dirichlet_eps = 0.25f;

  float temperature = 1.0f;
  int temperature_moves = 16;

  // 部署分布自博弈：这些对局由**零搜索策略**决定落子，但训练目标仍由 MCTS 产生。
  // 目的是让样本覆盖到部署时真正会走进去的局面，消除训练/部署的分布漂移。
  float raw_policy_fraction = 0.0f;

  // 部署分布对局的随机开局手数。**这一项是必需的，不是可选项。**
  //
  // 零搜索落子是 argmax，是个确定性函数：同一份权重看到同一个局面必然给出同一手。
  // 而这条路径又绕过了温度采样、用的是加噪之前的先验 —— 于是同时开局的部署分布
  // 对局会走出**完全一样的棋**。实测（固定权重、64 局部署分布对局）：
  // 912 条样本里只有 20 个不重复局面，2.2% —— 64 局就是同一盘棋。
  // 对照组（同样配置、部署分布关掉）是 96.8%。
  //
  // 后果不只是浪费算力：那一小撮局面在回放池里被严重超额加权。
  // renju15f 早期真实分片实测重复率约 26%，与 raw_policy_fraction=0.25 吻合。
  //
  // 开局用与竞技场、网页观战同一套规则（中央区域里的一个小窗口）。
  // 这样"零搜索策略从第 k 手起接管"的语义不变，而局面各不相同。
  int raw_policy_opening_plies = 0;

  // 随机开局落在哪：中央 center_region 见方里的一个 opening_window 见方的窗口。
  // 与 gomoku_instinct/eval/opening.py 保持一致（那边是评测与网页用的同一规则）。
  int center_region = 9;
  int opening_window = 5;

  // 尾段重搜（两趟走）：对局照常走完，**落子过程中不产出任何样本**；
  // 终局后回头把最后 research_last_plies 个局面用满 sims 重新搜一遍，
  // 那次搜索的访问分布才是训练目标。0 = 关闭，走原来的边下边采。
  //
  // 为什么要两趟：搜索深度在落子前就定了，而「这一手是不是最后 N 手」
  // 要等对局结束才知道 —— 两者在一趟里没法同时满足。分成两趟就都满足了：
  // 最后 N 个局面**一个不漏**，且每一个都是满 sims 的干净目标。
  //
  // 动机：残局的价值标签方差最小。开局局面的胜负标签噪声极大 ——
  // 一局棋的结果和第 3 手的关系很弱，却被当成同等确定的监督信号。
  //
  // 重搜时**不加 Dirichlet 噪声**：噪声是为了让自博弈去试没试过的着法，
  // 而这里只是评估一个已经确定的局面，加噪只会污染目标。
  int research_last_plies = 0;

  bool resign_enabled = true;
  float resign_threshold = -0.92f;
  float resign_audit_fraction = 0.05f;

  int num_threads = 8;
  uint64_t seed = 20260725;
  ForbiddenSemantics semantics = ForbiddenSemantics::LOSE;

  MctsConfig mcts;
  RuleConfig rules;
};

struct Stats {
  int64_t games = 0;
  int64_t moves = 0;          // 所有落子，含仍在进行中的对局
  int64_t completed_plies = 0;  // 仅已完成对局的手数
  int64_t samples = 0;
  int64_t black_wins = 0;
  int64_t white_wins = 0;
  int64_t draws = 0;
  int64_t forbidden_losses = 0;  // 黑方走出禁手而告负的局数
  int64_t resigns = 0;
  int64_t resign_false_positives = 0;  // 审计局里认输方其实会赢的次数
  int64_t resign_audits = 0;
  int64_t raw_policy_games = 0;
};

class SelfPlayRunner {
 public:
  explicit SelfPlayRunner(const SelfPlayConfig& cfg);
  ~SelfPlayRunner();

  int num_games() const { return cfg_.num_games; }
  int num_cells() const { return cfg_.board_size * cfg_.board_size; }

  // 收集一批待评估局面。needs_eval 标记哪些槽位真的需要网络输出
  // （终局叶子不需要，但仍占位以保持批大小恒定）。
  void collect(uint8_t* boards, uint8_t* to_move, int32_t* history,
               int32_t* move_number, uint8_t* needs_eval);

  // 回填评估结果。policy 为 (G, N)，只需按空点屏蔽即可，
  // 展开时会在真正的合法着法上重新归一化。value 为 (G,)，行棋方视角。
  void apply(const float* policy, const float* value);

  // 已完成对局累积的样本数。
  int pending_samples() const;

  // 取出样本并清空队列，返回实际取出的条数。
  int drain(int max_samples, uint8_t* boards, uint8_t* to_move, int32_t* history,
            int32_t* move_number, float* policy, float* value, int32_t* plies,
            int32_t* next_move, float* root_value, float* blunder_gap,
            uint8_t* searched);

  Stats stats() const;
  void reset_stats();

  // 换机续训用：导出/恢复各局随机数发生器的**完整**状态。
  // 用标准库的流序列化，恢复后随机流逐位一致。
  std::vector<std::string> rng_state() const;
  void set_rng_state(const std::vector<std::string>& state);

 private:
  struct Sample {
    std::vector<uint8_t> board;
    std::vector<float> policy;
    uint8_t to_move = BLACK;
    int32_t history[HISTORY_PLANES] = {-1, -1, -1, -1};
    int32_t move_number = 0;
    int32_t ply = 0;
    float root_value = 0.0f;
    // 失误挖掘信号：搜索认定的最优手，与零搜索策略会选的那手，价值差多少。
    // 差得越大，说明这个局面上「没有搜索就会走错」越严重。
    float blunder_gap = 0.0f;
    float value = 0.0f;
    int32_t plies_remaining = 0;
    int32_t next_move = -1;
    uint8_t searched = 1;
  };

  // 每局一个槽位。统计量与产出样本都存在槽位内部，工作线程之间因此互不接触，
  // 全程无锁；汇总只在主线程调用 stats()/drain() 时做。
  struct Slot {
    std::unique_ptr<Position> pos;
    std::unique_ptr<MctsTree> tree;
    std::mt19937_64 rng;
    std::vector<Sample> pending;   // 本局已记录、等待填入对局结果的样本
    std::vector<Sample> finished;  // 已完成对局的样本，等待 drain
    std::vector<int32_t> visit_counts;
    std::vector<float> child_values;
    std::vector<float> root_policy;  // 加噪前的网络先验，供零搜索落子使用
    std::vector<float> noise;
    Stats stats;

    int root_ply = 0;
    int sims_target = 0;
    int sims_done = 0;
    int descent_depth = 0;
    int leaf = -1;
    bool leaf_needs_eval = false;
    float leaf_terminal_value = 0.0f;
    bool full_search = true;
    bool raw_policy_game = false;
    bool disable_resign = false;
    uint8_t would_resign_side = 0;  // 审计局里「本该认输」的一方

    // ── 尾段重搜 ──
    // undo() 会把 moves_ 一起弹掉，所以回退之前必须先把着法列表拷出来，
    // 否则重搜时就取不到「这一手之后对手怎么应」了。
    bool researching = false;
    std::vector<int32_t> game_moves;
    Outcome game_outcome = Outcome::ONGOING;
    int total_plies = 0;
  };

  void start_game(Slot& slot);
  void play_random_opening(Slot& slot, int plies);
  void start_move(Slot& slot);
  void finish_move(Slot& slot);
  void finish_game(Slot& slot, Outcome outcome, bool by_resign);
  void begin_research(Slot& slot);
  void start_research_move(Slot& slot);
  void finish_research_move(Slot& slot);
  void flush_game(Slot& slot);
  int choose_move(Slot& slot);

  SelfPlayConfig cfg_;
  std::unique_ptr<Rules> rules_;
  std::vector<Slot> slots_;
  std::unique_ptr<ThreadPool> pool_;
};

}  // namespace gi
