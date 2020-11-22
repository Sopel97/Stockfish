#include "gensfen.h"

#include "sfen_writer.h"
#include "packed_sfen.h"
#include "opening_book.h"

#include "misc.h"
#include "position.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#include "extra/nnue_data_binpack_format.h"

#include "nnue/evaluate_nnue.h"
#include "nnue/evaluate_nnue_learner.h"

#include "syzygy/tbprobe.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <unordered_set>

using namespace std;

namespace Learner
{
    // Class to generate sfen with multiple threads
    struct Gensfen
    {
        struct Params
        {
            // Min and max depths for search during gensfen
            int search_depth_min = 3;
            int search_depth_max = -1;

            // Number of the nodes to be searched.
            // 0 represents no limits.
            uint64_t nodes = 0;

            // Upper limit of evaluation value of generated situation
            int eval_limit = 3000;

            // For when using multi pv instead of random move.
            // random_multi_pv is the number of candidates for MultiPV.
            // When adopting the move of the candidate move, the difference
            // between the evaluation value of the move of the 1st place
            // and the evaluation value of the move of the Nth place is.
            // Must be in the range random_multi_pv_diff.
            // random_multi_pv_depth is the search depth for MultiPV.
            int random_multi_pv = 0;
            int random_multi_pv_diff = 32000;
            int random_multi_pv_depth = -1;

            // The minimum and maximum ply (number of steps from
            // the initial phase) of the sfens to write out.
            int write_minply = 16;
            int write_maxply = 400;

            uint64_t save_every = std::numeric_limits<uint64_t>::max();

            std::string output_file_name = "generated_kifu";

            SfenOutputType sfen_format = SfenOutputType::Binpack;

            std::string seed;

            bool write_out_draw_game_in_training_data_generation = true;

            uint64_t num_threads;

            std::string book;

            void enforce_constraints()
            {
                search_depth_max = std::max(search_depth_min, search_depth_max);
                if (random_multi_pv_depth <= 0)
                {
                    random_multi_pv_depth = search_depth_min;
                }

                // Limit the maximum to a one-stop score. (Otherwise you might not end the loop)
                eval_limit = std::min(eval_limit, (int)mate_in(2));

                save_every = std::max(save_every, REPORT_STATS_EVERY);

                num_threads = Options["Threads"];
            }
        };

        static constexpr uint64_t REPORT_DOT_EVERY = 5000;
        static constexpr uint64_t REPORT_STATS_EVERY = 200000;
        static_assert(REPORT_STATS_EVERY % REPORT_DOT_EVERY == 0);

        Gensfen(
            const Params& prm
        ) :
            params(prm),
            prng(prm.seed),
            sfen_writer(prm.output_file_name, prm.num_threads, prm.save_every, prm.sfen_format)
        {

            if (!prm.book.empty())
            {
                opening_book = open_opening_book(prm.book, prng);
                if (opening_book == nullptr)
                {
                    std::cout << "WARNING: Failed to open opening book " << prm.book << ". Falling back to startpos.\n";
                }
            }

            // Output seed to veryfy by the user if it's not identical by chance.
            std::cout << prng << std::endl;
        }

        void generate(uint64_t limit);

    private:
        Params params;

        PRNG prng;

        std::mutex stats_mutex;
        TimePoint last_stats_report_time;

        // sfen exporter
        SfenWriter sfen_writer;

        SynchronizedRegionLogger::Region out;

        std::unique_ptr<OpeningBook> opening_book;

        static void set_gensfen_search_limits();

        void generate_worker(
            Thread& th,
            std::atomic<uint64_t>& counter,
            uint64_t limit);

        optional<int8_t> get_current_game_result(
            Position& pos,
            const vector<int>& move_hist_scores) const;

        optional<Move> choose_random_move(
            Position& pos,
            int ply);

        bool commit_psv(
            Thread& th,
            PSVector& sfens,
            int8_t lastTurnIsWin,
            std::atomic<uint64_t>& counter,
            uint64_t limit,
            Color result_color);

        void report(uint64_t done, uint64_t new_done);

        void maybe_report(uint64_t done);
    };

    void Gensfen::set_gensfen_search_limits()
    {
        // About Search::Limits
        // Be careful because this member variable is global and affects other threads.
        auto& limits = Search::Limits;

        // Make the search equivalent to the "go infinite" command. (Because it is troublesome if time management is done)
        limits.infinite = true;

        // Since PV is an obstacle when displayed, erase it.
        limits.silent = true;

        // If you use this, it will be compared with the accumulated nodes of each thread. Therefore, do not use it.
        limits.nodes = 0;

        // depth is also processed by the one passed as an argument of Learner::search().
        limits.depth = 0;
    }

    void Gensfen::generate(uint64_t limit)
    {
        last_stats_report_time = 0;

        set_gensfen_search_limits();

        std::atomic<uint64_t> counter{0};
        Threads.execute_with_workers([&counter, limit, this](Thread& th) {
            generate_worker(th, counter, limit);
        });
        Threads.wait_for_workers_finished();

        sfen_writer.flush();

        if (limit % REPORT_STATS_EVERY != 0)
        {
            report(limit, limit % REPORT_STATS_EVERY);
        }

        std::cout << std::endl;
    }

    void Gensfen::generate_worker(
        Thread& th,
        std::atomic<uint64_t>& counter,
        uint64_t limit)
    {
        // For the time being, it will be treated as a draw
        // at the maximum number of steps to write.
        // Maximum StateInfo + Search PV to advance to leaf buffer
        std::vector<StateInfo, AlignedAllocator<StateInfo>> states(
            params.write_maxply + MAX_PLY /* == search_depth_min + α */);

        StateInfo si;

        // end flag
        bool quit = false;

        // repeat until the specified number of times
        while (!quit)
        {
            // It is necessary to set a dependent thread for Position.
            // When parallelizing, Threads (since this is a vector<Thread*>,
            // Do the same for up to Threads[0]...Threads[thread_num-1].
            auto& pos = th.rootPos;
            if (opening_book != nullptr)
            {
                auto& fen = opening_book->next_fen();
                pos.set(fen, false, &si, &th);
            }
            else
            {
                pos.set(StartFEN, false, &si, &th);
            }

            // Vector for holding the sfens in the current simulated game.
            PSVector packed_sfens;
            packed_sfens.reserve(params.write_maxply + MAX_PLY);

            // Save history of move scores for adjudication
            vector<int> move_hist_scores;

            auto flush_psv = [&](int8_t result) {
                quit = commit_psv(th, packed_sfens, result, counter, limit, pos.side_to_move());
            };

            for (int ply = 0; ; ++ply)
            {
                // Current search depth
                const int depth = params.search_depth_min + (int)prng.rand(params.search_depth_max - params.search_depth_min + 1);

                // Starting search calls init_for_search
                auto [search_value, search_pv] = Search::search(pos, depth, 1, params.nodes);

                // This has to be performed after search because it needs to know
                // rootMoves which are filled in init_for_search.
                const auto result = get_current_game_result(pos, move_hist_scores);
                if (result.has_value())
                {
                    flush_psv(result.value());
                    break;
                }

                // Always adjudivate by eval limit.
                // Also because of this we don't have to check for TB/MATE scores
                if (abs(search_value) >= params.eval_limit)
                {
                    flush_psv((search_value >= params.eval_limit) ? 1 : -1);
                    break;
                }

                // In case there is no PV and the game was not ended here
                // there is nothing we can do, we can't continue the game,
                // we don't know the result, so discard this game.
                if (search_pv.empty())
                {
                    break;
                }

                // Save the move score for adjudication.
                move_hist_scores.push_back(search_value);

                // Discard stuff before write_minply is reached
                // because it can harm training due to overfitting.
                // Initial positions would be too common.
                if (ply >= params.write_minply)
                {
                    packed_sfens.emplace_back(PackedSfenValue());

                    auto& psv = packed_sfens.back();

                    // Here we only write the position data.
                    // Result is added after the whole game is done.
                    pos.sfen_pack(psv.sfen);

                    if (pos.checkers())
                    {
                        psv.score = search_value;
                    }
                    else
                    {
                        psv.score = Eval::evaluate_raw(pos);
                    }

                    const auto rootColor = pos.side_to_move();
                    int old_ply = ply;
                    bool should_update = true;
                    for (auto m : search_pv)
                    {
                        if (pos.capture_or_promotion(m))
                        {
                            should_update = false;
                        }
                        pos.do_move(m, states[ply++]);
                        if (pos.checkers())
                        {
                            should_update = false;
                        }
                        if (should_update)
                        {
                            psv.score = (rootColor == pos.side_to_move()) ? Eval::evaluate_raw(pos) : -Eval::evaluate_raw(pos);
                        }
                    }

                    // Get back to the game
                    for (auto it = search_pv.rbegin(); it != search_pv.rend(); ++it)
                    {
                        pos.undo_move(*it);
                    }

                    ply = old_ply;
                    psv.move = search_pv[0];
                    psv.gamePly = ply;
                }

                // Update the next move according to best search result or random move.
                auto random_move = choose_random_move(pos, ply);
                const Move next_move = random_move.has_value() ? *random_move : search_pv[0];

                // We don't have the whole game yet, but it ended,
                // so the writing process ends and the next game starts.
                // This shouldn't really happen.
                if (!is_ok(next_move))
                {
                    break;
                }

                // Do move.
                pos.do_move(next_move, states[ply]);
            }
        }
    }

    optional<int8_t> Gensfen::get_current_game_result(
        Position& pos,
        const vector<int>& move_hist_scores) const
    {
        // Variables for draw adjudication.
        // Todo: Make this as an option.

        // start the adjudication when ply reaches this value
        constexpr int adj_draw_ply = 80;

        // 4 move scores for each side have to be checked
        constexpr int adj_draw_cnt = 8;

        // move score in CP
        constexpr int adj_draw_score = 0;

        // For the time being, it will be treated as a
        // draw at the maximum number of steps to write.
        const int ply = move_hist_scores.size();

        // has it reached the max length or is a draw
        if (ply >= params.write_maxply || pos.is_draw(ply))
        {
            return 0;
        }

        if (pos.this_thread()->rootInTB)
        {
            Tablebases::ProbeState probe_state;
            Tablebases::WDLScore wdl = Tablebases::probe_wdl(pos, &probe_state);
            if (wdl == Tablebases::WDLScore::WDLWin) {
                return 1;
            } else if (wdl == Tablebases::WDLScore::WDLLoss) {
                return -1;
            } else {
                return 0;
            }
        }

        if(pos.this_thread()->rootMoves.empty())
        {
            // If there is no legal move
            return pos.checkers()
                ? -1 /* mate */
                : 0 /* stalemate */;
        }

        // Adjudicate game to a draw if the last 4 scores of each engine is 0.
        if (ply >= adj_draw_ply)
        {
            int num_cons_plies_within_draw_score = 0;
            bool is_adj_draw = false;

            for (auto it = move_hist_scores.rbegin();
                it != move_hist_scores.rend(); ++it)
            {
                if (abs(*it) <= adj_draw_score)
                {
                    num_cons_plies_within_draw_score++;
                }
                else
                {
                    // Draw scores must happen on consecutive plies
                    break;
                }

                if (num_cons_plies_within_draw_score >= adj_draw_cnt)
                {
                    is_adj_draw = true;
                    break;
                }
            }

            if (is_adj_draw)
            {
                return 0;
            }
        }

        if (pos.count<ALL_PIECES>() <= 4)
        {
            int num_pieces = pos.count<ALL_PIECES>();

            // (1) KvK
            if (num_pieces == 2)
            {
                return 0;
            }

            // (2) KvK + 1 minor piece
            if (num_pieces == 3)
            {
                int minor_pc = pos.count<BISHOP>(WHITE) + pos.count<KNIGHT>(WHITE) +
                    pos.count<BISHOP>(BLACK) + pos.count<KNIGHT>(BLACK);
                if (minor_pc == 1)
                {
                    return 0;
                }
            }

            // (3) KBvKB, bishops of the same color
            else if (num_pieces == 4)
            {
                if (pos.count<BISHOP>(WHITE) == 1 && pos.count<BISHOP>(BLACK) == 1)
                {
                    // Color of bishops is black.
                    if ((pos.pieces(WHITE, BISHOP) & DarkSquares)
                        && (pos.pieces(BLACK, BISHOP) & DarkSquares))
                    {
                        return 0;
                    }
                    // Color of bishops is white.
                    if ((pos.pieces(WHITE, BISHOP) & ~DarkSquares)
                        && (pos.pieces(BLACK, BISHOP) & ~DarkSquares))
                    {
                        return 0;
                    }
                }
            }
        }

        return nullopt;
    }

    optional<Move> Gensfen::choose_random_move(Position& pos, int ply)
    {
        if (params.random_multi_pv == 0 || prng.rand(10) > 1)
        {
            return nullopt;
        }

        Search::search(pos, params.random_multi_pv_depth, params.random_multi_pv);

        // Select one from the top N hands of root Moves
        auto& rm = pos.this_thread()->rootMoves;

        uint64_t s = min((uint64_t)rm.size(), (uint64_t)params.random_multi_pv);
        for (uint64_t i = 1; i < s; ++i)
        {
            // The difference from the evaluation value of rm[0] must
            // be within the range of random_multi_pv_diff.
            // It can be assumed that rm[x].score is arranged in descending order.
            if (rm[0].score > rm[i].score + params.random_multi_pv_diff / (ply / 8 + 1))
            {
                s = i;
                break;
            }
        }

        return rm[prng.rand(s)].pv[0];
    }

    // Write out the phases loaded in sfens to a file.
    // result: win/loss in the next phase after the final phase in sfens
    // 1 when winning. -1 when losing. Pass 0 for a draw.
    // Return value: true if the specified number of
    // sfens has already been reached and the process ends.
    bool Gensfen::commit_psv(
        Thread& th,
        PSVector& sfens,
        int8_t result,
        std::atomic<uint64_t>& counter,
        uint64_t limit,
        Color result_color)
    {
        if (!params.write_out_draw_game_in_training_data_generation && result == 0)
        {
            // We didn't write anything so why quit.
            return false;
        }

        auto side_to_move_from_sfen = [](auto& sfen){
            return (Color)(sfen.sfen.data[0] & 1);
        };

        // From the final stage (one step before) to the first stage, give information on the outcome of the game for each stage.
        // The phases stored in sfens are assumed to be continuous (in order).
        for (auto it = sfens.rbegin(); it != sfens.rend(); ++it)
        {
            // The side to move is packed as the lowest bit of the first byte
            const Color side_to_move = side_to_move_from_sfen(*it);
            it->game_result = side_to_move == result_color ? result : -result;
        }

        // Write sfens in move order to make potential compression easier
        for (auto& sfen : sfens)
        {
            // Return true if there is already enough data generated.
            const auto iter = counter.fetch_add(1);
            if (iter >= limit)
                return true;

            // because `iter` was done, now we do one more
            maybe_report(iter + 1);

            // Write out one sfen.
            sfen_writer.write(th.thread_idx(), sfen);
        }

        return false;
    }

    void Gensfen::report(uint64_t done, uint64_t new_done)
    {
        const auto now_time = now();
        const TimePoint elapsed = now_time - last_stats_report_time + 1;

        out
            << endl
            << done << " sfens, "
            << new_done * 1000 / elapsed << " sfens/second, "
            << "at " << now_string() << sync_endl;

        last_stats_report_time = now_time;

        out = sync_region_cout.new_region();
    }

    void Gensfen::maybe_report(uint64_t done)
    {
        if (done % REPORT_DOT_EVERY == 0)
        {
            std::lock_guard lock(stats_mutex);

            if (last_stats_report_time == 0)
            {
                last_stats_report_time = now();
                out = sync_region_cout.new_region();
            }

            if (done != 0)
            {
                out << '.';

                if (done % REPORT_STATS_EVERY == 0)
                {
                    report(done, REPORT_STATS_EVERY);
                }
            }
        }
    }

    // Command to generate a game record
    void gensfen(istringstream& is)
    {
        // Number of generated game records default = 8 billion phases (Ponanza specification)
        uint64_t loop_max = 8000000000UL;

        Gensfen::Params params;

        // Add a random number to the end of the file name.
        bool random_file_name = false;
        std::string sfen_format = "binpack";

        string token;
        while (true)
        {
            token = "";
            is >> token;
            if (token == "")
                break;

            if (token == "depth")
                is >> params.search_depth_min;
            else if (token == "depth2")
                is >> params.search_depth_max;
            else if (token == "nodes")
                is >> params.nodes;
            else if (token == "loop")
                is >> loop_max;
            else if (token == "output_file_name")
                is >> params.output_file_name;
            else if (token == "eval_limit")
                is >> params.eval_limit;
            else if (token == "random_multi_pv")
                is >> params.random_multi_pv;
            else if (token == "random_multi_pv_diff")
                is >> params.random_multi_pv_diff;
            else if (token == "random_multi_pv_depth")
                is >> params.random_multi_pv_depth;
            else if (token == "write_minply")
                is >> params.write_minply;
            else if (token == "write_maxply")
                is >> params.write_maxply;
            else if (token == "save_every")
                is >> params.save_every;
            else if (token == "book")
                is >> params.book;
            else if (token == "random_file_name")
                is >> random_file_name;
            // Accept also the old option name.
            else if (token == "use_draw_in_training_data_generation" || token == "write_out_draw_game_in_training_data_generation")
                is >> params.write_out_draw_game_in_training_data_generation;
            else if (token == "sfen_format")
                is >> sfen_format;
            else if (token == "seed")
                is >> params.seed;
            else if (token == "set_recommended_uci_options")
            {
                UCI::setoption("Contempt", "0");
                UCI::setoption("Skill Level", "20");
                UCI::setoption("UCI_Chess960", "false");
                UCI::setoption("UCI_AnalyseMode", "false");
                UCI::setoption("UCI_LimitStrength", "false");
            }
            else
                cout << "ERROR: Ignoring unknown option " << token << endl;
        }

        if (!sfen_format.empty())
        {
            if (sfen_format == "bin")
                params.sfen_format = SfenOutputType::Bin;
            else if (sfen_format == "binpack")
                params.sfen_format = SfenOutputType::Binpack;
            else
                cout << "WARNING: Unknown sfen format `" << sfen_format << "`. Using bin\n";
        }

        if (random_file_name)
        {
            // Give a random number to output_file_name at this point.
            // Do not use std::random_device().  Because it always the same integers on MinGW.
            PRNG r(params.seed);

            // Just in case, reassign the random numbers.
            for (int i = 0; i < 10; ++i)
                r.rand(1);

            auto to_hex = [](uint64_t u) {
                std::stringstream ss;
                ss << std::hex << u;
                return ss.str();
            };

            // I don't want to wear 64bit numbers by accident, so I'next_move going to make a 64bit number 2 just in case.
            params.output_file_name += "_" + to_hex(r.rand<uint64_t>()) + to_hex(r.rand<uint64_t>());
        }

        params.enforce_constraints();

        std::cout << "INFO: Executing gensfen command\n";

        std::cout << "INFO: Parameters:\n";
        std::cout
            << "  - search_depth_min       = " << params.search_depth_min << endl
            << "  - search_depth_max       = " << params.search_depth_max << endl
            << "  - nodes                  = " << params.nodes << endl
            << "  - num sfens to generate  = " << loop_max << endl
            << "  - eval_limit             = " << params.eval_limit << endl
            << "  - num threads (UCI)      = " << params.num_threads << endl
            << "  - random_multi_pv        = " << params.random_multi_pv << endl
            << "  - random_multi_pv_diff   = " << params.random_multi_pv_diff << endl
            << "  - random_multi_pv_depth  = " << params.random_multi_pv_depth << endl
            << "  - write_minply           = " << params.write_minply << endl
            << "  - write_maxply           = " << params.write_maxply << endl
            << "  - book                   = " << params.book << endl
            << "  - output_file_name       = " << params.output_file_name << endl
            << "  - save_every             = " << params.save_every << endl
            << "  - random_file_name       = " << random_file_name << endl
            << "  - write_drawn_games      = " << params.write_out_draw_game_in_training_data_generation << endl;

        // Show if the training data generator uses NNUE.
        Eval::NNUE::verify();

        Threads.main()->ponder = false;

        Gensfen gensfen(params);
        gensfen.generate(loop_max);

        std::cout << "INFO: Gensfen finished." << endl;
    }
}
