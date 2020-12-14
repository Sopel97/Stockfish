#include "imbalance.h"
#include "index_list.h"

#include <algorithm>
#include <cmath>

//Definition of input feature P of NNUE evaluation function
namespace Eval::NNUE::Features {

    // Orient a square according to perspective (rotate the board 180° for black)
    // this has to stay until we find a better arch that works with "flip".
    // allows us to use current master net for gensfen (primarily needed for higher quality data)
    static inline Square orient(Color perspective, Square s) {
        return Square(int(s) ^ (bool(perspective) * 63));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    static inline IndexType make_index(
        PieceType pt0, PieceType pt1,
        int diff0, int diff1) {

        const IndexType di0 = diff0 + Imbalance::kMaxPieceCountDiff;
        const IndexType di1 = diff1 + Imbalance::kMaxPieceCountDiff;

        const IndexType di = (Imbalance::kMaxPieceCountDiff * 2 + 1) * di0 + di1;
        const IndexType pti = PIECE_TYPE_NB * pt0 + pt1;

        return (PIECE_TYPE_NB * PIECE_TYPE_NB) * di + pti;
    }

    // Get a list of indices with a value of 1 among the features
    void Imbalance::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        int diffs[PIECE_TYPE_NB];
        for (auto pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN})
        {
            const int our_count = popcount(pos.pieces(perspective, pt));
            const int their_count = popcount(pos.pieces(~perspective, pt));
            diffs[pt] = std::clamp(our_count - their_count, -(int)kMaxPieceCountDiff, (int)kMaxPieceCountDiff);
        }

        for (auto pt0 = PAWN; pt0 <= QUEEN; ++pt0)
        {
            for (auto pt1 = pt0; pt1 <= QUEEN; ++pt1)
            {
                if (diffs[pt0] != 0 && diffs[pt1] != 0)
                {
                    active->push_back(make_index(pt0, pt1, diffs[pt0], diffs[pt1]));
                }
            }
        }
    }

    // Get a list of indices whose values ​​have changed from the previous one in the feature quantity
    void Imbalance::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        return;
    }

}  // namespace Eval::NNUE::Features
