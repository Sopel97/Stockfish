#include "imbalance.h"
#include "index_list.h"

// Definition of input feature A of NNUE evaluation function
namespace Eval::NNUE::Features {

    template <PieceType Pt>
    void add(
        const Position& pos,
        Color perspective,
        IndexList* active,
        IndexType id)
    {
        const IndexType wc = pos.count<Pt>(perspective);
        const IndexType bc = pos.count<Pt>(~perspective);
        active->push_back(
            id * Imbalance::kNumDistinctPieceCounts * Imbalance::kNumDistinctPieceCounts
            + wc * Imbalance::kNumDistinctPieceCounts
            + bc);
    }

    // Get a list of indices with a value of 1 among the features
    void Imbalance::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        add<PAWN>(pos, perspective, active, 0);
        add<KNIGHT>(pos, perspective, active, 1);
        add<BISHOP>(pos, perspective, active, 2);
        add<ROOK>(pos, perspective, active, 3);
        add<QUEEN>(pos, perspective, active, 4);
    }

    // Get a list of indices whose values ​​have changed from the previous one in the feature quantity
    void Imbalance::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

    }

}  // namespace Eval::NNUE::Features
