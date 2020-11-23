#ifndef _NNUE_FEATURES_IMBALANCE_H_
#define _NNUE_FEATURES_IMBALANCE_H_

#include "features_common.h"

#include "evaluate.h"

namespace Eval::NNUE::Features {

    // Feature P: PieceSquare of pieces other than balls
    class Imbalance {
    public:
        // feature quantity name
        static constexpr const char* kName = "Imbalance";

        // Hash value embedded in the evaluation function file
        static constexpr std::uint32_t kHashValue = 0x7ABC111Cu;

        // 10 pieces max, but can be 0.
        static constexpr IndexType kNumDistinctPieceCounts = 11;

        // number of feature dimensions
        // PAWN, KNIGHT, BISHOP, ROOK, QUEEN
        static constexpr IndexType kDimensions = 5 * kMaxPieceCount * kMaxPieceCount;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions = 5;

        // Timing of full calculation instead of difference calculation
        // TODO: replace with any capture or promotion
        static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPieceMoved;

        // Get a list of indices with a value of 1 among the features
        static void append_active_indices(
            const Position& pos,
            Color perspective,
            IndexList* active);

        // Get a list of indices whose values ​​have changed from the previous one in the feature quantity
        static void append_changed_indices(
            const Position& pos,
            Color perspective,
            IndexList* removed,
            IndexList* added);
    };

}  // namespace Eval::NNUE::Features

#endif // #ifndef _NNUE_FEATURES_UNION_P_K_H_
