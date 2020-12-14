#ifndef _NNUE_FEATURES_IMBALANCE_H_
#define _NNUE_FEATURES_IMBALANCE_H_

#include "features_common.h"

#include "evaluate.h"

//Definition of input feature P of NNUE evaluation function
namespace Eval::NNUE::Features {

    // Feature P: PieceSquare of pieces other than balls
    class Imbalance {
    public:
        // feature quantity name
        static constexpr const char* kName = "Imbalance";

        // Hash value embedded in the evaluation function file
        static constexpr std::uint32_t kHashValue = 0x764CFB4Bu;

        static constexpr IndexType kMaxPieceCountDiff = 8;

        // number of feature dimensions
        static constexpr IndexType kDimensions =
            (kMaxPieceCountDiff * 2 + 1)
            * (kMaxPieceCountDiff * 2 + 1)
            * PIECE_TYPE_NB * PIECE_TYPE_NB;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions = (PIECE_TYPE_NB - 1) * PIECE_TYPE_NB / 2;

        // Timing of full calculation instead of difference calculation
        static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kMaterialChanged;

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

#endif
