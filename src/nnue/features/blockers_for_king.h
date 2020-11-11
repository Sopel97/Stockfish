//Definition of input feature P of NNUE evaluation function

#ifndef _NNUE_FEATURES_BLOCKERS_FOR_KING_H_
#define _NNUE_FEATURES_BLOCKERS_FOR_KING_H_

#include "../../evaluate.h"
#include "features_common.h"
#include "types.h"

namespace Eval::NNUE::Features {

    // Feature P: PieceSquare of pieces other than balls
    class BlockersForKing {
    public:
        // feature quantity name
        static constexpr const char* kName = "BlockersForKing";
        // Hash value embedded in the evaluation function file
        static constexpr std::uint32_t kHashValue = 0x123DA24Bu;
        // number of feature dimensions
        static constexpr IndexType kDimensions = PS_END;
        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions = 32;
        // Timing of full calculation instead of difference calculation
        static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPieceMoved;

        // Get a list of indices with a value of 1 among the features
        static void append_active_indices(const Position& pos, Color perspective,
                                        IndexList* active);

        // Get a list of indices whose values ​​have changed from the previous one in the feature quantity
        static void append_changed_indices(const Position& pos, Color perspective,
                                         IndexList* removed, IndexList* added);

    private:
        // Index of a feature for a given piece on some square
        static IndexType make_index(Color perspective, Square s, Piece pc);
    };

}

#endif
