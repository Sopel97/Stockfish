#ifndef _NNUE_FEATURES_PAWN_TRAITS_H_
#define _NNUE_FEATURES_PAWN_TRAITS_H_

#include "features_common.h"

#include "evaluate.h"
#include "types.h"
#include "pawns.h"
#include "bitboard.h"

#include <cstdint>

// Definition of input features for pawns of NNUE evaluation function
namespace Eval::NNUE::Features {

    class IndexList;

    namespace PawnTraitType {
        enum : uint32_t
        {
            Scores = 1 << 0,
            Passed = 1 << 1,
            Backward = 1 << 2,
            Doubled = 1 << 3,
            Opposed = 1 << 4,
            Blocked = 1 << 5,
            Stoppers = 1 << 6,
            Lever = 1 << 7,
            LeverPush = 1 << 8,
            Neighbours = 1 << 9,
            Phalanx = 1 << 10,
            Support = 1 << 11,

            All = 0xFFFFFFFFu
        };
    }

    namespace Detail
    {
        void append_active_indices(
            Bitboard& bb,
            IndexType offset,
            IndexList* active);

        void append_score_indices(
            Value score,
            IndexType offset,
            IndexList* active);
    }

    template <uint32_t Traits = PawnTraitType::All>
    class PawnTraits
    {
    public:
        static constexpr const char* kName = "PawnTraits(TODO:ListTraits)";

        static constexpr std::uint32_t kHashValue = 0x1F3A6B33u;

        static constexpr IndexType kScoreBuckets = 32 * 2;

        static constexpr IndexType kNumBitboardTraits =
            + bool(Traits & PawnTraitType::Passed)
            + bool(Traits & PawnTraitType::Backward)
            + bool(Traits & PawnTraitType::Doubled)
            + bool(Traits & PawnTraitType::Opposed)
            + bool(Traits & PawnTraitType::Blocked)
            + bool(Traits & PawnTraitType::Stoppers)
            + bool(Traits & PawnTraitType::Lever)
            + bool(Traits & PawnTraitType::LeverPush)
            + bool(Traits & PawnTraitType::Neighbours)
            + bool(Traits & PawnTraitType::Phalanx)
            + bool(Traits & PawnTraitType::Support);

        static constexpr IndexType kDimensions =
            SQUARE_NB * COLOR_NB * kNumBitboardTraits
            + COLOR_NB * kScoreBuckets * bool(Traits & PawnTraitType::Scores);

        static constexpr IndexType kMaxActiveDimensions = 8 * kNumBitboardTraits + bool(Traits & PawnTraitType::Scores);

        static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPawnMoved;

        static void append_active_indices(
            const Position& pos,
            Color perspective,
            IndexList* active)
        {
            IndexType offset = 0;
            auto* pe = Pawns::probe(pos);

            /* TODO: needs a conversion to a value */ /*
            if constexpr (Traits & PawnTraitType::Scores)
            {
                Detail::append_score_indices(pe->scores[perspective], offset + perspective * kScoreBuckets, active);
                offset += COLOR_NB * kScoreBuckets;
            }
            */

            if constexpr (Traits & PawnTraitType::Passed)
            {
                Detail::append_active_indices(pe->passedPawns[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Backward)
            {
                Detail::append_active_indices(pe->backward[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Doubled)
            {
                Detail::append_active_indices(pe->doubled[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Opposed)
            {
                Detail::append_active_indices(pe->opposed[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Blocked)
            {
                Detail::append_active_indices(pe->blocked[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Stoppers)
            {
                Detail::append_active_indices(pe->stoppers[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Lever)
            {
                Detail::append_active_indices(pe->lever[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::LeverPush)
            {
                Detail::append_active_indices(pe->leverPush[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Neighbours)
            {
                Detail::append_active_indices(pe->neighbours[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Phalanx)
            {
                Detail::append_active_indices(pe->phalanx[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }

            if constexpr (Traits & PawnTraitType::Support)
            {
                Detail::append_active_indices(pe->support[perspective], offset + perspective * SQUARE_NB, active);
                offset += SQUARE_NB * COLOR_NB;
            }
        }

        static void append_changed_indices(
            const Position&,
            Color,
            IndexList*,
            IndexList*)
        {
            // Nothing to do here. But it will be called.
        }
    };

}  // namespace Eval::NNUE::Features

#endif
