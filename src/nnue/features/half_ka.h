/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//Definition of input features HalfKP of NNUE evaluation function

#ifndef NNUE_FEATURES_HALF_KA_H_INCLUDED
#define NNUE_FEATURES_HALF_KA_H_INCLUDED

#include "../../evaluate.h"
#include "features_common.h"

namespace Stockfish::Eval::NNUE::Features {

  // Feature HalfKP: Combination of the position of own king
  // and the position of pieces other than kings
  template <Side AssociatedKing>
  class HalfKA {
   public:
    static constexpr int KingBuckets[SQUARE_NB] = {
      24, 25, 26, 27, 28, 29, 30, 31,
      16, 17, 18, 19, 20, 21, 22, 23,
      12, 12, 13, 13, 14, 14, 15, 15,
       8,  8,  9,  9, 10, 10, 11, 11,
       4,  4,  5,  5,  6,  6,  7,  7,
       4,  4,  5,  5,  6,  6,  7,  7,
       0,  0,  1,  1,  2,  2,  3,  3,
       0,  0,  1,  1,  2,  2,  3,  3
    };

    // Feature name
    static constexpr const char* kName = "HalfKA(Friend)";
    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t kHashValue =
        0x5f134cb9u ^ (AssociatedKing == Side::kFriend);
    // Number of feature dimensions
    static constexpr IndexType kDimensions =
        static_cast<IndexType>(SQUARE_NB) * static_cast<IndexType>(PS_END2) / 2;
    // Maximum number of simultaneously active features
    static constexpr IndexType kMaxActiveDimensions = 32;
    // Trigger for full calculation instead of difference calculation
    static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kFriendKingMoved;

    // Get a list of indices for active features
    static void AppendActiveIndices(const Position& pos, Color perspective,
                                    IndexList* active);

    // Get a list of indices for recently changed features
    static void AppendChangedIndices(const Position& pos, const DirtyPiece& dp, Color perspective,
                                     IndexList* removed, IndexList* added);
  };

}  // namespace Stockfish::Eval::NNUE::Features

#endif // #ifndef NNUE_FEATURES_HALF_KA_H_INCLUDED
