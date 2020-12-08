/*
    Stockfish, a UCI chess playing engine derived from Glaurung 2.1
    Copyright (C) 2004-2020 The Stockfish developers (see AUTHORS file)

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

#ifndef NNUE_FEATURES_PAWN_CONV_9_H_INCLUDED
#define NNUE_FEATURES_PAWN_CONV_9_H_INCLUDED

#include "features_common.h"

#include "evaluate.h"

#include <cstdint>

//Definition of input features HalfKPK of NNUE evaluation function
namespace Eval::NNUE::Features {

    template <Side AssociatedKing>
    class PawnConv9 {

    public:
        static constexpr uint64_t conv_magics[SQUARE_NB] = {
  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0x80200400800c0000ULL,

  0x1010020041009300ULL,

  0x8010020020800ULL,

  0x2004008010840000ULL,

  0x2004008000440ULL,

  0x901002004080a2aULL,

  0,

  0,

  0x1448200400810000ULL,

  0x421900200440508ULL,

  0x6200080100204544ULL,

  0x8040080100000ULL,

  0x4020040085000ULL,

  0x810020040080ULL,

  0,

  0,

  0x401042004008008ULL,

  0x8001002004014ULL,

  0x2520801002210ULL,

  0x8900400801000ULL,

  0x2088010200400880ULL,

  0x1000100200401ULL,

  0,

  0,

  0x1040829a20040081ULL,

  0x8008000010020040ULL,

  0x101080c08010020ULL,

  0x9084000004008011ULL,

  0x90000842004008ULL,

  0x8101002004ULL,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,
        };

        static constexpr uint64_t conv_masks[SQUARE_NB] = {
  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0x7070700ULL,

  0xe0e0e00ULL,

  0x1c1c1c00ULL,

  0x38383800ULL,

  0x70707000ULL,

  0xe0e0e000ULL,

  0,

  0,

  0x707070000ULL,

  0xe0e0e0000ULL,

  0x1c1c1c0000ULL,

  0x3838380000ULL,

  0x7070700000ULL,

  0xe0e0e00000ULL,

  0,

  0,

  0x70707000000ULL,

  0xe0e0e000000ULL,

  0x1c1c1c000000ULL,

  0x383838000000ULL,

  0x707070000000ULL,

  0xe0e0e0000000ULL,

  0,

  0,

  0x7070700000000ULL,

  0xe0e0e00000000ULL,

  0x1c1c1c00000000ULL,

  0x38383800000000ULL,

  0x70707000000000ULL,

  0xe0e0e000000000ULL,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,
        };

        static constexpr uint16_t conv_square_idx[SQUARE_NB] = {
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          2,
          3,
          4,
          5,
          0,
          0,
          6,
          7,
          8,
          9,
          10,
          11,
          0,
          0,
          12,
          13,
          14,
          15,
          16,
          17,
          0,
          0,
          18,
          19,
          20,
          21,
          22,
          23,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
        };

        static constexpr Square conv_squares[24] = {
            SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3,
            SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4,
            SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5,
            SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6,
        };

        // Feature name
        static constexpr const char* kName = (AssociatedKing == Side::kFriend) ?
            "PawnConv9(Friend)" : "PawnConv9(Enemy)";

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t kHashValue =
            0x5F134CB9u ^ (AssociatedKing == Side::kFriend);

        static constexpr IndexType kConvStates = 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3 * 3;

        // Number of feature dimensions
        static constexpr IndexType kDimensions =
            4 * 6 * kConvStates;

        // Maximum number of simultaneously active features
        static constexpr IndexType kMaxActiveDimensions = 4 * 6;

        // Trigger for full calculation instead of difference calculation
        static constexpr TriggerEvent kRefreshTrigger = TriggerEvent::kAnyPawnMoved;

        // Get a list of indices for active features
        static void append_active_indices(
            const Position& pos,
            Color perspective,
            IndexList* active);

        // Get a list of indices for recently changed features
        static void append_changed_indices(
            const Position& pos,
            Color perspective,
            IndexList* removed,
            IndexList* added);

    private:
        // Index of a feature for a given king position and another piece on some square
        static IndexType make_index(
            Color perspective,
            Bitboard our_pawns,
            Bitboard their_pawns,
            Square s);
    };

}  // namespace Eval::NNUE::Features

#endif // #ifndef NNUE_FEATURES_HALF_KA_H_INCLUDED
