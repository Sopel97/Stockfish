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

#ifndef NNUE_FEATURES_PAWN_CONV_7_H_INCLUDED
#define NNUE_FEATURES_PAWN_CONV_7_H_INCLUDED

#include "features_common.h"

#include "evaluate.h"

#include <cstdint>
#include <vector>

//Definition of input features HalfKPK of NNUE evaluation function
namespace Eval::NNUE::Features {

    template <Side AssociatedKing>
    class PawnConv7 {

    public:

        struct Factors
        {
            IndexType conv_id = IndexType(-1);
            IndexType num_piece_ids = 0;
            IndexType piece_ids[7]{};
            Square sq = SQ_NONE;
        };

        static const std::vector<Factors> factors_lookup;

        static constexpr uint64_t conv_magics[SQUARE_NB] = {
  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0,

  0x4020100800c40401ULL,

  0x402200200480050ULL,

  0x1008010200100020ULL,

  0x804008048004808ULL,

  0x204004028020000ULL,

  0x3040220000202120ULL,

  0x1021000118a001ULL,

  0x502004000011000ULL,

  0x40a01000082000ULL,

  0x20100210100000ULL,

  0x8010080100001084ULL,

  0x20c1008800040078ULL,

  0x804400209200ULL,

  0x2010022001080ULL,

  0x1021005880a4ULL,

  0xc408500000442ULL,

  0x2008081040045000ULL,

  0x800040220010000ULL,

  0x8402020110000000ULL,

  0x42000810800006eULL,

  0x8100840200401280ULL,

  0xa000010200224001ULL,

  0x4c0410080101410ULL,

  0xc201120100400200ULL,

  0x8000108810400028ULL,

  0x828020402200000ULL,

  0x11008010001ULL,

  0xa80880408008083ULL,

  0x4201840040840008ULL,

  0x8081101201002020ULL,

  0x1d0000100801084ULL,

  0x2088188506004000ULL,

  0x4000004588402002ULL,

  0x2c02a802042000ULL,

  0x100500d040810800ULL,

  0x1400000804008cULL,

  0x808884f002040040ULL,

  0xc052010020ULL,

  0x4222400081008010ULL,

  0x8085060040ULL,

  0x8000400148402010ULL,

  0x8040218020420ULL,

  0x3009002010210ULL,

  0x861000a0008108ULL,

  0x4001008900004084ULL,

  0x2020801201104022ULL,

  0x1108010000002011ULL,

  0x84085ULL,

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

  0x30202ULL,

  0x70505ULL,

  0xe0a0aULL,

  0x1c1414ULL,

  0x382828ULL,

  0x705050ULL,

  0xe0a0a0ULL,

  0xc04040ULL,

  0x3020200ULL,

  0x7050500ULL,

  0xe0a0a00ULL,

  0x1c141400ULL,

  0x38282800ULL,

  0x70505000ULL,

  0xe0a0a000ULL,

  0xc0404000ULL,

  0x302020000ULL,

  0x705050000ULL,

  0xe0a0a0000ULL,

  0x1c14140000ULL,

  0x3828280000ULL,

  0x7050500000ULL,

  0xe0a0a00000ULL,

  0xc040400000ULL,

  0x30202000000ULL,

  0x70505000000ULL,

  0xe0a0a000000ULL,

  0x1c1414000000ULL,

  0x382828000000ULL,

  0x705050000000ULL,

  0xe0a0a0000000ULL,

  0xc04040000000ULL,

  0x3020200000000ULL,

  0x7050500000000ULL,

  0xe0a0a00000000ULL,

  0x1c141400000000ULL,

  0x38282800000000ULL,

  0x70505000000000ULL,

  0xe0a0a000000000ULL,

  0xc0404000000000ULL,

  0x302020000000000ULL,

  0x705050000000000ULL,

  0xe0a0a0000000000ULL,

  0x1c14140000000000ULL,

  0x3828280000000000ULL,

  0x7050500000000000ULL,

  0xe0a0a00000000000ULL,

  0xc040400000000000ULL,

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
          0, 0, 0, 0, 0, 0, 0, 0,
          0, 1, 2, 3, 4, 5, 6, 7,
          8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23,
          24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, 37, 38, 39,
          40, 41, 42, 43, 44, 45, 46, 47,
          0, 0, 0, 0, 0, 0, 0, 0,
        };

        static constexpr Square conv_squares[48] = {
            SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
            SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
            SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
            SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
            SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
            SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
        };

        // Feature name
        static constexpr const char* kName = (AssociatedKing == Side::kFriend) ?
            "PawnConv7(Friend)" : "PawnConv7(Enemy)";

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t kHashValue =
            0x5F134CB9u ^ (AssociatedKing == Side::kFriend);

        static constexpr IndexType kConvStates = 3 * 3 * 3 * 3 * 3 * 3 * 3;

        // Number of feature dimensions
        static constexpr IndexType kDimensions =
            6 * 8 * kConvStates;

        // Maximum number of simultaneously active features
        static constexpr IndexType kMaxActiveDimensions = 8;

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
            Bitboard our_pawns,
            Bitboard their_pawns,
            Square s);
    };

}  // namespace Eval::NNUE::Features

#endif // #ifndef NNUE_FEATURES_HALF_KA_H_INCLUDED
