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

#include <array>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <set>
#include <utility>

#include "pawn_conv_7.h"
#include "index_list.h"

static uint64_t calc_as_ternary(uint64_t i, int num_bits)
{
    uint64_t tern = 0;
    for (int j = 1; j <= num_bits; ++j)
    {
        tern *= 3;
        tern += (i >> (num_bits - j)) & 1;
    }
    return tern;
}

static std::array<uint32_t, 128> as_ternary = []() {
    std::array<uint32_t, 128> s_as_ternary{};
    for (int i = 0; i < 128; ++i)
    {
        s_as_ternary[i] = calc_as_ternary(i, 7);
    }
    return s_as_ternary;
}();

namespace Eval::NNUE::Features {

    template <Side AssociatedKing>
    inline uint64_t pext_conv(Color perspective, uint64_t bb, int s)
    {
        return ((bb & PawnConv7<AssociatedKing>::conv_masks[perspective][s]) * PawnConv7<AssociatedKing>::conv_magics[perspective][s]) >> 57ull;
    }

    // Orient a square according to perspective (rotate the board 180° for black)
    // Important note for "halfka": this arch was designed with "flip" in mind
    // although it still is untested which approach is better.
    // this has to stay until we find a better arch that works with "flip".
    // allows us to use current master net for gensfen (primarily needed for higher quality data)
    inline Square orient(Color perspective, Square s) {
        return Square(int(s) ^ (bool(perspective) * 63));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    template <Side AssociatedKing>
    inline IndexType PawnConv7<AssociatedKing>::make_index(
        Color perspective,
        Bitboard our_pawns,
        Bitboard their_pawns,
        Square s) {

        return
            as_ternary[pext_conv<AssociatedKing>(perspective, our_pawns, s)]
            + 2 * as_ternary[pext_conv<AssociatedKing>(perspective, their_pawns, s)]
            + (kConvStates * conv_square_idx[orient(perspective, s)]);
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void PawnConv7<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        auto c = AssociatedKing == Side::kFriend ? perspective : ~perspective;

        auto our_pawns = pos.pieces(c, PAWN);
        auto their_pawns = pos.pieces(~c, PAWN);
        auto all_pawns = our_pawns | their_pawns;

        auto our_pawns_cpy = our_pawns;
        while (our_pawns_cpy)
        {
            Square s = pop_lsb(&our_pawns_cpy);
            active->push_back(make_index(perspective, our_pawns, their_pawns, s));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void PawnConv7<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        return;
    }

    template class PawnConv7<Side::kFriend>;
    template class PawnConv7<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
