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

//Definition of input features HalfKAS2v1 of NNUE evaluation function

#include "half_kas2v1.h"
#include "index_list.h"

namespace Stockfish::Eval::NNUE::Features {

    // Orient a square according to perspective (rotate the board 180° for black)
    // Important note for "halfka": this arch was designed with "flip" in mind
    // although it still is untested which approach is better.
    // this has to stay until we find a better arch that works with "flip".
    // allows us to use current master net for gensfen (primarily needed for higher quality data)
    inline Square orient(Color perspective, Square s) {
        return Square(int(s) ^ (bool(perspective) * SQ_A8));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    template <Side AssociatedKing>
    inline IndexType HalfKAS2v1<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq,
        Bitboard special) {

        bool is_special = special & (1ull << s);
        return IndexType(orient(perspective, s) + kpp_board_index[perspective][pc] + PS_END2 * ksq)
            + PS_END2 * SQUARE_NB * is_special;
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void HalfKAS2v1<AssociatedKing>::AppendActiveIndices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard special = pos.state()->special;

        Bitboard bb = pos.pieces();
        while (bb) {
            Square s = pop_lsb(&bb);
            active->push_back(make_index(perspective, s, pos.piece_on(s), ksq, special));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void HalfKAS2v1<AssociatedKing>::AppendChangedIndices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        auto& st = *pos.state();
        Bitboard prev_special = st.previous->special;
        Bitboard curr_special = st.special;
        auto& dp = st.dirtyPiece;

        Bitboard updated = 0;
        for (int i = 0; i < dp.dirty_num; ++i) {
            Piece pc = dp.piece[i];

            if (dp.from[i] != SQ_NONE)
            {
                updated |= dp.from[i];
                removed->push_back(make_index(perspective, dp.from[i], pc, ksq, prev_special));
            }

            if (dp.to[i] != SQ_NONE)
            {
                updated |= dp.to[i];
                added->push_back(make_index(perspective, dp.to[i], pc, ksq, curr_special));
            }
        }

        Bitboard special_diff = prev_special ^ curr_special;
        Bitboard affected_pieces = pos.pieces() & ~updated & special_diff;
        while (affected_pieces)
        {
            Square s = pop_lsb(&affected_pieces);
            Piece pc = pos.piece_on(s);
            auto r = make_index(perspective, s, pc, ksq, prev_special);
            auto a = make_index(perspective, s, pc, ksq, curr_special);
            if (r != a)
            {
                removed->push_back(r);
                added->push_back(a);
            }
        }
    }

    template class HalfKAS2v1<Side::kFriend>;

}  // namespace Eval::NNUE::Features
