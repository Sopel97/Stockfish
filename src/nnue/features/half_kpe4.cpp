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

//Definition of input features HalfKPE4 of NNUE evaluation function

#include "half_kpe4.h"
#include "index_list.h"

namespace Eval::NNUE::Features {

    // Orient a square according to perspective (rotate the board 180° for black)
    // this has to stay until we find a better arch that works with "flip".
    // allows us to use current master net for gensfen (primarily needed for higher quality data)
    inline Square orient(Color perspective, Square s) {
        return Square(int(s) ^ (bool(perspective) * 63));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    template <Side AssociatedKing>
    inline IndexType HalfKPE4<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq,
        Bitboard* mobility) {

        const IndexType mobility_index =
            bool(mobility[perspective] & s) * 2
            + bool(mobility[~perspective] & s);

        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective] + PS_END * ksq)
            + (static_cast<IndexType>(SQUARE_NB) * static_cast<IndexType>(PS_END)) * mobility_index;
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void HalfKPE4<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard bb = pos.pieces() & ~pos.pieces(KING);
        while (bb) {
            Square s = pop_lsb(&bb);
            active->push_back(make_index(perspective, s, pos.piece_on(s), ksq, pos.state()->mobility));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void HalfKPE4<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard* curr_mobility = pos.state()->mobility;
        Bitboard* prev_mobility = pos.state()->previous->mobility;

        const auto& dp = pos.state()->dirtyPiece;
        Bitboard updated_ps = 0;
        for (int i = 0; i < dp.dirty_num; ++i) {
            Piece pc = dp.piece[i];

            if (type_of(pc) == KING)
                continue;

            if (dp.from[i] != SQ_NONE)
            {
                updated_ps |= dp.from[i];
                removed->push_back(make_index(perspective, dp.from[i], pc, ksq, prev_mobility));
            }

            if (dp.to[i] != SQ_NONE)
            {
                updated_ps |= dp.to[i];
                added->push_back(make_index(perspective, dp.to[i], pc, ksq, curr_mobility));
            }
        }

        const Bitboard ps = (pos.pieces() ^ pos.pieces(KING)) & ~updated_ps;
        const Bitboard mobility_diff =
            (curr_mobility[WHITE] ^ prev_mobility[WHITE])
            | (curr_mobility[BLACK] ^ prev_mobility[BLACK]);

        Bitboard affected_pieces = ps & mobility_diff;
        while (affected_pieces)
        {
            Square s = pop_lsb(&affected_pieces);
            Piece pc = pos.piece_on(s);
            removed->push_back(make_index(perspective, s, pc, ksq, prev_mobility));
            added->push_back(make_index(perspective, s, pc, ksq, curr_mobility));
        }
    }

    template class HalfKPE4<Side::kFriend>;
    template class HalfKPE4<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
