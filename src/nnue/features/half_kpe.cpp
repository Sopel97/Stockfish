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

//Definition of input features HalfKP of NNUE evaluation function

#include "half_kpe.h"
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
    inline IndexType HalfKPE<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq) {

        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective] + PS_END3 * ksq);
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void HalfKPE<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard bb = AllSquares;
        while (bb) {
            Square s = pop_lsb(&bb);
            active->push_back(make_index(perspective, s, pos.piece_on(s), ksq));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void HalfKPE<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        const auto& dp = pos.state()->dirtyPiece;
        Bitboard changed_occupancy = 0;
        for (int i = 0; i < dp.dirty_num; ++i) {
            Piece pc = dp.piece[i];

            if (dp.from[i] != SQ_NONE)
            {
                changed_occupancy ^= dp.from[i];
                removed->push_back(make_index(perspective, dp.from[i], pc, ksq));
            }

            if (dp.to[i] != SQ_NONE)
            {
                changed_occupancy ^= dp.to[i];
                added->push_back(make_index(perspective, dp.to[i], pc, ksq));
            }
        }

        while (changed_occupancy)
        {
            Square s = pop_lsb(&changed_occupancy);
            Piece pc = pos.piece_on(s);
            if (pc == NO_PIECE)
                added->push_back(make_index(perspective, s, NO_PIECE, ksq));
            else
                removed->push_back(make_index(perspective, s, NO_PIECE, ksq));
        }
    }

    template class HalfKPE<Side::kFriend>;
    template class HalfKPE<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
