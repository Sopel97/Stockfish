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

//Definition of input features HalfKA of NNUE evaluation function

#include "half_kae5.h"
#include "index_list.h"

namespace Eval::NNUE::Features {

    // We do flip for this arch.
    inline Square orient(Color perspective, Square s) {
        return Square(int(s) ^ (bool(perspective) * SQ_A8));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    template <Side AssociatedKing>
    inline IndexType HalfKAE5<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq,
        Bitboard mobility[COLOR_NB][3]) {

        int mobility_index = 2;
        for (int i = 0; i < 3; ++i)
        {
            mobility_index += bool(mobility[perspective][i] & s);
            mobility_index -= bool(mobility[~perspective][i] & s);
        }
        mobility_index = std::clamp(mobility_index, 0, 4);

        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective] + PS_END2 * ksq)
            + (static_cast<IndexType>(SQUARE_NB) * static_cast<IndexType>(PS_END2) * static_cast<IndexType>(mobility_index));
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void HalfKAE5<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard bb = pos.pieces();
        while (bb) {
            Square s = pop_lsb(&bb);
            active->push_back(make_index(perspective, s, pos.piece_on(s), ksq, pos.state()->mobility));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void HalfKAE5<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard (&curr_mobility)[COLOR_NB][3] = pos.state()->mobility;
        Bitboard (&prev_mobility)[COLOR_NB][3] = pos.state()->previous->mobility;

        const auto& dp = pos.state()->dirtyPiece;
        Bitboard updated_ps = 0;
        for (int i = 0; i < dp.dirty_num; ++i) {
            Piece pc = dp.piece[i];

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

        const Bitboard ps = pos.pieces() & ~updated_ps;
        Bitboard mobility_diff = 0;
        for (int i = 0; i < 3; ++i)
        {
            mobility_diff |= (curr_mobility[WHITE][i] ^ prev_mobility[WHITE][i]);
            mobility_diff |= (curr_mobility[BLACK][i] ^ prev_mobility[BLACK][i]);
        }

        Bitboard affected_pieces = ps & mobility_diff;
        while (affected_pieces)
        {
            Square s = pop_lsb(&affected_pieces);
            Piece pc = pos.piece_on(s);
            removed->push_back(make_index(perspective, s, pc, ksq, prev_mobility));
            added->push_back(make_index(perspective, s, pc, ksq, curr_mobility));
        }
    }

    template class HalfKAE5<Side::kFriend>;
    template class HalfKAE5<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
