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

//Definition of input features HalfKPE9 of NNUE evaluation function

#include "half_kpe9e.h"
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
    inline IndexType HalfKPE9E<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq,
        Bitboard* mobility,
        Bitboard* mobility2) {

        const IndexType mobility_index =
            (bool(mobility[perspective] & s) + bool(mobility2[perspective] & s)) * 3
            + bool(mobility[~perspective] & s) + bool(mobility2[~perspective] & s);

        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective] + PS_END3 * ksq)
            + (static_cast<IndexType>(SQUARE_NB) * static_cast<IndexType>(PS_END3)) * mobility_index;
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void HalfKPE9E<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard* mobility = pos.state()->mobility;
        Bitboard* mobility2 = pos.state()->mobility2;
        Bitboard bb =
            (mobility[WHITE] | mobility[BLACK])
            | pos.pieces();
        while (bb) {
            Square s = pop_lsb(&bb);
            Piece pc = pos.piece_on(s);
            active->push_back(make_index(perspective, s, pc, ksq, mobility, mobility2));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void HalfKPE9E<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        Square ksq = orient(
            perspective,
            pos.square<KING>(
                AssociatedKing == Side::kFriend ? perspective : ~perspective));

        Bitboard* curr_mobility = pos.state()->mobility;
        Bitboard* curr_mobility2 = pos.state()->mobility2;
        Bitboard* prev_mobility = pos.state()->previous->mobility;
        Bitboard* prev_mobility2 = pos.state()->previous->mobility2;

        const auto& dp = pos.state()->dirtyPiece;
        Bitboard updated_ps = 0;
        Bitboard changed_occupancy = 0;
        for (int i = 0; i < dp.dirty_num; ++i) {
            Piece pc = dp.piece[i];

            if (dp.from[i] != SQ_NONE)
            {
                updated_ps |= dp.from[i];
                changed_occupancy ^= dp.from[i];
                removed->push_back(make_index(perspective, dp.from[i], pc, ksq, prev_mobility, prev_mobility2));
            }

            if (dp.to[i] != SQ_NONE)
            {
                updated_ps |= dp.to[i];
                changed_occupancy ^= dp.to[i];
                added->push_back(make_index(perspective, dp.to[i], pc, ksq, curr_mobility, curr_mobility2));
            }
        }

        while (changed_occupancy)
        {
            Square s = pop_lsb(&changed_occupancy);
            Piece pc = pos.piece_on(s);
            if (pc == NO_PIECE)
            {
                if ((curr_mobility[WHITE] & s) || (curr_mobility[BLACK] & s))
                    added->push_back(make_index(perspective, s, NO_PIECE, ksq, curr_mobility, curr_mobility2));
            }
            else
            {
                if ((prev_mobility[WHITE] & s) || (prev_mobility[BLACK] & s))
                    removed->push_back(make_index(perspective, s, NO_PIECE, ksq, prev_mobility, prev_mobility2));
            }
        }

        const Bitboard ps = ~updated_ps;
        const Bitboard mobility_diff =
            (curr_mobility[WHITE] ^ prev_mobility[WHITE])
            | (curr_mobility[BLACK] ^ prev_mobility[BLACK]);
        const Bitboard mobility2_diff =
            (curr_mobility2[WHITE] ^ prev_mobility2[WHITE])
            | (curr_mobility2[BLACK] ^ prev_mobility2[BLACK]);

        Bitboard affected_pieces = ps & (mobility_diff | mobility2_diff);
        Bitboard prev_mobility_any = prev_mobility[WHITE] | prev_mobility[BLACK];
        Bitboard curr_mobility_any = curr_mobility[WHITE] | curr_mobility[BLACK];
        Bitboard occupied = pos.pieces();
        Bitboard affected_pieces_removed = affected_pieces & (prev_mobility_any | occupied);
        Bitboard affected_pieces_added = affected_pieces & (curr_mobility_any | occupied);
        while (affected_pieces_removed)
        {
            Square s = pop_lsb(&affected_pieces_removed);
            Piece pc = pos.piece_on(s);
            removed->push_back(make_index(perspective, s, pc, ksq, prev_mobility, prev_mobility2));
        }

        while (affected_pieces_added)
        {
            Square s = pop_lsb(&affected_pieces_added);
            Piece pc = pos.piece_on(s);
            added->push_back(make_index(perspective, s, pc, ksq, curr_mobility, curr_mobility2));
        }
    }

    template class HalfKPE9E<Side::kFriend>;
    template class HalfKPE9E<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
