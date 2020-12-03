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

#include "constrained_half_na.h"
#include "index_list.h"

namespace Eval::NNUE::Features {

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
    inline IndexType ConstrainedHalfNA<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq) {

        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective] + PS_END2 * ksq);
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void ConstrainedHalfNA<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        auto c = AssociatedKing == Side::kFriend ? perspective : ~perspective;

        auto queens = pos.pieces(c, KNIGHT);

        if (popcount(queens) == 0)
        {
            active->push_back(0);
            return;
        }

        for (int j = 0; j < 2; ++j)
        {
            Square ksq = pop_lsb(&queens);

            Bitboard influence = PseudoAttacks[KNIGHT][ksq];

            // We exclude our king as this feature should be covered by halfka
            Bitboard bb =
                (pos.pieces() & ~pos.pieces(KING) & influence)
                | (pos.pieces(WHITE, PAWN) & PiecePawnInfluence[WHITE][ksq])
                | (pos.pieces(BLACK, PAWN) & PiecePawnInfluence[BLACK][ksq]);
            Square oriented_ksq = orient(
                perspective,
                ksq);

            while (bb) {
                Square s = pop_lsb(&bb);
                active->push_back(make_index(perspective, s, pos.piece_on(s), oriented_ksq));
            }
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void ConstrainedHalfNA<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        auto c = AssociatedKing == Side::kFriend ? perspective : ~perspective;

        auto queens = pos.pieces(c, KNIGHT);

        if (popcount(queens) == 0)
            return;

        for (int j = 0; j < 2; ++j)
        {
            Square ksq = pop_lsb(&queens);

            Bitboard influence = PseudoAttacks[KNIGHT][ksq];
            Bitboard white_pawn_influence = PiecePawnInfluence[WHITE][ksq];
            Bitboard black_pawn_influence = PiecePawnInfluence[BLACK][ksq];

            const auto& dp = pos.state()->dirtyPiece;
            for (int i = 0; i < dp.dirty_num; ++i) {
                Piece pc = dp.piece[i];

                if (type_of(pc) == KING)
                    continue;

                Bitboard bb = influence;
                if (pc == make_piece(WHITE, PAWN))
                    bb = white_pawn_influence;
                else if (pc == make_piece(BLACK, PAWN))
                    bb = black_pawn_influence;

                if (dp.from[i] != SQ_NONE && (bb & dp.from[i]))
                    removed->push_back(make_index(perspective, dp.from[i], pc, ksq));

                if (dp.to[i] != SQ_NONE && (bb & dp.to[i]))
                    added->push_back(make_index(perspective, dp.to[i], pc, ksq));
            }
        }
    }

    template class ConstrainedHalfNA<Side::kFriend>;
    template class ConstrainedHalfNA<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
