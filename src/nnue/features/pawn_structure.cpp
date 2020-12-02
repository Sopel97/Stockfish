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

#include "pawn_structure.h"
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
    inline IndexType PawnStructure<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq) {

        return IndexType(orient(perspective, s) + kpp_board_index[pc][perspective] + PS_END2 * ksq);
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void PawnStructure<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        auto c = AssociatedKing == Side::kFriend ? perspective : ~perspective;

        auto our_pawns = pos.pieces(c, PAWN);
        auto their_pawns = pos.pieces(~c, PAWN);
        auto all_pawns = our_pawns | their_pawns;

        if (popcount(our_pawns) == 0)
        {
            active->push_back(0);
            return;
        }

        while(our_pawns)
        {
            Square ksq = pop_lsb(&our_pawns);

            Bitboard influence = PawnInfluence[c][ksq];

            Bitboard bb = all_pawns & influence;
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
    void PawnStructure<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        return;
    }

    template class PawnStructure<Side::kFriend>;
    template class PawnStructure<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
