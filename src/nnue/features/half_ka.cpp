/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2021 The Stockfish developers (see AUTHORS file)

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

#include "half_ka.h"
#include "index_list.h"

namespace Stockfish::Eval::NNUE::Features {

  // Orient a square according to perspective (rotates by 180 for black)
  inline Square orient(Color perspective, Square s) {
    return Square(int(s) ^ (bool(perspective) * SQ_A8));
  }

  // Index of a feature for a given king position and another piece on some square
  inline IndexType make_index(Color perspective, Square s, Piece pc, Square ksq, int cr) {
    int king_bucket = static_cast<int>(ksq);
    if (cr != 0)
      king_bucket = 63 + cr;
    return IndexType(orient(perspective, s) + kpp_board_index[perspective][pc] + PS_END2 * king_bucket);
  }

  static inline int get_cr(const Position& pos, Square oriented_ksq, Color perspective) {
    int cr = 0;
    auto rights = pos.castling_rights(perspective);
    if (oriented_ksq == SQ_E1)
    {
      if (rights & KING_SIDE) cr += 1;
      if (rights & QUEEN_SIDE) cr += 2;
    }
    return cr;
  }

  // Get a list of indices for active features
  template <Side AssociatedKing>
  void HalfKA<AssociatedKing>::AppendActiveIndices(
      const Position& pos, Color perspective, IndexList* active) {

    Square ksq = orient(perspective, pos.square<KING>(perspective));
    int cr = get_cr(pos, ksq, perspective);
    Bitboard bb = pos.pieces();
    while (bb) {
      Square s = pop_lsb(&bb);
      active->push_back(make_index(perspective, s, pos.piece_on(s), ksq, cr));
    }
  }

  // Get a list of indices for recently changed features
  template <Side AssociatedKing>
  void HalfKA<AssociatedKing>::AppendChangedIndices(
      const Position& pos, const DirtyPiece& dp, Color perspective,
      IndexList* removed, IndexList* added) {

    Square ksq = orient(perspective, pos.square<KING>(perspective));
    int cr = get_cr(pos, ksq, perspective);
    for (int i = 0; i < dp.dirty_num; ++i) {
      Piece pc = dp.piece[i];
      if (dp.from[i] != SQ_NONE)
        removed->push_back(make_index(perspective, dp.from[i], pc, ksq, cr));
      if (dp.to[i] != SQ_NONE)
        added->push_back(make_index(perspective, dp.to[i], pc, ksq, cr));
    }
  }

  template class HalfKA<Side::kFriend>;

}  // namespace Stockfish::Eval::NNUE::Features
