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

// Code for calculating NNUE evaluation function

#ifndef NNUE_IMBALANCE_NET_H_INCLUDED
#define NNUE_IMBALANCE_NET_H_INCLUDED

#include <cstdint>

#include "nnue_common.h"

#include "../position.h"

namespace Stockfish::Eval::NNUE {
  class ImbalanceNet {
  public:
    using WeightType = std::int32_t;

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      for (std::size_t i = 0; i < 15; ++i)
        weights[i] = read_little_endian<WeightType>(stream);
      return !stream.fail();
    }

    std::int32_t evaluate(const Position& pos) {
      const Color us = pos.side_to_move();
      const Color them = ~us;

      const int our_pawns = popcount(pos.pieces(us, PAWN));
      const int their_pawns = popcount(pos.pieces(them, PAWN));

      const int our_knights = popcount(pos.pieces(us, KNIGHT));
      const int their_knights = popcount(pos.pieces(them, KNIGHT));

      const Bitboard our_bishops = pos.pieces(us, BISHOP);
      const Bitboard their_bishops = pos.pieces(them, BISHOP);

      const int our_ls_bishops = popcount(our_bishops & ~DarkSquares);
      const int their_ls_bishops = popcount(their_bishops & ~DarkSquares);

      const int our_ds_bishops = popcount(our_bishops & DarkSquares);
      const int their_ds_bishops = popcount(their_bishops & DarkSquares);

      const int our_rooks = popcount(pos.pieces(us, ROOK));
      const int their_rooks = popcount(pos.pieces(them, ROOK));

      const int our_queens = popcount(pos.pieces(us, QUEEN));
      const int their_queens = popcount(pos.pieces(them, QUEEN));

      std::int32_t output =
            (our_knights - their_knights) * (our_pawns - their_pawns) * weights[0]

          + (our_ls_bishops - their_ls_bishops) * (our_pawns - their_pawns) * weights[1]
          + (our_ls_bishops - their_ls_bishops) * (our_knights - their_knights) * weights[2]

          + (our_ds_bishops - their_ds_bishops) * (our_pawns - their_pawns) * weights[3]
          + (our_ds_bishops - their_ds_bishops) * (our_knights - their_knights) * weights[4]
          + (our_ds_bishops - their_ds_bishops) * (our_ls_bishops - their_ls_bishops) * weights[5]

          + (our_rooks - their_rooks) * (our_pawns - their_pawns) * weights[6]
          + (our_rooks - their_rooks) * (our_knights - their_knights) * weights[7]
          + (our_rooks - their_rooks) * (our_ls_bishops - their_ls_bishops) * weights[8]
          + (our_rooks - their_rooks) * (our_ds_bishops - their_ds_bishops) * weights[9]

          + (our_queens - their_queens) * (our_pawns - their_pawns) * weights[10]
          + (our_queens - their_queens) * (our_knights - their_knights) * weights[11]
          + (our_queens - their_queens) * (our_ls_bishops - their_ls_bishops) * weights[12]
          + (our_queens - their_queens) * (our_ds_bishops - their_ds_bishops) * weights[13]
          + (our_queens - their_queens) * (our_rooks - their_rooks) * weights[14];

      return output;
    }
  private:
    WeightType weights[15];
  };
}

#endif