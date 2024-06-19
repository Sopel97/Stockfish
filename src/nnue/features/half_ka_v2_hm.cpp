/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

//Definition of input features HalfKAv2_hm of NNUE evaluation function

#include "half_ka_v2_hm.h"

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../nnue_accumulator.h"

namespace Stockfish::Eval::NNUE::Features {

// Index of a feature for a given king position and another piece on some square
template<Color Perspective>
inline IndexType HalfKAv2_hm::make_index(Square s, Piece pc, Square ksq) {
    return IndexType((int(s) ^ OrientTBL[Perspective][ksq]) + PieceSquareIndex[Perspective][pc]
                     + KingBuckets[Perspective][ksq]);
}

// Get a list of indices for active features
template<Color Perspective>
void HalfKAv2_hm::append_active_indices(const Position& pos, IndexList& active) {
    Square   ksq = pos.square<KING>(Perspective);
    Bitboard bb  = pos.pieces();
    while (bb)
    {
        Square s = pop_lsb(bb);
        active.push_back(make_index<Perspective>(s, pos.piece_on(s), ksq));
    }
}

// Explicit template instantiations
template void HalfKAv2_hm::append_active_indices<WHITE>(const Position& pos, IndexList& active);
template void HalfKAv2_hm::append_active_indices<BLACK>(const Position& pos, IndexList& active);
template IndexType HalfKAv2_hm::make_index<WHITE>(Square s, Piece pc, Square ksq);
template IndexType HalfKAv2_hm::make_index<BLACK>(Square s, Piece pc, Square ksq);

// Get a list of indices for recently changed features
template<Color Perspective, int Start>
void HalfKAv2_hm::append_changed_indices(Square            ksq,
                                         const DirtyPiece& dp,
                                         IndexList&        removed,
                                         IndexList&        added) {
    for (int i = Start; i < dp.dirty_num; ++i)
    {
        if (dp.from[i] != SQ_NONE)
            removed.push_back(make_index<Perspective>(dp.from[i], dp.piece[i], ksq));
        if (dp.to[i] != SQ_NONE)
            added.push_back(make_index<Perspective>(dp.to[i], dp.piece[i], ksq));
    }
}

template<Color Perspective>
HalfKAv2_hm::MoveKeyType HalfKAv2_hm::make_move_key(Square ksq, const DirtyPiece& dp) {
    // At most 2 removed and 2 added
    static_assert(sizeof(MoveKeyType) * 8 >= 4 * DimensionsBits);

    MoveKeyType ir = 2;
    MoveKeyType ia = 0;

    MoveKeyType key = 0;

    for (int i = 0; i < dp.dirty_num; ++i)
    {
        if (dp.from[i] != SQ_NONE)
        {
            MoveKeyType idx = make_index<Perspective>(dp.from[i], dp.piece[i], ksq);
            assert(idx < Dimensions);
            assert(idx != 0);
            key |= idx << (ir++ * DimensionsBits);
        }
        if (dp.to[i] != SQ_NONE)
        {
            MoveKeyType idx = make_index<Perspective>(dp.to[i], dp.piece[i], ksq);
            assert(idx < Dimensions);
            assert(idx != 0);
            key |= idx << (ia++ * DimensionsBits);
        }
    }

    while (ir < 4)
        key |= MoveKeyType(Dimensions) << (ir++ * DimensionsBits);

    while (ia < 2)
        key |= MoveKeyType(Dimensions) << (ia++ * DimensionsBits);

    assert(ir == 4 && ia == 2);

    return key;
}


template<Color Perspective>
HalfKAv2_hm::QuietMoveKeyType HalfKAv2_hm::make_quiet_move_key(Square ksq, Square from, Square to, Piece pc) {
    static_assert(sizeof(QuietMoveKeyType) * 8 >= 2 * DimensionsBits);

    assert(from != SQ_NONE);
    assert(to != SQ_NONE);

    QuietMoveKeyType key = 0;

    key |= make_index<Perspective>(from, pc, ksq) << (1 * DimensionsBits);
    key |= make_index<Perspective>(to, pc, ksq) << (0 * DimensionsBits);

    return key;
}

void HalfKAv2_hm::decode_move_key(HalfKAv2_hm::MoveKeyType key,
                                  IndexList&               removed,
                                  IndexList&               added) {
    constexpr MoveKeyType DimensionsMask = (MoveKeyType(1) << DimensionsBits) - 1;

    IndexType r0 = (key >> (2 * DimensionsBits)) & DimensionsMask;
    IndexType r1 = (key >> (3 * DimensionsBits)) & DimensionsMask;
    IndexType a0 = (key >> (0 * DimensionsBits)) & DimensionsMask;
    IndexType a1 = (key >> (1 * DimensionsBits)) & DimensionsMask;

    if (r0 != Dimensions)
    {
        assert(r0 < Dimensions);
        assert(r0 != 0);
        removed.push_back(r0);
    }
    if (r1 != Dimensions)
    {
        assert(r1 < Dimensions);   
        assert(r1 != 0);     
        removed.push_back(r1);
    }
    if (a0 != Dimensions)
    {
        assert(a0 < Dimensions);
        assert(a0 != 0);     
        added.push_back(a0);
    }
    if (a1 != Dimensions)
    {
        assert(a1 < Dimensions);
        assert(a1 != 0);     
        added.push_back(a1);
    }
}

void HalfKAv2_hm::decode_quiet_move_key(HalfKAv2_hm::QuietMoveKeyType key,
                                  IndexList&               removed,
                                  IndexList&               added) {
    constexpr MoveKeyType DimensionsMask = (MoveKeyType(1) << DimensionsBits) - 1;

    IndexType r0 = (key >> (1 * DimensionsBits)) & DimensionsMask;
    IndexType a0 = (key >> (0 * DimensionsBits)) & DimensionsMask;

    assert(r0 < Dimensions);
    assert(r0 != 0);
    removed.push_back(r0);

    assert(a0 < Dimensions);
    assert(a0 != 0);     
    added.push_back(a0);
}

// Explicit template instantiations
template void HalfKAv2_hm::append_changed_indices<WHITE>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);
template void HalfKAv2_hm::append_changed_indices<BLACK>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);
template void HalfKAv2_hm::append_changed_indices<WHITE, 1>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);
template void HalfKAv2_hm::append_changed_indices<BLACK, 1>(Square            ksq,
                                                         const DirtyPiece& dp,
                                                         IndexList&        removed,
                                                         IndexList&        added);


// Explicit template instantiations
template HalfKAv2_hm::MoveKeyType HalfKAv2_hm::make_move_key<WHITE>(Square            ksq,
                                                                    const DirtyPiece& dp);
template HalfKAv2_hm::MoveKeyType HalfKAv2_hm::make_move_key<BLACK>(Square            ksq,
                                                                    const DirtyPiece& dp);
                                                                    
template HalfKAv2_hm::QuietMoveKeyType HalfKAv2_hm::make_quiet_move_key<WHITE>(Square ksq, Square from, Square to, Piece pc);
template HalfKAv2_hm::QuietMoveKeyType HalfKAv2_hm::make_quiet_move_key<BLACK>(Square ksq, Square from, Square to, Piece pc);

int HalfKAv2_hm::update_cost(const StateInfo* st) { return st->dirtyPiece.dirty_num; }

int HalfKAv2_hm::refresh_cost(const Position& pos) { return pos.count<ALL_PIECES>(); }

bool HalfKAv2_hm::requires_refresh(const StateInfo* st, Color perspective) {
    return st->dirtyPiece.piece[0] == make_piece(perspective, KING);
}

}  // namespace Stockfish::Eval::NNUE::Features
