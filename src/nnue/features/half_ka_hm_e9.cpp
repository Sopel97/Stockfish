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

//Definition of input features HalfKA_E3 of NNUE evaluation function

#include "half_ka_hm_e9.h"
#include "index_list.h"

namespace Stockfish::Eval::NNUE::Features {

    // We do flip for this arch.
    inline Square orient(Color perspective, Square s, Square ksq) {
        return Square(int(s) ^ (bool(perspective) * SQ_A8) ^ ((file_of(ksq) < FILE_E) * SQ_H1));
    }

    // Find the index of the feature quantity from the king position and PieceSquare
    template <Side AssociatedKing>
    inline IndexType HalfKAv2_hm_e9<AssociatedKing>::make_index(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq,
        Bitboard mobility[COLOR_NB][2]) {

        Square o_ksq = orient(perspective, ksq, ksq);

        int mobility_index_us = 0;
        int mobility_index_them = 0;
        const Bitboard sqbb = 1ull << s;
        for (int i = 0; i < 2; ++i)
        {
            mobility_index_us += bool(mobility[perspective][i] & sqbb);
            mobility_index_them += bool(mobility[~perspective][i] & sqbb);
        }

        IndexType idx = 0;
        // idx *= 32
        idx += KingBuckets[o_ksq];
        idx *= PS_END2;
        idx += kpp_board_index[perspective][pc];
        idx += orient(perspective, s, ksq);
        idx *= 3;
        idx += mobility_index_us;
        idx *= 3;
        idx += mobility_index_them;

        return idx;
    }

    template <Side AssociatedKing>
    inline std::pair<IndexType, IndexType> HalfKAv2_hm_e9<AssociatedKing>::make_index_2(
        Color perspective,
        Square s,
        Piece pc,
        Square ksq,
        Bitboard prev_mobility[COLOR_NB][2],
        Bitboard curr_mobility[COLOR_NB][2]) {

        Square o_ksq = orient(perspective, ksq, ksq);

        int prev_mobility_index_us = 0;
        int prev_mobility_index_them = 0;
        int curr_mobility_index_us = 0;
        int curr_mobility_index_them = 0;
        const Color us = perspective;
        const Color them = ~perspective;
        const Bitboard sqbb = 1ull << s;
        for (int i = 0; i < 2; ++i)
        {
            prev_mobility_index_us += bool(prev_mobility[us][i] & sqbb);
            prev_mobility_index_them += bool(prev_mobility[them][i] & sqbb);

            curr_mobility_index_us += bool(curr_mobility[us][i] & sqbb);
            curr_mobility_index_them += bool(curr_mobility[them][i] & sqbb);
        }

        IndexType idx_base = 0;
        // idx *= 32
        idx_base += KingBuckets[o_ksq];
        idx_base *= PS_END2;
        idx_base += kpp_board_index[perspective][pc];
        idx_base += orient(perspective, s, ksq);

        IndexType idx_prev = idx_base;
        idx_prev *= 3;
        idx_prev += prev_mobility_index_us;
        idx_prev *= 3;
        idx_prev += prev_mobility_index_them;

        IndexType idx_curr = idx_base;
        idx_curr *= 3;
        idx_curr += curr_mobility_index_us;
        idx_curr *= 3;
        idx_curr += curr_mobility_index_them;

        return std::make_pair(
            idx_prev,
            idx_curr
        );
    }

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void HalfKAv2_hm_e9<AssociatedKing>::AppendActiveIndices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        Square ksq = pos.square<KING>(perspective);

        Bitboard bb = pos.pieces();
        Bitboard (&curr_mobility)[COLOR_NB][2] = pos.state()->mobility;
        while (bb) {
            Square s = pop_lsb(&bb);
            active->push_back(make_index(perspective, s, pos.piece_on(s), ksq, curr_mobility));
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void HalfKAv2_hm_e9<AssociatedKing>::AppendChangedIndices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        Square ksq = pos.square<KING>(perspective);

        Bitboard (&curr_mobility)[COLOR_NB][2] = pos.state()->mobility;
        Bitboard (&prev_mobility)[COLOR_NB][2] = pos.state()->previous->mobility;

        const auto& dp = pos.state()->dirtyPiece;
        Bitboard updated = 0;
        for (int i = 0; i < dp.dirty_num; ++i) {
            Piece pc = dp.piece[i];

            if (dp.from[i] != SQ_NONE)
            {
                updated |= dp.from[i];
                removed->push_back(make_index(perspective, dp.from[i], pc, ksq, prev_mobility));
            }

            if (dp.to[i] != SQ_NONE)
            {
                updated |= dp.to[i];
                added->push_back(make_index(perspective, dp.to[i], pc, ksq, curr_mobility));
            }
        }

        Bitboard mobility_diff = 0;
        for (int i = 0; i < 2; ++i)
        {
            mobility_diff |= (curr_mobility[WHITE][i] ^ prev_mobility[WHITE][i]);
            mobility_diff |= (curr_mobility[BLACK][i] ^ prev_mobility[BLACK][i]);
        }

        Bitboard affected_pieces = pos.pieces() & ~updated & mobility_diff;
        while (affected_pieces)
        {
            Square s = pop_lsb(&affected_pieces);
            Piece pc = pos.piece_on(s);
            auto [r, a] = make_index_2(perspective, s, pc, ksq, prev_mobility, curr_mobility);
            if (r != a)
            {
                removed->push_back(r);
                added->push_back(a);
            }
        }
    }

    template class HalfKAv2_hm_e9<Side::kFriend>;

}  // namespace Eval::NNUE::Features
