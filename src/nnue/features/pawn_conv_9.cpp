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

#include <array>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <set>
#include <utility>

#include "pawn_conv_9.h"
#include "index_list.h"

static uint64_t calc_as_ternary(uint64_t i, int num_bits)
{
    uint64_t tern = 0;
    for (int j = 1; j <= num_bits; ++j)
    {
        tern *= 3;
        tern += (i >> (num_bits - j)) & 1;
    }
    return tern;
}

static std::array<uint32_t, 512> as_ternary = []() {
    std::array<uint32_t, 512> s_as_ternary{};
    for (int i = 0; i < 512; ++i)
    {
        s_as_ternary[i] = calc_as_ternary(i, 9);
    }
    return s_as_ternary;
}();

namespace Eval::NNUE::Features {

    template <Side AssociatedKing>
    inline uint64_t pext_conv(uint64_t bb, int s)
    {
        return ((bb & PawnConv9<AssociatedKing>::conv_masks[s]) * PawnConv9<AssociatedKing>::conv_magics[s]) >> 55ull;
    }

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
    inline IndexType PawnConv9<AssociatedKing>::make_index(
        Color perspective,
        Bitboard our_pawns,
        Bitboard their_pawns,
        Square s) {

        return
            as_ternary[pext_conv<AssociatedKing>(our_pawns, s)]
            + 2 * as_ternary[pext_conv<AssociatedKing>(their_pawns, s)]
            + (kConvStates * conv_square_idx[orient(perspective, s)]);
    }

    template <Side AssociatedKing>
    const std::vector<typename PawnConv9<AssociatedKing>::Factors> PawnConv9<AssociatedKing>::factors_lookup = [](){
        std::vector<Factors> s_factors;
        s_factors.resize(kDimensions);

        Color perspective = WHITE;

        for (Square sq : conv_squares)
        {
            const auto square_index = conv_square_idx[orient(perspective, sq)];
            std::cout << "Computing conv factors lookup for square " << (int)sq << " with index " << square_index << '\n';
            const Bitboard mask = conv_masks[sq];

            constexpr int conv_offsets[9] = {-9, -8, -7, -1, 0, 1, 7, 8, 9};

            std::set<IndexType> hashed_our;
            std::set<IndexType> hashed_their;
            std::set<std::pair<IndexType, IndexType>> hashed_our_their;
            std::set<IndexType> hashed_conv_ids;
            bool duplicate_feature_ids = false;
            for (IndexType conv_idx = 0; conv_idx < kConvStates; ++conv_idx)
            {
                auto conv_idx_cpy = conv_idx;
                std::array<IndexType, 9> states;
                for (int i = 0; i < 9; ++i)
                {
                    states[i] = conv_idx_cpy % 3;
                    conv_idx_cpy /= 3;
                }

                Bitboard our_pawns = 0;
                Bitboard their_pawns = 0;
                auto add_state = [&our_pawns, &their_pawns, sq](IndexType state, int sq_off) {
                    if (state == 1) our_pawns |= 1ull << (sq + sq_off);
                    else if (state == 2) their_pawns |= 1ull << (sq + sq_off);
                };
                for (int i = 0; i < 9; ++i)
                    add_state(states[i], conv_offsets[i]);

                if ((our_pawns & mask) != our_pawns)
                {
                    std::cerr << "our_pawns outside of mask. Exiting.\n";
                    std::exit(1);
                }

                if ((their_pawns & mask) != their_pawns)
                {
                    std::cerr << "their_pawns outside of mask. Exiting.\n";
                    std::exit(1);
                }

                if (our_pawns & their_pawns)
                {
                    std::cerr << "our_pawns & their_pawns non-zero. Exiting.\n";
                    std::exit(1);
                }

                const IndexType hashed_conv_idx =
                    // We can't use as_ternary because it might not be initialized
                    calc_as_ternary(pext_conv<AssociatedKing>(our_pawns, sq), 9)
                    + 2 * calc_as_ternary(pext_conv<AssociatedKing>(their_pawns, sq), 9);

                hashed_our.emplace(pext_conv<AssociatedKing>(our_pawns, sq));
                hashed_their.emplace(pext_conv<AssociatedKing>(their_pawns, sq));
                hashed_our_their.emplace(
                    pext_conv<AssociatedKing>(our_pawns, sq),
                    pext_conv<AssociatedKing>(their_pawns, sq)
                    );
                hashed_conv_ids.emplace(hashed_conv_idx);

                if (hashed_conv_idx >= kConvStates)
                {
                    std::cerr << "too high conv idx: " << hashed_conv_idx << '\n';
                    std::cerr << "our_pawns: " << our_pawns << '\n';
                    std::cerr << "their_pawns: " << their_pawns << '\n';
                    std::cerr << "pext_conv(our_pawns): " << pext_conv<AssociatedKing>(our_pawns, sq) << '\n';
                    std::cerr << "pext_conv(their_pawns): " << pext_conv<AssociatedKing>(their_pawns, sq) << '\n';
                    std::cerr << "tern our_pawns: " << calc_as_ternary(pext_conv<AssociatedKing>(our_pawns, sq), 9) << '\n';
                    std::cerr << "tern their_pawns: " << calc_as_ternary(pext_conv<AssociatedKing>(their_pawns, sq), 9) << '\n';
                    std::exit(1);
                }

                const IndexType idx =
                    hashed_conv_idx
                    + (kConvStates * square_index);

                auto& factors = s_factors[idx];
                duplicate_feature_ids |= factors.conv_id != IndexType(-1);
                factors.conv_id = conv_idx;
                auto add_factor = [&factors, sq, square_index](IndexType state, int sq_off) {
                    if (state == 1) factors.piece_ids[factors.num_piece_ids++] = (SQUARE_NB * 2 * square_index) + sq + sq_off;
                    else if (state == 2) factors.piece_ids[factors.num_piece_ids++] = (SQUARE_NB * 2 * square_index) + SQUARE_NB + sq + sq_off;
                };
                for (int i = 0; i < 9; ++i)
                    add_factor(states[i], conv_offsets[i]);
            }

            if (hashed_our_their.size() != kConvStates)
            {
                std::cerr << "kConvStates: " << kConvStates << '\n';
                std::cerr << "size()     : " << hashed_our_their.size() << '\n';
                std::cerr << "Some duplicate hashed_our_their idx... Exiting.\n";
                std::exit(1);
            }

            if (hashed_our.size() != 512)
            {
                std::cerr << "512   : " << 512 << '\n';
                std::cerr << "size(): " << hashed_our.size() << '\n';
                std::cerr << "Some duplicate hashed_our idx... Exiting.\n";
                std::exit(1);
            }

            if (hashed_their.size() != 512)
            {
                std::cerr << "512   : " << 512 << '\n';
                std::cerr << "size(): " << hashed_their.size() << '\n';
                std::cerr << "Some duplicate hashed_their idx... Exiting.\n";
                std::exit(1);
            }

            if (hashed_conv_ids.size() != kConvStates)
            {
                std::cerr << "kConvStates: " << kConvStates << '\n';
                std::cerr << "size():      " << hashed_conv_ids.size() << '\n';
                std::cerr << "Some duplicate hashed conv idx... Exiting.\n";
                std::exit(1);
            }

            if (duplicate_feature_ids)
            {
                std::cerr << "Some duplicate feature ids... Exiting.\n";
                std::exit(1);
            }
        }

        return s_factors;
    }();

    // Get a list of indices for active features
    template <Side AssociatedKing>
    void PawnConv9<AssociatedKing>::append_active_indices(
        const Position& pos,
        Color perspective,
        IndexList* active) {

        auto c = AssociatedKing == Side::kFriend ? perspective : ~perspective;

        auto our_pawns = pos.pieces(c, PAWN);
        auto their_pawns = pos.pieces(~c, PAWN);
        auto all_pawns = our_pawns | their_pawns;

        for (Square sq : conv_squares)
        {
            // Only include convolutions when there's at least 2 pawns
            if (more_than_one(all_pawns & conv_masks[sq]))
            {
                active->push_back(make_index(perspective, our_pawns, their_pawns, sq));
            }
        }
    }

    // Get a list of indices for recently changed features
    template <Side AssociatedKing>
    void PawnConv9<AssociatedKing>::append_changed_indices(
        const Position& pos,
        Color perspective,
        IndexList* removed,
        IndexList* added) {

        return;
    }

    template class PawnConv9<Side::kFriend>;
    template class PawnConv9<Side::kEnemy>;

}  // namespace Eval::NNUE::Features
