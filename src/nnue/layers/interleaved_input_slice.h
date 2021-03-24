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

#ifndef NNUE_LAYERS_INTERLEAVED_INPUT_SLICE_H_INCLUDED
#define NNUE_LAYERS_INTERLEAVED_INPUT_SLICE_H_INCLUDED

#include "../nnue_common.h"

#include <string>
#include <cstdint>
#include <type_traits>

namespace Stockfish::Eval::NNUE::Layers {

    // Input layer
    template <IndexType HalfOutputDimensions, IndexType Stride, IndexType Offset = 0>
    class InterleavedInputSlice {
    public:
        // Need to maintain alignment
        static_assert(Offset % kMaxSimdWidth == 0, "");
        static_assert(Stride % kMaxSimdWidth == 0, "");

        static_assert(Offset == 0, "For the current implementation");

        // Output type
        using OutputType = TransformedFeatureType;

        // Output dimensionality
        static constexpr IndexType kHalfOutputDimensions = HalfOutputDimensions;
        static constexpr IndexType kOutputDimensions = HalfOutputDimensions * 2;

        // Size of forward propagation buffer used from the input layer to this layer
        static constexpr std::size_t kBufferSize = kOutputDimensions * sizeof(OutputType);

        static constexpr int kLayerIndex = 1;

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t GetHashValue() {
            std::uint32_t hash_value = 0xDF31E90Du;
            hash_value ^= kOutputDimensions ^ (Offset << 10) ^ (Stride << 4);
            return hash_value;
        }

        // Read network parameters
        bool ReadParameters(std::istream& /*stream*/) {
            return true;
        }

        // Forward propagation
        const OutputType* Propagate(
            const TransformedFeatureType* transformed_features,
            char* buffer) const {

#if !defined(USE_SSE2)
            static_assert(false);
#endif

            static_assert(std::is_same_v<OutputType, TransformedFeatureType>);
            static_assert(sizeof(TransformedFeatureType) == 1);
            static_assert(kHalfOutputDimensions % 16 == 0);

            auto* output = reinterpret_cast<TransformedFeatureType*>(buffer);

            for (IndexType i = 0; i < kHalfOutputDimensions; i += 16)
            {
                __m128i p0 = _mm_load_si128(reinterpret_cast<const __m128i*>(transformed_features + i));
                __m128i p1 = _mm_load_si128(reinterpret_cast<const __m128i*>(transformed_features + Stride + i));

                __m128i low = _mm_unpacklo_epi8(p0, p1);
                __m128i high = _mm_unpackhi_epi8(p0, p1);

                _mm_store_si128(reinterpret_cast<__m128i*>(output + i * 2), low);
                _mm_store_si128(reinterpret_cast<__m128i*>(output + i * 2 + 16), high);
            }

            return reinterpret_cast<OutputType*>(buffer);
        }

    private:
    };

}  // namespace Layers

#endif // #ifndef NNUE_LAYERS_INTERLEAVED_INPUT_SLICE_H_INCLUDED
