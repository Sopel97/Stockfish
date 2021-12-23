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

// Definition of layer ClippedReLU of NNUE evaluation function

#ifndef NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
#define NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED

#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

  // Clipped ReLU
  template <IndexType InDims>
  class ClippedReLU {
   public:
    // Input/output type
    using InputType = std::int32_t;
    using OutputType = std::uint8_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = InputDimensions;
    static constexpr IndexType PaddedOutputDimensions =
        ceil_to_multiple<IndexType>(OutputDimensions, 32);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevhash) {
      std::uint32_t hash_value = 0x538D24C7u;
      hash_value += prevhash;
      return hash_value;
    }

    // Read network parameters
    bool read_parameters(std::istream&) {
      return true;
    }

    // Write network parameters
    bool write_parameters(std::ostream&) const {
      return true;
    }

    // Forward propagation
    const OutputType* propagate(
        const InputType* input, OutputType* output) const {

      constexpr unsigned additional_precision_bits = 4;

  #if defined(USE_AVX2)
      if constexpr (InputDimensions % 32 == 0) {
        constexpr int OutputChunkSize = 256 / 8;
        constexpr int NumOutputChunks = InputDimensions / OutputChunkSize;

        const __m256i cst_127_epi16 = _mm256_set1_epi16(127 << additional_precision_bits);
        const __m256i cst_126_epi8 = _mm256_set1_epi8(126);
        const __m256i Offsets = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

        const __m256i* in  = reinterpret_cast<const __m256i*>(input );
              __m256i* out = reinterpret_cast<      __m256i*>(output);

        for (int i = 0; i < NumOutputChunks; ++i)
        {
          __m256i v0a = in[i*4 + 0];
          __m256i v0b = in[i*4 + 1];
          __m256i v1a = in[i*4 + 2];
          __m256i v1b = in[i*4 + 3];

          __m256i v0 = _mm256_srai_epi16(_mm256_packs_epi32(v0a, v0b), WeightScaleBits - additional_precision_bits);
          __m256i v1 = _mm256_srai_epi16(_mm256_packs_epi32(v1a, v1b), WeightScaleBits - additional_precision_bits);

          __m256i sign = _mm256_packs_epi16(v0, v1);

          // std::int16_t v = min(abs(in[i]), 127) - 127;
          v0 = _mm256_subs_epu16(cst_127_epi16, _mm256_abs_epi16(v0));
          v1 = _mm256_subs_epu16(cst_127_epi16, _mm256_abs_epi16(v1));

          if constexpr (additional_precision_bits != 4) {
            static_assert(additional_precision_bits <= 4);
            v0 = _mm256_slli_epi16(v0, 4 - additional_precision_bits);
            v1 = _mm256_slli_epi16(v1, 4 - additional_precision_bits);
          }

          v0 = _mm256_mulhi_epi16(v0, v0);
          v1 = _mm256_mulhi_epi16(v1, v1);

          v0 = _mm256_packs_epi16(v0, v1);

          v0 = _mm256_blendv_epi8(_mm256_subs_epi8(cst_126_epi8, v0), v0, sign);

          out[i] = _mm256_permutevar8x32_epi32(v0, Offsets);
        }
      } else if constexpr (InputDimensions % 16 == 0) {
        constexpr int OutputChunkSize = 128 / 8;
        constexpr int NumOutputChunks = InputDimensions / OutputChunkSize;

        const __m128i cst_127_epi16 = _mm_set1_epi16(127 << additional_precision_bits);
        const __m128i cst_126_epi8 = _mm_set1_epi8(126);

        const __m128i* in  = reinterpret_cast<const __m128i*>(input );
              __m128i* out = reinterpret_cast<      __m128i*>(output);

        for (int i = 0; i < NumOutputChunks; ++i)
        {
          __m128i v0a = in[i*4 + 0];
          __m128i v0b = in[i*4 + 1];
          __m128i v1a = in[i*4 + 2];
          __m128i v1b = in[i*4 + 3];

          __m128i v0 = _mm_srai_epi16(_mm_packs_epi32(v0a, v0b), WeightScaleBits - additional_precision_bits);
          __m128i v1 = _mm_srai_epi16(_mm_packs_epi32(v1a, v1b), WeightScaleBits - additional_precision_bits);

          __m128i sign = _mm_packs_epi16(v0, v1);

          // std::int16_t v = min(abs(in[i]), 127) - 127;
          v0 = _mm_subs_epu16(cst_127_epi16, _mm_abs_epi16(v0));
          v1 = _mm_subs_epu16(cst_127_epi16, _mm_abs_epi16(v1));

          if constexpr (additional_precision_bits != 4) {
            static_assert(additional_precision_bits <= 4);
            v0 = _mm_slli_epi16(v0, 4 - additional_precision_bits);
            v1 = _mm_slli_epi16(v1, 4 - additional_precision_bits);
          }

          v0 = _mm_mulhi_epi16(v0, v0);
          v1 = _mm_mulhi_epi16(v1, v1);

          v0 = _mm_packs_epi16(v0, v1);

          v0 = _mm_blendv_epi8(_mm_subs_epi8(cst_126_epi8, v0), v0, sign);

          out[i] = v0;
        }
      }

      constexpr IndexType Start = InputDimensions / 16 * 16;
  #else
      constexpr IndexType Start = 0;
  #endif

      for (IndexType i = Start; i < InputDimensions; ++i) {
        std::int16_t v = std::min(std::abs(input[i] >> (WeightScaleBits - additional_precision_bits)), 127 << additional_precision_bits) - (127 << additional_precision_bits);
        std::int16_t vv = v * v >> (8 + additional_precision_bits * 2);
        if (input[i] > 0)
            vv = 126 - vv;
        output[i] = static_cast<OutputType>(vv);
      }

      // Affine transform layers expect that there is at least
      // ceil_to_multiple(OutputDimensions, 32) initialized values.
      // We cannot do this in the affine transform because it requires
      // preallocating space here.
      for (IndexType i = OutputDimensions; i < PaddedOutputDimensions; ++i) {
        output[i] = 0;
      }

      return output;
    }
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
