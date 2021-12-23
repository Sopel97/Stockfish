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

// Definition of layer AffineTransform of NNUE evaluation function

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED

#include <iostream>
#include "../nnue_common.h"
#include "../../simd.h"

namespace Stockfish::Eval::NNUE::Layers {

/// popcount() counts the number of non-zero bits in a bitboard

inline int popcount(Bitboard b) {

#ifndef USE_POPCNT

  union { Bitboard bb; uint16_t u[4]; } v = { b };
  return PopCnt16[v.u[0]] + PopCnt16[v.u[1]] + PopCnt16[v.u[2]] + PopCnt16[v.u[3]];

#elif defined(_MSC_VER) || defined(__INTEL_COMPILER)

  return (int)_mm_popcnt_u64(b);

#else // Assumed gcc or compatible compiler

  return __builtin_popcountll(b);

#endif
}


/// lsb() and msb() return the least/most significant bit in a non-zero bitboard

#if defined(__GNUC__)  // GCC, Clang, ICC

static inline IndexType lsb_(std::uint64_t b) {
  assert(b);
  return IndexType(__builtin_ctzll(b));
}

static inline IndexType msb_(std::uint64_t b) {
  assert(b);
  return IndexType(63 ^ __builtin_clzll(b));
}

#elif defined(_MSC_VER)  // MSVC

# ifdef _WIN64  // MSVC, WIN64

static inline IndexType lsb_(std::uint64_t b) {
  assert(b);
  unsigned long idx;
  _BitScanForward64(&idx, b);
  return (IndexType) idx;
}

static inline IndexType msb_(std::uint64_t b) {
  assert(b);
  unsigned long idx;
  _BitScanReverse64(&idx, b);
  return (IndexType) idx;
}

# else  // MSVC, WIN32

static inline IndexType lsb_(std::uint64_t b) {
  assert(b);
  unsigned long idx;

  if (b & 0xffffffff) {
      _BitScanForward(&idx, int32_t(b));
      return IndexType(idx);
  } else {
      _BitScanForward(&idx, int32_t(b >> 32));
      return IndexType(idx + 32);
  }
}

static inline IndexType msb_(std::uint64_t b) {
  assert(b);
  unsigned long idx;

  if (b >> 32) {
      _BitScanReverse(&idx, int32_t(b >> 32));
      return IndexType(idx + 32);
  } else {
      _BitScanReverse(&idx, int32_t(b));
      return IndexType(idx);
  }
}

# endif

#else  // Compiler is neither GCC nor MSVC compatible

#error "Compiler not supported."

#endif

  // Affine transformation layer
  template <IndexType InDims, IndexType OutDims>
  class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType = std::uint8_t;
    using OutputType = std::int32_t;
    static_assert(std::is_same<InputType, std::uint8_t>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

#if defined (USE_AVX512)
    static constexpr const IndexType NnzInputSimdWidth = 64;
    static constexpr const IndexType InputSimdWidth = 32;
    static constexpr const IndexType OutputSimdWidth = 8;
#elif defined (USE_AVX2)
    static constexpr const IndexType NnzInputSimdWidth = 32;
    static constexpr const IndexType InputSimdWidth = 32;
    static constexpr const IndexType OutputSimdWidth = 8;
#elif defined (USE_SSE2)
    static constexpr const IndexType NnzInputSimdWidth = 16;
    static constexpr const IndexType InputSimdWidth = 16;
    static constexpr const IndexType OutputSimdWidth = 4;
#endif

    static constexpr const IndexType MaxInputSimdWidth = 64;
    static constexpr const IndexType MaxOutputSimdWidth = 64;
    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static_assert(InputDimensions % MaxInputSimdWidth == 0, "We don't account for the padding left in the serialized network data correctly yet.");

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevhash) {
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= prevhash >> 1;
      hashValue ^= prevhash << 31;
      return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      for (std::size_t i = 0; i < OutputDimensions; ++i)
        biases[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < PaddedOutputDimensions; ++i)
        for (std::size_t j = 0; j < InputDimensions; ++j)
          weights[j*PaddedOutputDimensions + i] = read_little_endian<WeightType>(stream);

#if defined (USE_AVX2)
      for (std::size_t i = 0; i < InputDimensions; ++i)
        for (std::size_t j = 0; j < PaddedOutputDimensions; ++j)
        {
          int simdlane = j % 16;
          int simdlane64 = simdlane / 4;
          if (simdlane64 == 1)
            std::swap(weights[i*PaddedOutputDimensions + j], weights[i*PaddedOutputDimensions + j + 4]);
        }
#endif

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      for (std::size_t i = 0; i < OutputDimensions; ++i)
        write_little_endian<BiasType>(stream, biases[i]);
      for (std::size_t i = 0; i < PaddedOutputDimensions; ++i)
        for (std::size_t j = 0; j < InputDimensions; ++j)
          write_little_endian<WeightType>(stream, weights[j*PaddedOutputDimensions + i]);

      return !stream.fail();
    }

    // Forward propagation
    const OutputType* propagate(
        const InputType* input, OutputType* output) const {

#if defined (USE_AVX512)
      using vec_t = __m512i;
      #define vec_zero _mm512_setzero_si512()
      #define vec_broadcast_32(a) _mm512_set1_epi32(a)
      #define vec_unpacklo_16(a, b) _mm512_unpacklo_epi16(a, b)
      #define vec_unpackhi_16(a, b) _mm512_unpackhi_epi16(a, b)
      #define vec_add_32(a, b) _mm512_add_epi32(a, b)
      #define vec_madd_16(a, b) _mm512_madd_epi16(a, b)
#elif defined (USE_AVX2)
      using vec_t = __m256i;
      #define vec_zero _mm256_setzero_si256()
      #define vec_broadcast_32(a) _mm256_set1_epi32(a)
      #define vec_unpacklo_16(a, b) _mm256_unpacklo_epi16(a, b)
      #define vec_unpackhi_16(a, b) _mm256_unpackhi_epi16(a, b)
      #define vec_add_32(a, b) _mm256_add_epi32(a, b)
      #define vec_madd_16(a, b) _mm256_madd_epi16(a, b)
#elif defined (USE_SSE2)
      using vec_t = __m128i;
      #define vec_zero _mm_setzero_si128()
      #define vec_broadcast_32(a) _mm_set1_epi32(a)
      #define vec_unpacklo_16(a, b) _mm_unpacklo_epi16(a, b)
      #define vec_unpackhi_16(a, b) _mm_unpackhi_epi16(a, b)
      #define vec_add_32(a, b) _mm_add_epi32(a, b)
      #define vec_madd_16(a, b) _mm_madd_epi16(a, b)
#endif

#if defined (USE_SSE2)
      std::uint16_t nnzInputIndices[InputDimensions];
      IndexType numNnzInputIndices = 0;

      static_assert(InDims % 128 == 0);
      constexpr IndexType NumNnzCountChunks = InputDimensions / NnzInputSimdWidth;
      const auto inputVector = reinterpret_cast<const vec_t*>(input);
# if defined (USE_AVX512)
      for (IndexType i = 0; i < NumNnzCountChunks; i += 2) {
        const auto inputChunk0a = inputVector[i+0];
        const auto inputChunk1a = inputVector[i+1];
        std::uint64_t nnz0 = ~((std::uint64_t)_mm512_movepi8_mask(inputChunk0a));
        std::uint64_t nnz1 = ~((std::uint64_t)_mm512_movepi8_mask(inputChunk1a));
# elif defined (USE_AVX2)
      for (IndexType i = 0; i < NumNnzCountChunks; i += 4) {
        const auto inputChunk0a = inputVector[i+0];
        const auto inputChunk0b = inputVector[i+1];
        const auto inputChunk1a = inputVector[i+2];
        const auto inputChunk1b = inputVector[i+3];
        std::uint64_t nnz0 = ~(   (std::uint64_t)_mm256_movemask_epi8(inputChunk0a)
                               | ((std::uint64_t)_mm256_movemask_epi8(inputChunk0b) << 32ull));
        std::uint64_t nnz1 = ~(   (std::uint64_t)_mm256_movemask_epi8(inputChunk1a)
                               | ((std::uint64_t)_mm256_movemask_epi8(inputChunk1b) << 32ull));
# elif defined (USE_SSE2)
      for (IndexType i = 0; i < NumNnzCountChunks; i += 8) {
        const auto inputChunk0a = inputVector[i+0];
        const auto inputChunk0b = inputVector[i+1];
        const auto inputChunk0c = inputVector[i+2];
        const auto inputChunk0d = inputVector[i+3];
        const auto inputChunk1a = inputVector[i+4];
        const auto inputChunk1b = inputVector[i+5];
        const auto inputChunk1c = inputVector[i+6];
        const auto inputChunk1d = inputVector[i+7];
        std::uint64_t nnz0 = ~(   (std::uint64_t)_mm_movemask_epi8(inputChunk0a)
                               | ((std::uint64_t)_mm_movemask_epi8(inputChunk0b) << 16ull)
                               | ((std::uint64_t)_mm_movemask_epi8(inputChunk0c) << 32ull)
                               | ((std::uint64_t)_mm_movemask_epi8(inputChunk0d) << 48ull));
        std::uint64_t nnz1 = ~(   (std::uint64_t)_mm_movemask_epi8(inputChunk1a)
                               | ((std::uint64_t)_mm_movemask_epi8(inputChunk1b) << 16ull)
                               | ((std::uint64_t)_mm_movemask_epi8(inputChunk1c) << 32ull)
                               | ((std::uint64_t)_mm_movemask_epi8(inputChunk1d) << 48ull));
# endif
        const unsigned offset    = popcount(nnz0);
        const unsigned nnz1count = popcount(nnz1);
        unsigned c = 0;
        for (; c < std::min(offset, nnz1count); ++c)
        {
            const IndexType lsbIndex0 = lsb_(nnz0);
            const IndexType lsbIndex1 = lsb_(nnz1);
            nnz0 &= nnz0 - 1;
            nnz1 &= nnz1 - 1;
            nnzInputIndices[numNnzInputIndices          + c] = i*InputSimdWidth + lsbIndex0;
            nnzInputIndices[numNnzInputIndices + offset + c] = i*InputSimdWidth + lsbIndex1 + 64;
        }

        while (nnz0) {
            const IndexType lsbIndex = lsb_(nnz0);
            nnz0 &= nnz0 - 1;
            nnzInputIndices[numNnzInputIndices + c++] = i*InputSimdWidth + lsbIndex;
        }

        while (nnz1) {
            const IndexType lsbIndex = lsb_(nnz1);
            nnz1 &= nnz1 - 1;
            nnzInputIndices[numNnzInputIndices + offset + c++] = i*InputSimdWidth + lsbIndex + 64;
        }

        numNnzInputIndices += offset + nnz1count;
      }

      for (IndexType i = 0; i < OutputDimensions; ++i)
        output[i] = biases[i];

      auto outputVector = reinterpret_cast<vec_t*>(output);

      constexpr IndexType NumChunks = OutputDimensions / (OutputSimdWidth * 4);

      IndexType i = 0;
      for (; i + 3 < numNnzInputIndices; i += 4) {
        const auto mul0 = vec_broadcast_32(input[nnzInputIndices[i+0]] | (input[nnzInputIndices[i+1]] << 16));
        const auto mul2 = vec_broadcast_32(input[nnzInputIndices[i+2]] | (input[nnzInputIndices[i+3]] << 16));
        const auto col0 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+0] * PaddedOutputDimensions]);
        const auto col1 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+1] * PaddedOutputDimensions]);
        const auto col2 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+2] * PaddedOutputDimensions]);
        const auto col3 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+3] * PaddedOutputDimensions]);
        for (IndexType j = 0; j < NumChunks; ++j) {
          auto sum0 = outputVector[j*4+0];
          auto sum1 = outputVector[j*4+1];
          auto sum2 = outputVector[j*4+2];
          auto sum3 = outputVector[j*4+3];

          sum0 = vec_add_32(sum0, vec_madd_16(mul0, vec_unpacklo_16(col0[j*2+0], col1[j*2+0])));
          sum1 = vec_add_32(sum1, vec_madd_16(mul0, vec_unpackhi_16(col0[j*2+0], col1[j*2+0])));
          sum0 = vec_add_32(sum0, vec_madd_16(mul2, vec_unpacklo_16(col2[j*2+0], col3[j*2+0])));
          sum1 = vec_add_32(sum1, vec_madd_16(mul2, vec_unpackhi_16(col2[j*2+0], col3[j*2+0])));

          sum2 = vec_add_32(sum2, vec_madd_16(mul0, vec_unpacklo_16(col0[j*2+1], col1[j*2+1])));
          sum3 = vec_add_32(sum3, vec_madd_16(mul0, vec_unpackhi_16(col0[j*2+1], col1[j*2+1])));
          sum2 = vec_add_32(sum2, vec_madd_16(mul2, vec_unpacklo_16(col2[j*2+1], col3[j*2+1])));
          sum3 = vec_add_32(sum3, vec_madd_16(mul2, vec_unpackhi_16(col2[j*2+1], col3[j*2+1])));

          outputVector[j*4+0] = sum0;
          outputVector[j*4+1] = sum1;
          outputVector[j*4+2] = sum2;
          outputVector[j*4+3] = sum3;
        }
      }
      for (; i < numNnzInputIndices; ++i) {
        const auto mul0 = vec_broadcast_32(input[nnzInputIndices[i]]);
        const auto col0 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i] * PaddedOutputDimensions]);
        for (IndexType j = 0; j < NumChunks; ++j) {
          auto sum0 = outputVector[j*4+0];
          auto sum1 = outputVector[j*4+1];
          auto sum2 = outputVector[j*4+2];
          auto sum3 = outputVector[j*4+3];

          sum0 = vec_add_32(sum0, vec_madd_16(mul0, vec_unpacklo_16(col0[j*2+0], vec_zero)));
          sum1 = vec_add_32(sum1, vec_madd_16(mul0, vec_unpackhi_16(col0[j*2+0], vec_zero)));

          sum2 = vec_add_32(sum2, vec_madd_16(mul0, vec_unpacklo_16(col0[j*2+1], vec_zero)));
          sum3 = vec_add_32(sum3, vec_madd_16(mul0, vec_unpackhi_16(col0[j*2+1], vec_zero)));

          outputVector[j*4+0] = sum0;
          outputVector[j*4+1] = sum1;
          outputVector[j*4+2] = sum2;
          outputVector[j*4+3] = sum3;
        }
      }

#else
      for (IndexType i = 0; i < OutputDimensions; ++i)
        output[i] = biases[i];

      for (IndexType i = 0; i < InputDimensions; ++i) {
        if (input[i] != 0)
          for (IndexType j = 0; j < OutputDimensions; ++j)
            output[j] += weights[i*PaddedOutputDimensions + j] * input[i];
      }
#endif

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;
    using LoadedWeightType = std::int16_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) LoadedWeightType weights[InputDimensions * PaddedOutputDimensions];
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
