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

#ifdef _WIN64  // MSVC, WIN64

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

#else  // MSVC, WIN32

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

#endif

#else  // Compiler is neither GCC nor MSVC compatible

#error "Compiler not supported."

#endif

  // Affine transformation layer
  template <typename PreviousLayer, IndexType OutDims>
  class AffineTransformSparseInput {
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = std::int32_t;
    static_assert(std::is_same<InputType, std::uint8_t>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = PreviousLayer::OutputDimensions;
    static constexpr IndexType OutputDimensions = OutDims;

#if defined (USE_AVX512)
    static constexpr const IndexType InputSimdWidth = SimdWidth * 2;
    static constexpr const IndexType OutputSimdWidth = SimdWidth / 2;
#elif defined (USE_SSSE3)
    static constexpr const IndexType InputSimdWidth = SimdWidth;
    static constexpr const IndexType OutputSimdWidth = SimdWidth / 4;
#endif

    static constexpr const IndexType MaxInputSimdWidth = 64;
    static constexpr const IndexType MaxOutputSimdWidth = 64;
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxOutputSimdWidth);

    static_assert(InputDimensions % MaxInputSimdWidth == 0, "We don't account for the padding left in the serialized network data correctly yet.");

    // Size of forward propagation buffer used in this layer
    static constexpr std::size_t SelfBufferSize =
      ceil_to_multiple(OutputDimensions * sizeof(OutputType), CacheLineSize);

    // Size of the forward propagation buffer used from the input layer to this layer
    static constexpr std::size_t BufferSize =
      PreviousLayer::BufferSize + SelfBufferSize;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
      // IMPORTANT: MUST have the same hash as AffineTransform, because it's a drop in replacement.
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= PreviousLayer::get_hash_value() >> 1;
      hashValue ^= PreviousLayer::get_hash_value() << 31;
      return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      if (!previousLayer.read_parameters(stream)) return false;

      for (std::size_t i = 0; i < OutputDimensions; ++i)
        biases[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < PaddedOutputDimensions; ++i)
        for (std::size_t j = 0; j < InputDimensions; ++j)
          weights[j*PaddedOutputDimensions + i] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      if (!previousLayer.write_parameters(stream)) return false;

      for (std::size_t i = 0; i < OutputDimensions; ++i)
        write_little_endian<BiasType>(stream, biases[i]);
      for (std::size_t i = 0; i < PaddedOutputDimensions; ++i)
        for (std::size_t j = 0; j < InputDimensions; ++j)
          write_little_endian<WeightType>(stream, weights[j*PaddedOutputDimensions + i]);

      return !stream.fail();
    }

    // Forward propagation
    const OutputType* propagate(
      const TransformedFeatureType* transformedFeatures, char* buffer) const {

      const auto input = previousLayer.propagate(
          transformedFeatures, buffer + SelfBufferSize);

#if defined (USE_AVX512)

      [[maybe_unused]] const __m512i Ones512 = _mm512_set1_epi16(1);

      [[maybe_unused]] auto m512_hadd = [](__m512i sum, int bias) -> int {
        return _mm512_reduce_add_epi32(sum) + bias;
      };

      [[maybe_unused]] auto m512_add_dpbusd_epi32 = [=](__m512i& acc, __m512i a, __m512i b) {
#if defined (USE_VNNI)
        acc = _mm512_dpbusd_epi32(acc, a, b);
#else
        __m512i product0 = _mm512_maddubs_epi16(a, b);
        product0 = _mm512_madd_epi16(product0, Ones512);
        acc = _mm512_add_epi32(acc, product0);
#endif
      };

      [[maybe_unused]] auto m512_add_dpbusd_epi32x4 = [=](__m512i& acc, __m512i a0, __m512i b0, __m512i a1, __m512i b1,
                                                                        __m512i a2, __m512i b2, __m512i a3, __m512i b3) {
#if defined (USE_VNNI)
        acc = _mm512_dpbusd_epi32(acc, a0, b0);
        acc = _mm512_dpbusd_epi32(acc, a1, b1);
        acc = _mm512_dpbusd_epi32(acc, a2, b2);
        acc = _mm512_dpbusd_epi32(acc, a3, b3);
#else
        __m512i product0 = _mm512_maddubs_epi16(a0, b0);
        __m512i product1 = _mm512_maddubs_epi16(a1, b1);
        __m512i product2 = _mm512_maddubs_epi16(a2, b2);
        __m512i product3 = _mm512_maddubs_epi16(a3, b3);
        product0 = _mm512_adds_epi16(product0, product1);
        product0 = _mm512_madd_epi16(product0, Ones512);
        product2 = _mm512_adds_epi16(product2, product3);
        product2 = _mm512_madd_epi16(product2, Ones512);
        acc = _mm512_add_epi32(acc, _mm512_add_epi32(product0, product2));
#endif
      };

#endif
#if defined (USE_AVX2)

      [[maybe_unused]] const __m256i Ones256 = _mm256_set1_epi16(1);

      [[maybe_unused]] auto m256_hadd = [](__m256i sum, int bias) -> int {
        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
        return _mm_cvtsi128_si32(sum128) + bias;
      };

      [[maybe_unused]] auto m256_add_dpbusd_epi32 = [=](__m256i& acc, __m256i a, __m256i b) {
#if defined (USE_VNNI)
        acc = _mm256_dpbusd_epi32(acc, a, b);
#else
        __m256i product0 = _mm256_maddubs_epi16(a, b);
        product0 = _mm256_madd_epi16(product0, Ones256);
        acc = _mm256_add_epi32(acc, product0);
#endif
      };

      [[maybe_unused]] auto m256_add_dpbusd_epi32x4 = [=](__m256i& acc, __m256i a0, __m256i b0, __m256i a1, __m256i b1,
                                                                        __m256i a2, __m256i b2, __m256i a3, __m256i b3) {
#if defined (USE_VNNI)
        acc = _mm256_dpbusd_epi32(acc, a0, b0);
        acc = _mm256_dpbusd_epi32(acc, a1, b1);
        acc = _mm256_dpbusd_epi32(acc, a2, b2);
        acc = _mm256_dpbusd_epi32(acc, a3, b3);
#else
        __m256i product0 = _mm256_maddubs_epi16(a0, b0);
        __m256i product1 = _mm256_maddubs_epi16(a1, b1);
        __m256i product2 = _mm256_maddubs_epi16(a2, b2);
        __m256i product3 = _mm256_maddubs_epi16(a3, b3);
        product0 = _mm256_adds_epi16(product0, product1);
        product0 = _mm256_madd_epi16(product0, Ones256);
        product2 = _mm256_adds_epi16(product2, product3);
        product2 = _mm256_madd_epi16(product2, Ones256);
        acc = _mm256_add_epi32(acc, _mm256_add_epi32(product0, product2));
#endif
      };

#endif
#if defined (USE_SSSE3)

      [[maybe_unused]] const __m128i Ones128 = _mm_set1_epi16(1);

      [[maybe_unused]] auto m128_hadd = [](__m128i sum, int bias) -> int {
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E)); //_MM_PERM_BADC
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1)); //_MM_PERM_CDAB
        return _mm_cvtsi128_si32(sum) + bias;
      };

      [[maybe_unused]] auto m128_add_dpbusd_epi32 = [=](__m128i& acc, __m128i a, __m128i b) {
        __m128i product0 = _mm_maddubs_epi16(a, b);
        product0 = _mm_madd_epi16(product0, Ones128);
        acc = _mm_add_epi32(acc, product0);
      };

      [[maybe_unused]] auto m128_add_dpbusd_epi32x4 = [=](__m128i& acc, __m128i a0, __m128i b0, __m128i a1, __m128i b1,
                                                                        __m128i a2, __m128i b2, __m128i a3, __m128i b3) {
        __m128i product0 = _mm_maddubs_epi16(a0, b0);
        __m128i product1 = _mm_maddubs_epi16(a1, b1);
        __m128i product2 = _mm_maddubs_epi16(a2, b2);
        __m128i product3 = _mm_maddubs_epi16(a3, b3);
        product0 = _mm_adds_epi16(product0, product1);
        product0 = _mm_madd_epi16(product0, Ones128);
        product2 = _mm_adds_epi16(product2, product3);
        product2 = _mm_madd_epi16(product2, Ones128);
        acc = _mm_add_epi32(acc, _mm_add_epi32(product0, product2));
      };

#endif

#if defined (USE_AVX512)
      using vec_t = __m512i;
      #define vec_setzero _mm512_setzero_si512
      #define vec_nnz(a) _cvtmask64_u64(_mm512_cmpgt_epi8_mask(a, _mm512_setzero_si512()))
      #define vec_setzero _mm512_setzero_si512()
      #define vec_broadcast_8(a) _mm512_set1_epi8(a)
      #define vec_broadcast_16(a) _mm512_set1_epi16(a)
      #define vec_broadcast_32(a) _mm512_set1_epi32(a)
      #define vec_unpacklo_8(a, b) _mm512_unpacklo_epi8(a, b)
      #define vec_unpackhi_8(a, b) _mm512_unpackhi_epi8(a, b)
      #define vec_unpacklo_16(a, b) _mm512_unpacklo_epi16(a, b)
      #define vec_unpackhi_16(a, b) _mm512_unpackhi_epi16(a, b)
      #define vec_add_32(a, b) _mm512_add_epi32(a, b)
      #define vec_madd_16(a, b) _mm512_madd_epi16(a, b)
      #define vec_maddubs_16(a, b) _mm512_maddubs_epi16(a, b)
      #define vec_cmpgt_16(a, b) _mm512_cmpgt_epi16_mask(a, b)
      auto& vec_add_dpbusd_32 = m512_add_dpbusd_epi32;
      auto& vec_add_dpbusd_32x4 = m512_add_dpbusd_epi32x4;
      auto& vec_hadd = m512_hadd;
#elif defined (USE_AVX2)
      using vec_t = __m256i;
      #define vec_setzero _mm256_setzero_si256
      #define vec_nnz(a) _mm256_movemask_epi8(_mm256_cmpgt_epi8(a, _mm256_setzero_si256()))
      #define vec_setzero _mm256_setzero_si256()
      #define vec_broadcast_8(a) _mm256_set1_epi8(a)
      #define vec_broadcast_16(a) _mm256_set1_epi16(a)
      #define vec_broadcast_32(a) _mm256_set1_epi32(a)
      #define vec_unpacklo_8(a, b) _mm256_unpacklo_epi8(a, b)
      #define vec_unpackhi_8(a, b) _mm256_unpackhi_epi8(a, b)
      #define vec_unpacklo_16(a, b) _mm256_unpacklo_epi16(a, b)
      #define vec_unpackhi_16(a, b) _mm256_unpackhi_epi16(a, b)
      #define vec_add_32(a, b) _mm256_add_epi32(a, b)
      #define vec_madd_16(a, b) _mm256_madd_epi16(a, b)
      #define vec_maddubs_16(a, b) _mm256_maddubs_epi16(a, b)
      #define vec_cmpgt_16(a, b) _mm256_cmpgt_epi16(a, b)
      auto& vec_add_dpbusd_32 = m256_add_dpbusd_epi32;
      auto& vec_add_dpbusd_32x4 = m256_add_dpbusd_epi32x4;
      auto& vec_hadd = m256_hadd;
#elif defined (USE_SSE2)
      using vec_t = __m128i;
      #define vec_setzero _mm_setzero_si128
      #define vec_nnz(a) _mm_movemask_epi8(_mm_cmpgt_epi8(a, _mm_setzero_si128()))
      #define vec_setzero _mm_setzero_si128()
      #define vec_broadcast_8(a) _mm_set1_epi8(a)
      #define vec_broadcast_16(a) _mm_set1_epi16(a)
      #define vec_broadcast_32(a) _mm_set1_epi32(a)
      #define vec_unpacklo_8(a, b) _mm_unpacklo_epi8(a, b)
      #define vec_unpackhi_8(a, b) _mm_unpackhi_epi8(a, b)
      #define vec_unpacklo_16(a, b) _mm_unpacklo_epi16(a, b)
      #define vec_unpackhi_16(a, b) _mm_unpackhi_epi16(a, b)
      #define vec_add_32(a, b) _mm_add_epi32(a, b)
      #define vec_madd_16(a, b) _mm_madd_epi16(a, b)
      #define vec_maddubs_16(a, b) _mm_maddubs_epi16(a, b)
      #define vec_cmpgt_16(a, b) _mm_cmpgt_epi16(a, b)
      auto& vec_add_dpbusd_32 = m128_add_dpbusd_epi32;
      auto& vec_add_dpbusd_32x4 = m128_add_dpbusd_epi32x4;
      auto& vec_hadd = m128_hadd;
#endif

      std::uint16_t nnzInputIndices[InputDimensions];
      IndexType numNnzInputIndices = 0;

      constexpr IndexType NumNnzCountChunks = InputDimensions / InputSimdWidth;
      const auto inputVector = reinterpret_cast<const vec_t*>(input);
      const auto output = reinterpret_cast<OutputType*>(buffer);
      for (IndexType i = 0; i < NumNnzCountChunks; ++i) {
        const auto inputChunk = inputVector[i];
        auto nnz = vec_nnz(inputChunk);
        while (nnz) {
          const IndexType lsbIndex = lsb_(nnz);
          nnz &= nnz - 1;
          nnzInputIndices[numNnzInputIndices++] = i*InputSimdWidth + lsbIndex;
        }
      }

#if defined (USE_AVX2)
#error

#elif defined (USE_SSSE3)

      /* NOTE: appears much worse than the SSE2 version... */
      /*       Try using int16 weights also here, but keep the passes */

      auto outputVector = reinterpret_cast<vec_t*>(output);
      auto biasesVector = reinterpret_cast<const vec_t*>(biases);

      constexpr IndexType NumPasses = OutputDimensions / (OutputSimdWidth * 8);

      for (IndexType pass = 0; pass < NumPasses; ++pass) {
        IndexType outputOffset = pass * 8;
        auto out0 = biasesVector[outputOffset + 0];
        auto out1 = biasesVector[outputOffset + 1];
        auto out2 = biasesVector[outputOffset + 2];
        auto out3 = biasesVector[outputOffset + 3];
        auto out4 = biasesVector[outputOffset + 4];
        auto out5 = biasesVector[outputOffset + 5];
        auto out6 = biasesVector[outputOffset + 6];
        auto out7 = biasesVector[outputOffset + 7];

        IndexType i = 0;
        for (; i + 1 < numNnzInputIndices; i += 2) {
          const auto mul0 = vec_broadcast_16(input[nnzInputIndices[i+0]] | (input[nnzInputIndices[i+1]] << 8));
          const auto col0 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+0] * PaddedOutputDimensions + outputOffset * OutputSimdWidth]);
          const auto col1 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+1] * PaddedOutputDimensions + outputOffset * OutputSimdWidth]);

          {
            auto prod = vec_maddubs_16(mul0, vec_unpacklo_8(col0[0], col1[0]));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out0 = vec_add_32(out0, vec_unpacklo_16(prod, signs));
            out1 = vec_add_32(out1, vec_unpackhi_16(prod, signs));
          }
          {
            auto prod = vec_maddubs_16(mul0, vec_unpackhi_8(col0[0], col1[0]));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out2 = vec_add_32(out2, vec_unpacklo_16(prod, signs));
            out3 = vec_add_32(out3, vec_unpackhi_16(prod, signs));
          }
          {
            auto prod = vec_maddubs_16(mul0, vec_unpacklo_8(col0[1], col1[1]));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out4 = vec_add_32(out4, vec_unpacklo_16(prod, signs));
            out5 = vec_add_32(out5, vec_unpackhi_16(prod, signs));
          }
          {
            auto prod = vec_maddubs_16(mul0, vec_unpackhi_8(col0[1], col1[1]));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out6 = vec_add_32(out6, vec_unpacklo_16(prod, signs));
            out7 = vec_add_32(out7, vec_unpackhi_16(prod, signs));
          }
        }
        for (; i < numNnzInputIndices; ++i) {
          const auto mul0 = vec_broadcast_16(input[nnzInputIndices[i]]);
          const auto col0 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i] * PaddedOutputDimensions + outputOffset * OutputSimdWidth]);

          {
            auto prod = vec_maddubs_16(mul0, vec_unpacklo_8(col0[0], vec_setzero));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out0 = vec_add_32(out0, vec_unpacklo_16(prod, signs));
            out1 = vec_add_32(out1, vec_unpackhi_16(prod, signs));
          }
          {
            auto prod = vec_maddubs_16(mul0, vec_unpackhi_8(col0[0], vec_setzero));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out2 = vec_add_32(out2, vec_unpacklo_16(prod, signs));
            out3 = vec_add_32(out3, vec_unpackhi_16(prod, signs));
          }
          {
            auto prod = vec_maddubs_16(mul0, vec_unpacklo_8(col0[1], vec_setzero));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out4 = vec_add_32(out4, vec_unpacklo_16(prod, signs));
            out5 = vec_add_32(out5, vec_unpackhi_16(prod, signs));
          }
          {
            auto prod = vec_maddubs_16(mul0, vec_unpackhi_8(col0[1], vec_setzero));
            auto signs = vec_cmpgt_16(vec_setzero, prod);
            out6 = vec_add_32(out6, vec_unpacklo_16(prod, signs));
            out7 = vec_add_32(out7, vec_unpackhi_16(prod, signs));
          }
        }

        outputVector[outputOffset + 0] = out0;
        outputVector[outputOffset + 1] = out1;
        outputVector[outputOffset + 2] = out2;
        outputVector[outputOffset + 3] = out3;
        outputVector[outputOffset + 4] = out4;
        outputVector[outputOffset + 5] = out5;
        outputVector[outputOffset + 6] = out6;
        outputVector[outputOffset + 7] = out7;
      }

#elif defined (USE_SSE2)

      for (IndexType i = 0; i < OutputDimensions; ++i)
        output[i] = biases[i];

      auto outputVector = reinterpret_cast<vec_t*>(output);

      constexpr IndexType NumChunks = OutputDimensions / (OutputSimdWidth * 2);

      IndexType i = 0;
      for (; i + 3 < numNnzInputIndices; i += 4) {
        const auto mul0 = vec_broadcast_32(input[nnzInputIndices[i+0]] | (input[nnzInputIndices[i+1]] << 16));
        const auto mul2 = vec_broadcast_32(input[nnzInputIndices[i+2]] | (input[nnzInputIndices[i+3]] << 16));
        const auto col0 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+0] * PaddedOutputDimensions]);
        const auto col1 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+1] * PaddedOutputDimensions]);
        const auto col2 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+2] * PaddedOutputDimensions]);
        const auto col3 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i+3] * PaddedOutputDimensions]);
        for (IndexType j = 0; j < NumChunks; ++j) {
          auto sum0 = outputVector[j*2+0];
          auto sum1 = outputVector[j*2+1];
          auto sum2 = vec_setzero;
          auto sum3 = vec_setzero;
          sum0 = vec_add_32(sum0, vec_madd_16(mul0, vec_unpacklo_16(col0[j], col1[j])));
          sum1 = vec_add_32(sum1, vec_madd_16(mul0, vec_unpackhi_16(col0[j], col1[j])));
          sum2 =                  vec_madd_16(mul2, vec_unpacklo_16(col2[j], col3[j]));
          sum3 =                  vec_madd_16(mul2, vec_unpackhi_16(col2[j], col3[j]));
          outputVector[j*2+0] = vec_add_32(sum0, sum2);
          outputVector[j*2+1] = vec_add_32(sum1, sum3);
        }
      }
      for (; i < numNnzInputIndices; ++i) {
        const auto mul0 = vec_broadcast_32(input[nnzInputIndices[i]]);
        const auto col0 = reinterpret_cast<const vec_t*>(&weights[nnzInputIndices[i] * PaddedOutputDimensions]);
        for (IndexType j = 0; j < NumChunks; ++j) {
          auto sum0 = outputVector[j*2+0];
          auto sum1 = outputVector[j*2+1];
          sum0 = vec_add_32(sum0, vec_madd_16(mul0, vec_unpacklo_16(col0[j], vec_setzero)));
          sum1 = vec_add_32(sum1, vec_madd_16(mul0, vec_unpackhi_16(col0[j], vec_setzero)));
          outputVector[j*2+0] = sum0;
          outputVector[j*2+1] = sum1;
        }
      }

#else
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
#if defined(USE_SSSE3)
    using LoadedWeightType = std::int8_t;
#else
    using LoadedWeightType = std::int16_t;
#endif

    PreviousLayer previousLayer;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) LoadedWeightType weights[InputDimensions * PaddedOutputDimensions];
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
