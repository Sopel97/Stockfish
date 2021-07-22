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
#include <array>
#include "../nnue_common.h"

namespace Stockfish::Eval::NNUE::Layers {

static constexpr int lsb_constexpr(std::uint32_t v)
{
  int c = 0;
  if (!v) return 32;
  while (!(v & 1))
  {
    v >>= 1;
    ++c;
  }
  return c;
}

alignas(CacheLineSize) static constexpr std::array<std::array<std::uint16_t, 8>, 256> LookupTableIndices = [](){
  std::array<std::array<std::uint16_t, 8>, 256> v{};
  for (int i = 0; i < 256; ++i)
  {
    int j = i;
    int k = 0;
    while(j)
    {
      const IndexType lsbIndex = lsb_constexpr(std::uint32_t(j));
      j &= j - 1;
      v[i][k] = lsbIndex;
      ++k;
    }
  }
  return v;
}();

static constexpr std::array<std::uint8_t, 256> LookupTableCounts = [](){
  std::array<std::uint8_t, 256> v{};
  for (int i = 0; i < 256; ++i)
  {
    int j = i;
    int k = 0;
    while(j)
    {
      j &= j - 1;
      ++k;
    }
    v[i] = k;
  }
  return v;
}();

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
      ceil_to_multiple(PaddedOutputDimensions * sizeof(OutputType), CacheLineSize);

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

      for (std::size_t i = 0; i < PaddedOutputDimensions; ++i)
        weights[InputDimensions*PaddedOutputDimensions + i] = 0;

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
#elif defined (USE_AVX2)
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
#elif defined (USE_SSE2)
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
#endif

      const auto output = reinterpret_cast<OutputType*>(buffer);

#if defined (USE_SSSE3)

      alignas(CacheLineSize) std::uint16_t nnzInputIndices[InputDimensions+16];
      IndexType numNnzInputIndices = 0;
      non_zero_indices(input, nnzInputIndices, numNnzInputIndices);

      constexpr IndexType ChunkSize = 4;
      constexpr IndexType NumChunks = 8;
      constexpr IndexType TileSize = NumChunks * ChunkSize;
      static_assert(PaddedOutputDimensions % TileSize == 0);
      constexpr IndexType NumTiles = PaddedOutputDimensions / TileSize;

      const __m128i ones = _mm_set1_epi16(1);

      while (numNnzInputIndices % 4 != 0)
        nnzInputIndices[numNnzInputIndices++] = InputDimensions;

      __m128i acc[NumChunks];

      for (IndexType i = 0; i < NumTiles; ++i)
      {
        auto biasesTile = reinterpret_cast<const __m128i*>(&biases[i * TileSize]);
        auto outputTile = reinterpret_cast<      __m128i*>(&output[i * TileSize]);

        for (IndexType k = 0; k < NumChunks; ++k)
          acc[k] = biasesTile[k];

        for (IndexType j = 0; j < numNnzInputIndices; j += 4)
        {
          const auto mul0 = _mm_set1_epi16(input[nnzInputIndices[j+0]] | (input[nnzInputIndices[j+1]] << 8));
          const auto mul2 = _mm_set1_epi16(input[nnzInputIndices[j+2]] | (input[nnzInputIndices[j+3]] << 8));
          const auto col0 = reinterpret_cast<const __m128i*>(&weights[nnzInputIndices[j+0] * PaddedOutputDimensions + i * TileSize]);
          const auto col1 = reinterpret_cast<const __m128i*>(&weights[nnzInputIndices[j+1] * PaddedOutputDimensions + i * TileSize]);
          const auto col2 = reinterpret_cast<const __m128i*>(&weights[nnzInputIndices[j+2] * PaddedOutputDimensions + i * TileSize]);
          const auto col3 = reinterpret_cast<const __m128i*>(&weights[nnzInputIndices[j+3] * PaddedOutputDimensions + i * TileSize]);
          for (IndexType k = 0; k < NumChunks / 4; ++k)
          {
            __m128i prod0 = _mm_maddubs_epi16(mul0, _mm_unpacklo_epi8(col0[k], col1[k]));
            __m128i prod1 = _mm_maddubs_epi16(mul0, _mm_unpackhi_epi8(col0[k], col1[k]));
            __m128i prod2 = _mm_maddubs_epi16(mul2, _mm_unpacklo_epi8(col2[k], col3[k]));
            __m128i prod3 = _mm_maddubs_epi16(mul2, _mm_unpackhi_epi8(col2[k], col3[k]));
            acc[k*4 + 0] = _mm_add_epi32(acc[k*4 + 0], _mm_madd_epi16(ones, _mm_unpacklo_epi16(prod0, prod2)));
            acc[k*4 + 1] = _mm_add_epi32(acc[k*4 + 1], _mm_madd_epi16(ones, _mm_unpackhi_epi16(prod0, prod2)));
            acc[k*4 + 2] = _mm_add_epi32(acc[k*4 + 2], _mm_madd_epi16(ones, _mm_unpacklo_epi16(prod1, prod3)));
            acc[k*4 + 3] = _mm_add_epi32(acc[k*4 + 3], _mm_madd_epi16(ones, _mm_unpackhi_epi16(prod1, prod3)));
          }
        }

        for (IndexType k = 0; k < NumChunks; ++k)
          outputTile[k] = acc[k];
      }

#elif defined (USE_SSE2)

      alignas(CacheLineSize) std::uint16_t nnzInputIndices[InputDimensions+16];
      IndexType numNnzInputIndices = 0;
      non_zero_indices(input, nnzInputIndices, numNnzInputIndices);

      constexpr IndexType ChunkSize = 4;
      constexpr IndexType NumChunks = 8;
      constexpr IndexType TileSize = NumChunks * ChunkSize;
      static_assert(PaddedOutputDimensions % TileSize == 0);
      constexpr IndexType NumTiles = PaddedOutputDimensions / TileSize;

      while (numNnzInputIndices % 2 != 0)
        nnzInputIndices[numNnzInputIndices++] = InputDimensions;

      __m128i acc[NumChunks];

      for (IndexType i = 0; i < NumTiles; ++i)
      {
        auto biasesTile = reinterpret_cast<const __m128i*>(&biases[i * TileSize]);
        auto outputTile = reinterpret_cast<      __m128i*>(&output[i * TileSize]);

        for (IndexType k = 0; k < NumChunks; ++k)
          acc[k] = biasesTile[k];

        for (IndexType j = 0; j < numNnzInputIndices; j += 2)
        {
          const auto mul0 = _mm_set1_epi32(input[nnzInputIndices[j+0]] | (input[nnzInputIndices[j+1]] << 16));
          const auto col0 = reinterpret_cast<const __m128i*>(&weights[nnzInputIndices[j+0] * PaddedOutputDimensions + i * TileSize]);
          const auto col1 = reinterpret_cast<const __m128i*>(&weights[nnzInputIndices[j+1] * PaddedOutputDimensions + i * TileSize]);
          for (IndexType k = 0; k < NumChunks / 2; ++k)
          {
            acc[k*2 + 0] = _mm_add_epi32(acc[k*2 + 0], _mm_madd_epi16(mul0, _mm_unpacklo_epi16(col0[k], col1[k])));
            acc[k*2 + 1] = _mm_add_epi32(acc[k*2 + 1], _mm_madd_epi16(mul0, _mm_unpackhi_epi16(col0[k], col1[k])));
          }
        }

        for (IndexType k = 0; k < NumChunks; ++k)
          outputTile[k] = acc[k];
      }

#else
      std::memcpy(output, biases, sizeof(BiasType) * OutputDimensions);

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

#if defined(USE_SSE2) && !defined(USE_SSSE3)
    using LoadedWeightType = std::int16_t;
#else
    using LoadedWeightType = std::int8_t;
#endif

    PreviousLayer previousLayer;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) LoadedWeightType weights[(InputDimensions + 1) * PaddedOutputDimensions];


#if defined (USE_AVX2)

    static inline void non_zero_indices(const std::uint8_t* in, std::uint16_t* out, IndexType& count_out)
    {
        static constexpr unsigned NumChunks = InputDimensions / 32;

        const auto inputVector = reinterpret_cast<const __m256i*>(in);
        unsigned count = 0;
        __m128i base = _mm_set1_epi16(0);
        __m128i increment = _mm_set1_epi16(8);
        for (unsigned i = 0; i < NumChunks; ++i)
        {
            const __m256i inputChunk = inputVector[i];
            unsigned nnz = _mm256_movemask_epi8(_mm256_cmpgt_epi8(inputChunk, _mm256_setzero_si256()));
            unsigned b3 = (nnz >> 24) & 0xFF;
            unsigned b2 = (nnz >> 16) & 0xFF;
            unsigned b1 = (nnz >> 8) & 0xFF;
            unsigned b0 = (nnz) & 0xFF;
            unsigned c0 = LookupTableCounts[b0];
            unsigned c1 = LookupTableCounts[b1];
            unsigned c2 = LookupTableCounts[b2];
            unsigned c3 = LookupTableCounts[b3];
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b0])) + base);
            count += c0;
            base += increment;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b1])) + base);
            count += c1;
            base += increment;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b2])) + base);
            count += c2;
            base += increment;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b3])) + base);
            count += c3;
            base += increment;
        }
        count_out = count;
    }

#elif defined (USE_SSE2)

    static inline void non_zero_indices(const std::uint8_t* in, std::uint16_t* out, IndexType& count_out)
    {
        static constexpr unsigned NumChunks = InputDimensions / 16;

        const auto inputVector = reinterpret_cast<const __m128i*>(in);
        unsigned count = 0;
        __m128i base = _mm_set1_epi16(0);
        __m128i increment = _mm_set1_epi16(8);
        for (unsigned i = 0; i < NumChunks; ++i)
        {
            const __m128i inputChunk = inputVector[i];
            unsigned nnz = _mm_movemask_epi8(_mm_cmpgt_epi8(inputChunk, _mm_setzero_si128()));
            unsigned b1 = (nnz >> 8) & 0xFF;
            unsigned b0 = (nnz) & 0xFF;
            unsigned c0 = LookupTableCounts[b0];
            unsigned c1 = LookupTableCounts[b1];
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b0])) + base);
            count += c0;
            base += increment;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(out + count), _mm_loadu_si128(reinterpret_cast<const __m128i*>(&LookupTableIndices[b1])) + base);
            count += c1;
            base += increment;
        }
        count_out = count;
    }

#else

    static inline void non_zero_indices(const std::uint8_t* in, std::uint16_t* out, IndexType& count_out)
    {
      unsigned count = 0;
      for (unsigned i = 0; i < InputDimensions; ++i)
      {
        if (in[i])
          out[count++] = i;
      }
      count_out = count;
    }

#endif
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_SPARSE_INPUT_H_INCLUDED
