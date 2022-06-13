/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2022 The Stockfish developers (see AUTHORS file)

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

#ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
#define NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED

#include <iostream>
#include <algorithm>
#include <type_traits>
#include "../nnue_common.h"
#include "../../simd.h"

/*
  This file contains the definition for a fully connected layer (aka affine transform).
  Two approaches are employed, depending on the sizes of the transform.

  Approach 1:
    - used when the PaddedInputDimensions >= 128
    - uses AVX512 if possible
    - processes inputs in batches of 2*InputSimdWidth
      - so in batches of 128 for AVX512
    - the weight blocks of size InputSimdWidth are transposed such that
      access is sequential
    - N columns of the weight matrix are processed a time, where N
      depends on the architecture (the amount of registers)
    - accumulate + hadd is used

  Approach 2:
    - used when the PaddedInputDimensions < 128
    - does not use AVX512
    - expected use-case is for when PaddedInputDimensions == 32 and InputDimensions <= 32.
      - that's why AVX512 is hard to implement
    - expected use-case is small layers
      - not optimized as well as the approach 1
    - inputs are processed in chunks of 4, weights are respectively transposed
    - accumulation happens directly to int32s
*/

namespace Stockfish::Eval::NNUE::Layers {

  template <IndexType InDims, IndexType OutDims, typename Enabled = void>
  class AffineTransform;

  // A specialization for large inputs.
  template <IndexType InDims, IndexType OutDims>
  class AffineTransform<InDims, OutDims, std::enable_if_t<(ceil_to_multiple<IndexType>(InDims, MaxSimdWidth) >= 2*64)>> {
   public:
    // Input/output type
    using InputType = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static_assert(PaddedInputDimensions >= 128, "Something went wrong. This specialization should not have been chosen.");

#if defined (USE_AVX512)
    static constexpr const IndexType InputSimdWidth = 64;
    static constexpr const IndexType MaxNumOutputRegs = 16;
#elif defined (USE_AVX2)
    static constexpr const IndexType InputSimdWidth = 32;
    static constexpr const IndexType MaxNumOutputRegs = 8;
#elif defined (USE_SSSE3)
    static constexpr const IndexType InputSimdWidth = 16;
    static constexpr const IndexType MaxNumOutputRegs = 8;
#elif defined (USE_NEON)
    static constexpr const IndexType InputSimdWidth = 8;
    static constexpr const IndexType MaxNumOutputRegs = 8;
#else
    // The fallback implementation will not have permuted weights.
    // We define these to avoid a lot of ifdefs later.
    static constexpr const IndexType InputSimdWidth = 1;
    static constexpr const IndexType MaxNumOutputRegs = 1;
#endif

    // A big block is a region in the weight matrix of the size [PaddedInputDimensions, NumOutputRegs].
    // A small block is a region of size [InputSimdWidth, 1]

    static constexpr const IndexType NumOutputRegs = std::min(MaxNumOutputRegs, OutputDimensions);
    static constexpr const IndexType SmallBlockSize = InputSimdWidth;
    static constexpr const IndexType BigBlockSize = NumOutputRegs * PaddedInputDimensions;
    static constexpr const IndexType NumSmallBlocksInBigBlock = BigBlockSize / SmallBlockSize;
    static constexpr const IndexType NumSmallBlocksPerOutput = PaddedInputDimensions / SmallBlockSize;
    static constexpr const IndexType NumBigBlocks = OutputDimensions / NumOutputRegs;

    static_assert(OutputDimensions % NumOutputRegs == 0);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= prevHash >> 1;
      hashValue ^= prevHash << 31;
      return hashValue;
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      for (IndexType i = 0; i < OutputDimensions; ++i)
        biases[i] = read_little_endian<BiasType>(stream);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        weights[i] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      for (IndexType i = 0; i < OutputDimensions; ++i)
          write_little_endian<BiasType>(stream, biases[i]);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        write_little_endian<WeightType>(stream, weights[i]);

      return !stream.fail();
    }

    // Forward propagation
    const OutputType* propagate(
        const InputType* input, OutputType* output) const {

#if defined (USE_AVX512)
      using acc_vec_t = __m512i;
      using weight_vec_t = __m512i;
      using in_vec_t = __m512i;
      #define vec_zero _mm512_setzero_si512()
      #define vec_add_dpbusd_32x2 Simd::m512_add_dpbusd_epi32x2
      #define vec_hadd Simd::m512_hadd
#elif defined (USE_AVX2)
      using acc_vec_t = __m256i;
      using weight_vec_t = __m256i;
      using in_vec_t = __m256i;
      #define vec_zero _mm256_setzero_si256()
      #define vec_add_dpbusd_32x2 Simd::m256_add_dpbusd_epi32x2
      #define vec_hadd Simd::m256_hadd
#elif defined (USE_SSSE3)
      using acc_vec_t = __m128i;
      using weight_vec_t = __m128i;
      using in_vec_t = __m128i;
      #define vec_zero _mm_setzero_si128()
      #define vec_add_dpbusd_32x2 Simd::m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::m128_hadd
#elif defined (USE_NEON)
      using acc_vec_t = int32x4_t;
      using weight_vec_t = int8x8_t;
      using in_vec_t = int8x8_t;
      #define vec_zero {0}
      #define vec_add_dpbusd_32x2 Simd::neon_m128_add_dpbusd_epi32x2
      #define vec_hadd Simd::neon_m128_hadd
#else
      #error "not supported"
#endif

      const in_vec_t* invec = reinterpret_cast<const in_vec_t*>(input);
      const weight_vec_t* weightvec = reinterpret_cast<const weight_vec_t*>(&weights[PaddedInputDimensions * 15]);
      acc_vec_t acc = vec_zero;
      for (IndexType i = 0; i < InputDimensions / InputSimdWidth; i += 2)
      {
        vec_add_dpbusd_32x2(acc, invec[i], weightvec[i], invec[i+1], weightvec[i+1]);
      }
      output[15] = vec_hadd(acc, biases[15]);

# undef vec_zero
# undef vec_add_dpbusd_32x2
# undef vec_hadd
# undef vec_haddx4

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
  };

  template <IndexType InDims, IndexType OutDims>
  class AffineTransform<InDims, OutDims, std::enable_if_t<(ceil_to_multiple<IndexType>(InDims, MaxSimdWidth) < 2*64)>> {
   public:
    // Input/output type
    // Input/output type
    using InputType = std::uint8_t;
    using OutputType = std::int32_t;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions = InDims;
    static constexpr IndexType OutputDimensions = OutDims;

    static constexpr IndexType PaddedInputDimensions =
      ceil_to_multiple<IndexType>(InputDimensions, MaxSimdWidth);
    static constexpr IndexType PaddedOutputDimensions =
      ceil_to_multiple<IndexType>(OutputDimensions, MaxSimdWidth);

    using OutputBuffer = OutputType[PaddedOutputDimensions];

    static_assert(PaddedInputDimensions < 128, "Something went wrong. This specialization should not have been chosen.");

#if defined (USE_SSSE3)
    static constexpr const IndexType OutputSimdWidth = SimdWidth / 4;
    static constexpr const IndexType InputSimdWidth = SimdWidth;
#endif

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value(std::uint32_t prevHash) {
      std::uint32_t hashValue = 0xCC03DAE4u;
      hashValue += OutputDimensions;
      hashValue ^= prevHash >> 1;
      hashValue ^= prevHash << 31;
      return hashValue;
    }

    static IndexType get_weight_index_scrambled(IndexType i)
    {
      return
        (i / 4) % (PaddedInputDimensions / 4) * OutputDimensions * 4 +
        i / PaddedInputDimensions * 4 +
        i % 4;
    }

    static IndexType get_weight_index(IndexType i)
    {
#if defined (USE_SSSE3)
      return get_weight_index_scrambled(i);
#else
      return i;
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {
      for (IndexType i = 0; i < OutputDimensions; ++i)
        biases[i] = read_little_endian<BiasType>(stream);
      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        weights[get_weight_index(i)] = read_little_endian<WeightType>(stream);

      return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {
      for (IndexType i = 0; i < OutputDimensions; ++i)
        write_little_endian<BiasType>(stream, biases[i]);

      for (IndexType i = 0; i < OutputDimensions * PaddedInputDimensions; ++i)
        write_little_endian<WeightType>(stream, weights[get_weight_index(i)]);

      return !stream.fail();
    }
    // Forward propagation
    const OutputType* propagate(
        const InputType* input, OutputType* output) const {

#if defined (USE_AVX2)
      using vec_t = __m256i;
      #define vec_setzero _mm256_setzero_si256
      #define vec_set_32 _mm256_set1_epi32
      #define vec_add_dpbusd_32 Simd::m256_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m256_add_dpbusd_epi32x2
      #define vec_add_dpbusd_32x4 Simd::m256_add_dpbusd_epi32x4
      #define vec_hadd Simd::m256_hadd
      #define vec_haddx4 Simd::m256_haddx4
#elif defined (USE_SSSE3)
      using vec_t = __m128i;
      #define vec_setzero _mm_setzero_si128
      #define vec_set_32 _mm_set1_epi32
      #define vec_add_dpbusd_32 Simd::m128_add_dpbusd_epi32
      #define vec_add_dpbusd_32x2 Simd::m128_add_dpbusd_epi32x2
      #define vec_add_dpbusd_32x4 Simd::m128_add_dpbusd_epi32x4
      #define vec_hadd Simd::m128_hadd
      #define vec_haddx4 Simd::m128_haddx4
#endif

#if defined (USE_SSSE3)
      const auto inputVector = reinterpret_cast<const vec_t*>(input);

      static_assert(OutputDimensions % OutputSimdWidth == 0 || OutputDimensions == 1);

      if constexpr (OutputDimensions % OutputSimdWidth == 0)
      {
        constexpr IndexType NumChunks = ceil_to_multiple<IndexType>(InputDimensions, 8) / 4;
        constexpr IndexType NumRegs = OutputDimensions / OutputSimdWidth;

        const auto input32 = reinterpret_cast<const std::int32_t*>(input);
        const vec_t* biasvec = reinterpret_cast<const vec_t*>(biases);
        vec_t acc[NumRegs];
        for (IndexType k = 0; k < NumRegs; ++k)
          acc[k] = biasvec[k];

        for (IndexType i = 0; i < NumChunks; i += 2)
        {
          const vec_t in0 = vec_set_32(input32[i + 0]);
          const vec_t in1 = vec_set_32(input32[i + 1]);
          const auto col0 = reinterpret_cast<const vec_t*>(&weights[(i + 0) * OutputDimensions * 4]);
          const auto col1 = reinterpret_cast<const vec_t*>(&weights[(i + 1) * OutputDimensions * 4]);
          for (IndexType k = 0; k < NumRegs; ++k)
            vec_add_dpbusd_32x2(acc[k], in0, col0[k], in1, col1[k]);
        }

        vec_t* outptr = reinterpret_cast<vec_t*>(output);
        for (IndexType k = 0; k < NumRegs; ++k)
          outptr[k] = acc[k];
      }
      else if constexpr (OutputDimensions == 1)
      {
        constexpr IndexType NumChunks = PaddedInputDimensions / SimdWidth;
        vec_t sum0 = vec_setzero();
        const auto row0 = reinterpret_cast<const vec_t*>(&weights[0]);

        for (int j = 0; j < (int)NumChunks; ++j)
        {
          const vec_t in = inputVector[j];
          vec_add_dpbusd_32(sum0, in, row0[j]);
        }
        output[0] = vec_hadd(sum0, biases[0]);
      }

# undef vec_setzero
# undef vec_set_32
# undef vec_add_dpbusd_32
# undef vec_add_dpbusd_32x2
# undef vec_add_dpbusd_32x4
# undef vec_hadd
# undef vec_haddx4
#else
      // Use old implementation for the other architectures.
      affine_transform_non_ssse3<
        InputDimensions,
        PaddedInputDimensions,
        OutputDimensions>(output, weights, biases, input);
#endif

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    alignas(CacheLineSize) BiasType biases[OutputDimensions];
    alignas(CacheLineSize) WeightType weights[OutputDimensions * PaddedInputDimensions];
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
