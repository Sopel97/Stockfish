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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring>
#include <string>

namespace Eval::NNUE {

  // If vector instructions are enabled, we update and refresh the
  // accumulator tile by tile such that each tile fits in the CPU's
  // vector registers.
  #define VECTOR

  #ifdef USE_AVX512
  typedef __m512i vec_t;
  #define vec_load(a) _mm512_load_si512(a)
  #define vec_store(a,b) _mm512_store_si512(a,b)
  #define vec_add_16(a,b) _mm512_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm512_sub_epi16(a,b)
  #define vec_zero _mm512_setzero_si512()
  static constexpr IndexType kNumRegs = 8; // only 8 are needed

  #elif USE_AVX2
  typedef __m256i vec_t;
  #define vec_load(a) _mm256_load_si256(a)
  #define vec_store(a,b) _mm256_store_si256(a,b)
  #define vec_add_16(a,b) _mm256_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
  #define vec_zero _mm256_setzero_si256()
  static constexpr IndexType kNumRegs = 16;

  #elif USE_SSE2
  typedef __m128i vec_t;
  #define vec_load(a) (*(a))
  #define vec_store(a,b) *(a)=(b)
  #define vec_add_16(a,b) _mm_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm_sub_epi16(a,b)
  #define vec_zero _mm_setzero_si128()
  static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;

  #elif USE_MMX
  typedef __m64 vec_t;
  #define vec_load(a) (*(a))
  #define vec_store(a,b) *(a)=(b)
  #define vec_add_16(a,b) _mm_add_pi16(a,b)
  #define vec_sub_16(a,b) _mm_sub_pi16(a,b)
  #define vec_zero _mm_setzero_si64()
  static constexpr IndexType kNumRegs = 8;

  #elif USE_NEON
  typedef int16x8_t vec_t;
  #define vec_load(a) (*(a))
  #define vec_store(a,b) *(a)=(b)
  #define vec_add_16(a,b) vaddq_s16(a,b)
  #define vec_sub_16(a,b) vsubq_s16(a,b)
  #define vec_zero {0}
  static constexpr IndexType kNumRegs = 16;

  #else
  #undef VECTOR

  #endif

  // Input feature converter
  class FeatureTransformer {

   private:
    // Number of output dimensions for one side
    static constexpr IndexType kHalfDimensions = kTransformedFeatureDimensions;

    #ifdef VECTOR
    static constexpr IndexType kTileHeight = kNumRegs * sizeof(vec_t) / 2;
    static_assert(kHalfDimensions % kTileHeight == 0, "kTileHeight must divide kHalfDimensions");
    #endif

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions = RawFeatures::kDimensions;
    static constexpr IndexType kOutputDimensions = kHalfDimensions * 2;

    // Size of forward propagation buffer
    static constexpr std::size_t kBufferSize =
        kOutputDimensions * sizeof(OutputType);

    static constexpr int kLayerIndex = 0;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {

      return RawFeatures::kHashValue ^ kOutputDimensions;
    }

    static std::string get_name() {
      return RawFeatures::get_name() + "[" +
          std::to_string(kInputDimensions) + "->" +
          std::to_string(kHalfDimensions) + "x2]";
    }

    // a string representing the structure
    static std::string get_structure_string() {
      return get_name();
    }

    static std::string get_layers_info() {
      std::string info = "  - ";
      info += std::to_string(kLayerIndex);
      info += " - ";
      info += get_name();
      return info;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {

      for (std::size_t i = 0; i < kHalfDimensions; ++i)
        biases_[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < kInputDimensions; ++i)
      {
        for (std::size_t j = 0; j < kHalfDimensions; ++j)
        {
          weights_[i*kHalfDimensions + j] = read_little_endian<WeightType>(stream);
        }
        psqt_values_[i] = read_little_endian<PSQTValueType>(stream);
      }
      return !stream.fail();
    }

    // write parameters
    bool WriteParameters(std::ostream& stream) const {
      stream.write(reinterpret_cast<const char*>(biases_),
          kHalfDimensions * sizeof(BiasType));

      for (std::size_t i = 0; i < kInputDimensions; ++i)
      {
        stream.write(reinterpret_cast<const char*>(weights_+i*kHalfDimensions),
          kHalfDimensions * sizeof(WeightType));
        stream.write(reinterpret_cast<const char*>(psqt_values_+i), 1 * sizeof(PSQTValueType));
      }

      return !stream.fail();
    }

    // Proceed with the difference calculation if possible
    bool update_accumulator_if_possible(const Position& pos) const {

      const auto now = pos.state();
      if (now->accumulator.computed_accumulation)
        return true;

      const auto prev = now->previous;
      if (prev && prev->accumulator.computed_accumulation) {
        update_accumulator(pos);
        return true;
      }

      return false;
    }

    // Convert input features
    void Transform(const Position& pos, OutputType* output, std::int32_t& psqt) const {

      if (!update_accumulator_if_possible(pos))
        refresh_accumulator(pos);

      const auto& accumulation = pos.state()->accumulator.accumulation;
      const auto& psqt_accumulation = pos.state()->accumulator.psqt_accumulation;

  #if defined(USE_AVX512)
      constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth * 2);
      static_assert(kHalfDimensions % (kSimdWidth * 2) == 0);
      const __m512i kControl = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
      const __m512i kZero = _mm512_setzero_si512();

  #elif defined(USE_AVX2)
      constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
      constexpr int kControl = 0b11011000;
      const __m256i kZero = _mm256_setzero_si256();

  #elif defined(USE_SSE2)
      constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;

  #ifdef USE_SSE41
      const __m128i kZero = _mm_setzero_si128();
  #else
      const __m128i k0x80s = _mm_set1_epi8(-128);
  #endif

  #elif defined(USE_MMX)
      constexpr IndexType kNumChunks = kHalfDimensions / kSimdWidth;
      const __m64 k0x80s = _mm_set1_pi8(-128);

  #elif defined(USE_NEON)
      constexpr IndexType kNumChunks = kHalfDimensions / (kSimdWidth / 2);
      const int8x8_t kZero = {0};
  #endif

      const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
      for (IndexType p = 0; p < 2; ++p) {
        const IndexType offset = kHalfDimensions * p;

  #if defined(USE_AVX512)
        auto out = reinterpret_cast<__m512i*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m512i sum0 = _mm512_load_si512(
              &reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
          __m512i sum1 = _mm512_load_si512(
              &reinterpret_cast<const __m512i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
          for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
              sum0 = _mm512_add_epi16(sum0, reinterpret_cast<const __m512i*>(
                  accumulation[perspectives[p]][i])[j * 2 + 0]);
              sum1 = _mm512_add_epi16(sum1, reinterpret_cast<const __m512i*>(
                  accumulation[perspectives[p]][i])[j * 2 + 1]);
          }

          _mm512_store_si512(&out[j], _mm512_permutexvar_epi64(kControl,
              _mm512_max_epi8(_mm512_packs_epi16(sum0, sum1), kZero)));
        }

  #elif defined(USE_AVX2)
        auto out = reinterpret_cast<__m256i*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m256i sum0 = _mm256_load_si256(
              &reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 0]);
          __m256i sum1 = _mm256_load_si256(
              &reinterpret_cast<const __m256i*>(accumulation[perspectives[p]][0])[j * 2 + 1]);
          for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
              sum0 = _mm256_add_epi16(sum0, reinterpret_cast<const __m256i*>(
                  accumulation[perspectives[p]][i])[j * 2 + 0]);
              sum1 = _mm256_add_epi16(sum1, reinterpret_cast<const __m256i*>(
                  accumulation[perspectives[p]][i])[j * 2 + 1]);
          }

          _mm256_store_si256(&out[j], _mm256_permute4x64_epi64(_mm256_max_epi8(
              _mm256_packs_epi16(sum0, sum1), kZero), kControl));
        }

  #elif defined(USE_SSE2)
        auto out = reinterpret_cast<__m128i*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m128i sum0 = _mm_load_si128(&reinterpret_cast<const __m128i*>(
              accumulation[perspectives[p]][0])[j * 2 + 0]);
          __m128i sum1 = _mm_load_si128(&reinterpret_cast<const __m128i*>(
              accumulation[perspectives[p]][0])[j * 2 + 1]);
          for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
            sum0 = _mm_add_epi16(sum0, reinterpret_cast<const __m128i*>(
                accumulation[perspectives[p]][i])[j * 2 + 0]);
            sum1 = _mm_add_epi16(sum1, reinterpret_cast<const __m128i*>(
                accumulation[perspectives[p]][i])[j * 2 + 1]);
          }

      const __m128i packedbytes = _mm_packs_epi16(sum0, sum1);

          _mm_store_si128(&out[j],

  #ifdef USE_SSE41
              _mm_max_epi8(packedbytes, kZero)
  #else
              _mm_subs_epi8(_mm_adds_epi8(packedbytes, k0x80s), k0x80s)
  #endif

          );
        }

  #elif defined(USE_MMX)
        auto out = reinterpret_cast<__m64*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          __m64 sum0 = *(&reinterpret_cast<const __m64*>(
              accumulation[perspectives[p]][0])[j * 2 + 0]);
          __m64 sum1 = *(&reinterpret_cast<const __m64*>(
              accumulation[perspectives[p]][0])[j * 2 + 1]);
          for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
              sum0 = _mm_add_pi16(sum0, reinterpret_cast<const __m64*>(
                  accumulation[perspectives[p]][i])[j * 2 + 0]);
              sum1 = _mm_add_pi16(sum1, reinterpret_cast<const __m64*>(
                  accumulation[perspectives[p]][i])[j * 2 + 1]);
          }

          const __m64 packedbytes = _mm_packs_pi16(sum0, sum1);
          out[j] = _mm_subs_pi8(_mm_adds_pi8(packedbytes, k0x80s), k0x80s);
        }

  #elif defined(USE_NEON)
        const auto out = reinterpret_cast<int8x8_t*>(&output[offset]);
        for (IndexType j = 0; j < kNumChunks; ++j) {
          int16x8_t sum = reinterpret_cast<const int16x8_t*>(
              accumulation[perspectives[p]][0])[j];

          for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
              sum = vaddq_s16(sum, reinterpret_cast<const int16x8_t*>(
                  accumulation[perspectives[p]][i])[j]);
          }

          out[j] = vmax_s8(vqmovn_s16(sum), kZero);
        }

  #else
        for (IndexType j = 0; j < kHalfDimensions; ++j) {
          BiasType sum = accumulation[static_cast<int>(perspectives[p])][0][j];
          for (IndexType i = 1; i < kRefreshTriggers.size(); ++i) {
              sum += accumulation[static_cast<int>(perspectives[p])][i][j];
          }

          output[offset + j] = static_cast<OutputType>(
              std::max<int>(0, std::min<int>(127, sum)));
        }
  #endif

      }

      psqt = 0;
      for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
        psqt += psqt_accumulation[static_cast<int>(perspectives[0])][i];
        psqt -= psqt_accumulation[static_cast<int>(perspectives[1])][i];
      }
      psqt /= 2;

  #if defined(USE_MMX)
      _mm_empty();
  #endif
    }

   private:
    // Calculate cumulative value without using difference calculation
    void refresh_accumulator(const Position& pos) const {

  #ifdef VECTOR
      // Gcc-10.2 unnecessarily spills AVX2 registers if this array
      // is defined in the VECTOR code below, once in each branch
      vec_t acc[kNumRegs];
  #endif
      auto& accumulator = pos.state()->accumulator;
      for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
        Features::IndexList active_indices[2];
        RawFeatures::append_active_indices(pos, kRefreshTriggers[i],
                                           active_indices);
          for (Color perspective : { WHITE, BLACK }) {
#ifdef VECTOR
            for (IndexType j = 0; j < kHalfDimensions / kTileHeight; ++j) {
              auto accTile = reinterpret_cast<vec_t*>(
                  &accumulator.accumulation[perspective][i][j * kTileHeight]);

              if (i == 0) {
                auto biasesTile = reinterpret_cast<const vec_t*>(
                    &biases_[j * kTileHeight]);
                for (IndexType k = 0; k < kNumRegs; ++k)
                  acc[k] = biasesTile[k];
              } else {
                for (IndexType k = 0; k < kNumRegs; ++k)
                  acc[k] = vec_zero;
              }

              accumulator.psqt_accumulation[perspective][i] = 0;

              for (const auto index : active_indices[perspective]) {
                const IndexType offset = kHalfDimensions * index + j * kTileHeight;
                auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);

                for (IndexType k = 0; k < kNumRegs; ++k)
                  acc[k] = vec_add_16(acc[k], column[k]);

                accumulator.psqt_accumulation[perspective][i] += psqt_values_[index];
              }

              for (IndexType k = 0; k < kNumRegs; k++)
                vec_store(&accTile[k], acc[k]);
            }
#else
            if (i == 0) {
              std::memcpy(accumulator.accumulation[perspective][i], biases_,
                          kHalfDimensions * sizeof(BiasType));
            } else {
              std::memset(accumulator.accumulation[perspective][i], 0,
                          kHalfDimensions * sizeof(BiasType));
            }

            accumulator.psqt_accumulation[perspective][i] = 0;

            for (const auto index : active_indices[perspective]) {
              const IndexType offset = kHalfDimensions * index;

              for (IndexType j = 0; j < kHalfDimensions; ++j)
                accumulator.accumulation[perspective][i][j] += weights_[offset + j];

              accumulator.psqt_accumulation[perspective][i] += psqt_values_[index];
            }
#endif
          }

        }

#if defined(USE_MMX)
        _mm_empty();
#endif

        accumulator.computed_accumulation = true;
    }

    // Calculate cumulative value using difference calculation
    void update_accumulator(const Position& pos) const {

  #ifdef VECTOR
      // Gcc-10.2 unnecessarily spills AVX2 registers if this array
      // is defined in the VECTOR code below, once in each branch
      vec_t acc[kNumRegs];
  #endif
    const auto& prev_accumulator = pos.state()->previous->accumulator;
    auto& accumulator = pos.state()->accumulator;
    for (IndexType i = 0; i < kRefreshTriggers.size(); ++i) {
      Features::IndexList removed_indices[2], added_indices[2];
      bool reset[2] = { false, false };
      RawFeatures::append_changed_indices(pos, kRefreshTriggers[i],
                                          removed_indices, added_indices, reset);

#ifdef VECTOR
      for (IndexType j = 0; j < kHalfDimensions / kTileHeight; ++j) {
        for (Color perspective : { WHITE, BLACK }) {
          auto accTile = reinterpret_cast<vec_t*>(
              &accumulator.accumulation[perspective][i][j * kTileHeight]);

          if (reset[perspective]) {
            if (i == 0) {
              auto biasesTile = reinterpret_cast<const vec_t*>(
                  &biases_[j * kTileHeight]);
              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = biasesTile[k];
            } else {
              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_zero;
            }

            accumulator.psqt_accumulation[perspective][i] = 0;

          } else {
            auto prevAccTile = reinterpret_cast<const vec_t*>(
                &prev_accumulator.accumulation[perspective][i][j * kTileHeight]);

            for (IndexType k = 0; k < kNumRegs; ++k)
              acc[k] = vec_load(&prevAccTile[k]);

            accumulator.psqt_accumulation[perspective][i] = prev_accumulator.psqt_accumulation[perspective][i];

            // Difference calculation for the deactivated features
            for (const auto index : removed_indices[perspective]) {
              const IndexType offset = kHalfDimensions * index + j * kTileHeight;
              auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);

              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_sub_16(acc[k], column[k]);

              accumulator.psqt_accumulation[perspective][i] -= psqt_values_[index];
            }
          }

          { // Difference calculation for the activated features
            for (const auto index : added_indices[perspective]) {
              const IndexType offset = kHalfDimensions * index + j * kTileHeight;
              auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);

              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_add_16(acc[k], column[k]);

              accumulator.psqt_accumulation[perspective][i] += psqt_values_[index];
            }
          }

          for (IndexType k = 0; k < kNumRegs; ++k)
            vec_store(&accTile[k], acc[k]);
        }
      }
#if defined(USE_MMX)
      _mm_empty();
#endif

#else
      for (Color perspective : { WHITE, BLACK }) {

        if (reset[perspective]) {
          if (i == 0) {
            std::memcpy(accumulator.accumulation[perspective][i], biases_,
                        kHalfDimensions * sizeof(BiasType));
          } else {
            std::memset(accumulator.accumulation[perspective][i], 0,
                        kHalfDimensions * sizeof(BiasType));
          }

          accumulator.psqt_accumulation[perspective][i] = 0;

        } else {
          std::memcpy(accumulator.accumulation[perspective][i],
                      prev_accumulator.accumulation[perspective][i],
                      kHalfDimensions * sizeof(BiasType));

          accumulator.psqt_accumulation[perspective][i] = prev_accumulator.psqt_accumulation[perspective][i];

          // Difference calculation for the deactivated features
          for (const auto index : removed_indices[perspective]) {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              accumulator.accumulation[perspective][i][j] -= weights_[offset + j];

            accumulator.psqt_accumulation[perspective][i] -= psqt_values_[index];
          }
        }
        { // Difference calculation for the activated features
          for (const auto index : added_indices[perspective]) {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              accumulator.accumulation[perspective][i][j] += weights_[offset + j];

            accumulator.psqt_accumulation[perspective][i] += psqt_values_[index];
          }
        }
      }
#endif
      }
      accumulator.computed_accumulation = true;
    }

    using BiasType = std::int16_t;
    using WeightType = std::int16_t;
    using PSQTValueType = std::int32_t;

    // Make the learning class a friend
    friend class Trainer<FeatureTransformer>;

    alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
    alignas(kCacheLineSize)
        WeightType weights_[kHalfDimensions * kInputDimensions];
    alignas(kCacheLineSize) std::int32_t psqt_values_[kInputDimensions];
  };

}  // namespace Eval::NNUE

#endif // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
