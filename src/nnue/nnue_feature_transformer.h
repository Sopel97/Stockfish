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

// A class that converts the input features of the NNUE evaluation function

#ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
#define NNUE_FEATURE_TRANSFORMER_H_INCLUDED

#include "nnue_common.h"
#include "nnue_architecture.h"
#include "features/index_list.h"

#include <cstring> // std::memset()

namespace Stockfish::Eval::NNUE {

  // If vector instructions are enabled, we update and refresh the
  // accumulator tile by tile such that each tile fits in the CPU's
  // vector registers.
  #define VECTOR

  static_assert(kPSQTBuckets == 8);

  #ifdef USE_AVX512
  typedef __m512i vec_t;
  typedef __m256i psqt_vec_t;
  #define vec_load(a) _mm512_load_si512(a)
  #define vec_store(a,b) _mm512_store_si512(a,b)
  #define vec_add_16(a,b) _mm512_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm512_sub_epi16(a,b)
  #define vec_load_psqt(a) _mm256_load_si256(a)
  #define vec_store_psqt(a,b) _mm256_store_si256(a,b)
  #define vec_add_psqt_32(a,b) _mm256_add_epi32(a,b)
  #define vec_sub_psqt_32(a,b) _mm256_sub_epi32(a,b)
  #define vec_zero_psqt() _mm256_setzero_si256()
  static constexpr IndexType kNumRegs = 8; // only 8 are needed
  static constexpr IndexType kPsqtRegs = 1; // only 8 are needed

  #elif USE_AVX2
  typedef __m256i vec_t;
  typedef __m256i psqt_vec_t;
  #define vec_load(a) _mm256_load_si256(a)
  #define vec_store(a,b) _mm256_store_si256(a,b)
  #define vec_add_16(a,b) _mm256_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm256_sub_epi16(a,b)
  #define vec_load_psqt(a) _mm256_load_si256(a)
  #define vec_store_psqt(a,b) _mm256_store_si256(a,b)
  #define vec_add_psqt_32(a,b) _mm256_add_epi32(a,b)
  #define vec_sub_psqt_32(a,b) _mm256_sub_epi32(a,b)
  #define vec_zero_psqt() _mm256_setzero_si256()
  static constexpr IndexType kNumRegs = 16;
  static constexpr IndexType kPsqtRegs = 1; // only 8 are needed

  #elif USE_SSE2
  typedef __m128i vec_t;
  typedef __m128i psqt_vec_t;
  #define vec_load(a) (*(a))
  #define vec_store(a,b) *(a)=(b)
  #define vec_add_16(a,b) _mm_add_epi16(a,b)
  #define vec_sub_16(a,b) _mm_sub_epi16(a,b)
  #define vec_load_psqt(a) (*(a))
  #define vec_store_psqt(a,b) *(a)=(b)
  #define vec_add_psqt_32(a,b) _mm_add_epi32(a,b)
  #define vec_sub_psqt_32(a,b) _mm_sub_epi32(a,b)
  #define vec_zero_psqt() _mm_setzero_si128()
  static constexpr IndexType kNumRegs = Is64Bit ? 16 : 8;
  static constexpr IndexType kPsqtRegs = 2; // only 8 are needed

  #elif USE_MMX
  typedef __m64 vec_t;
  typedef std::int32_t psqt_vec_t;
  #define vec_load(a) (*(a))
  #define vec_store(a,b) *(a)=(b)
  #define vec_add_16(a,b) _mm_add_pi16(a,b)
  #define vec_sub_16(a,b) _mm_sub_pi16(a,b)
  #define vec_load_psqt(a) (*(a))
  #define vec_store_psqt(a,b) *(a)=(b)
  #define vec_add_psqt_32(a,b) a+b
  #define vec_sub_psqt_32(a,b) a-b
  #define vec_zero_psqt() 0
  static constexpr IndexType kNumRegs = 8;
  static constexpr IndexType kPsqtRegs = 4; // only 8 are needed

  #elif USE_NEON
  typedef int16x8_t vec_t;
  typedef int32x4_t psqt_vec_t;
  #define vec_load(a) (*(a))
  #define vec_store(a,b) *(a)=(b)
  #define vec_add_16(a,b) vaddq_s16(a,b)
  #define vec_sub_16(a,b) vsubq_s16(a,b)
  #define vec_load_psqt(a) (*(a))
  #define vec_store_psqt(a,b) *(a)=(b)
  #define vec_add_psqt_32(a,b) vaddq_s32(a,b)
  #define vec_sub_psqt_32(a,b) vsubq_s32(a,b)
  #define vec_zero_psqt() psqt_vec_t{0}
  static constexpr IndexType kNumRegs = 16;
  static constexpr IndexType kPsqtRegs = 2; // only 8 are needed

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
    static constexpr IndexType kPsqtTileHeight = kPsqtRegs * sizeof(psqt_vec_t) / 4;
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

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {

      return RawFeatures::kHashValue ^ kOutputDimensions;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {

      for (std::size_t i = 0; i < kHalfDimensions; ++i)
        biases_[i] = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < kHalfDimensions * kInputDimensions; ++i)
        weights_[i] = read_little_endian<WeightType>(stream);
      for (std::size_t i = 0; i < kInputDimensions; ++i)
        for (std::size_t j = 0; j < kPSQTBuckets; ++j)
          psqt_weights_[i*kPSQTBuckets + j] = read_little_endian<PSQTWeightType>(stream);
      return !stream.fail();
    }

    // Convert input features
    void Transform(const Position& pos, OutputType* output, std::int32_t& psqt, int bucket) const {

      UpdateAccumulator(pos, WHITE);
      UpdateAccumulator(pos, BLACK);

      const auto& accumulation = pos.state()->accumulator.accumulation;
      const auto& psqt_accumulation = pos.state()->accumulator.psqt_accumulation;

      const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
      for (IndexType p = 0; p < 2; ++p) {
        const IndexType offset = kHalfDimensions * p;

        for (IndexType j = 0; j < kHalfDimensions; ++j) {
          BiasType sum = accumulation[static_cast<int>(perspectives[p])][0][j];
          std::int16_t v = std::min(std::abs(sum), 127) - 127;
          std::int16_t vv = v * v >> 8;
          if (sum > 0)
              vv = 126 - vv;
          output[offset + j] = static_cast<OutputType>(vv);
        }
      }

      psqt = 0;
      psqt += psqt_accumulation[static_cast<int>(perspectives[0])][0][bucket];
      psqt -= psqt_accumulation[static_cast<int>(perspectives[1])][0][bucket];
      psqt /= 2;

  #if defined(USE_MMX)
      _mm_empty();
  #endif
    }

   private:
    void UpdateAccumulator(const Position& pos, const Color c) const {

  #ifdef VECTOR
      // Gcc-10.2 unnecessarily spills AVX2 registers if this array
      // is defined in the VECTOR code below, once in each branch
      vec_t acc[kNumRegs];
  #endif

      // Look for a usable accumulator of an earlier position. We keep track
      // of the estimated gain in terms of features to be added/subtracted.
      StateInfo *st = pos.state(), *next = nullptr;
      int gain = pos.count<ALL_PIECES>() - 2;
      while (st->accumulator.state[c] == EMPTY)
      {
        auto& dp = st->dirtyPiece;
        // The first condition tests whether an incremental update is
        // possible at all: if this side's king has moved, it is not possible.
        static_assert(std::is_same_v<RawFeatures::SortedTriggerSet,
              Features::CompileTimeList<Features::TriggerEvent, Features::TriggerEvent::kFriendKingMoved>>,
              "Current code assumes that only kFriendlyKingMoved refresh trigger is being used.");
        if (   dp.piece[0] == make_piece(c, KING)
            || (gain -= dp.dirty_num + 1) < 0)
          break;
        next = st;
        st = st->previous;
      }

      if (st->accumulator.state[c] == COMPUTED)
      {
        if (next == nullptr)
          return;

        // Update incrementally in two steps. First, we update the "next"
        // accumulator. Then, we update the current accumulator (pos.state()).

        // Gather all features to be updated. This code assumes HalfKA features
        // only and doesn't support refresh triggers.
        static_assert(std::is_same_v<Features::FeatureSet<Features::HalfKA<Features::Side::kFriend>>,
                                     RawFeatures>);
        Features::IndexList removed[2], added[2];
        Features::HalfKA<Features::Side::kFriend>::AppendChangedIndices(pos,
            next->dirtyPiece, c, &removed[0], &added[0]);
        for (StateInfo *st2 = pos.state(); st2 != next; st2 = st2->previous)
          Features::HalfKA<Features::Side::kFriend>::AppendChangedIndices(pos,
              st2->dirtyPiece, c, &removed[1], &added[1]);

        // Mark the accumulators as computed.
        next->accumulator.state[c] = COMPUTED;
        pos.state()->accumulator.state[c] = COMPUTED;

        // Now update the accumulators listed in info[], where the last element is a sentinel.
        StateInfo *info[3] =
          { next, next == pos.state() ? nullptr : pos.state(), nullptr };
  #ifdef VECTOR
        for (IndexType j = 0; j < kHalfDimensions / kTileHeight; ++j)
        {
          // Load accumulator
          auto accTile = reinterpret_cast<vec_t*>(
            &st->accumulator.accumulation[c][0][j * kTileHeight]);
          for (IndexType k = 0; k < kNumRegs; ++k)
            acc[k] = vec_load(&accTile[k]);

          for (IndexType i = 0; info[i]; ++i)
          {
            // Difference calculation for the deactivated features
            for (const auto index : removed[i])
            {
              const IndexType offset = kHalfDimensions * index + j * kTileHeight;
              auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_sub_16(acc[k], column[k]);
            }

            // Difference calculation for the activated features
            for (const auto index : added[i])
            {
              const IndexType offset = kHalfDimensions * index + j * kTileHeight;
              auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);
              for (IndexType k = 0; k < kNumRegs; ++k)
                acc[k] = vec_add_16(acc[k], column[k]);
            }

            // Store accumulator
            accTile = reinterpret_cast<vec_t*>(
              &info[i]->accumulator.accumulation[c][0][j * kTileHeight]);
            for (IndexType k = 0; k < kNumRegs; ++k)
              vec_store(&accTile[k], acc[k]);
          }
        }

        {
          psqt_vec_t psqt[kPsqtRegs];
          for (IndexType j = 0; j < kPSQTBuckets / kPsqtTileHeight; ++j)
          {
            // Load accumulator
            auto accTilePsqt = reinterpret_cast<psqt_vec_t*>(
              &st->accumulator.psqt_accumulation[c][0][j * kPsqtTileHeight]);
            for (std::size_t k = 0; k < kPsqtRegs; ++k)
              psqt[k] = vec_load_psqt(&accTilePsqt[k]);

            for (IndexType i = 0; info[i]; ++i)
            {
              // Difference calculation for the deactivated features
              for (const auto index : removed[i])
              {
                auto column_psqt = reinterpret_cast<const psqt_vec_t*>(&psqt_weights_[index*kPSQTBuckets + j * kPsqtTileHeight]);
                for (std::size_t k = 0; k < kPsqtRegs; ++k)
                  psqt[k] = vec_sub_psqt_32(psqt[k], column_psqt[k]);
              }

              // Difference calculation for the activated features
              for (const auto index : added[i])
              {
                auto column_psqt = reinterpret_cast<const psqt_vec_t*>(&psqt_weights_[index*kPSQTBuckets + j * kPsqtTileHeight]);
                for (std::size_t k = 0; k < kPsqtRegs; ++k)
                  psqt[k] = vec_add_psqt_32(psqt[k], column_psqt[k]);
              }

              // Store accumulator
              accTilePsqt = reinterpret_cast<psqt_vec_t*>(
                &info[i]->accumulator.psqt_accumulation[c][0][j * kPsqtTileHeight]);
              for (std::size_t k = 0; k < kPsqtRegs; ++k)
                vec_store_psqt(&accTilePsqt[k], psqt[k]);
            }
          }
        }
  #else
        for (IndexType i = 0; info[i]; ++i)
        {
          std::memcpy(info[i]->accumulator.accumulation[c][0],
              st->accumulator.accumulation[c][0],
              kHalfDimensions * sizeof(BiasType));

          for (std::size_t k = 0; k < kPSQTBuckets; ++k)
            info[i]->accumulator.psqt_accumulation[c][0][k] = st->accumulator.psqt_accumulation[c][0][k];

          st = info[i];

          // Difference calculation for the deactivated features
          for (const auto index : removed[i])
          {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              st->accumulator.accumulation[c][0][j] -= weights_[offset + j];

            for (std::size_t k = 0; k < kPSQTBuckets; ++k)
              st->accumulator.psqt_accumulation[c][0][k] -= psqt_weights_[index*kPSQTBuckets+k];
          }

          // Difference calculation for the activated features
          for (const auto index : added[i])
          {
            const IndexType offset = kHalfDimensions * index;

            for (IndexType j = 0; j < kHalfDimensions; ++j)
              st->accumulator.accumulation[c][0][j] += weights_[offset + j];

            for (std::size_t k = 0; k < kPSQTBuckets; ++k)
              st->accumulator.psqt_accumulation[c][0][k] += psqt_weights_[index*kPSQTBuckets+k];
          }
        }
  #endif
      }
      else
      {
        // Refresh the accumulator
        auto& accumulator = pos.state()->accumulator;
        accumulator.state[c] = COMPUTED;
        Features::IndexList active;
        Features::HalfKA<Features::Side::kFriend>::AppendActiveIndices(pos, c, &active);

  #ifdef VECTOR
        for (IndexType j = 0; j < kHalfDimensions / kTileHeight; ++j)
        {
          auto biasesTile = reinterpret_cast<const vec_t*>(
              &biases_[j * kTileHeight]);
          for (IndexType k = 0; k < kNumRegs; ++k)
            acc[k] = biasesTile[k];

          for (const auto index : active)
          {
            const IndexType offset = kHalfDimensions * index + j * kTileHeight;
            auto column = reinterpret_cast<const vec_t*>(&weights_[offset]);

            for (unsigned k = 0; k < kNumRegs; ++k)
              acc[k] = vec_add_16(acc[k], column[k]);
          }

          auto accTile = reinterpret_cast<vec_t*>(
              &accumulator.accumulation[c][0][j * kTileHeight]);
          for (unsigned k = 0; k < kNumRegs; k++)
            vec_store(&accTile[k], acc[k]);
        }

        {
          psqt_vec_t psqt[kPsqtRegs];
          for (IndexType j = 0; j < kPSQTBuckets / kPsqtTileHeight; ++j)
          {
            for (std::size_t k = 0; k < kPsqtRegs; ++k)
              psqt[k] = vec_zero_psqt();

            for (const auto index : active)
            {
              auto column_psqt = reinterpret_cast<const psqt_vec_t*>(&psqt_weights_[index*kPSQTBuckets + j * kPsqtTileHeight]);

              for (std::size_t k = 0; k < kPsqtRegs; ++k)
                psqt[k] = vec_add_psqt_32(psqt[k], column_psqt[k]);
            }

            auto accTilePsqt = reinterpret_cast<psqt_vec_t*>(
              &accumulator.psqt_accumulation[c][0][j * kPsqtTileHeight]);
            for (std::size_t k = 0; k < kPsqtRegs; ++k)
              vec_store_psqt(&accTilePsqt[k], psqt[k]);
          }
        }
  #else
        std::memcpy(accumulator.accumulation[c][0], biases_,
            kHalfDimensions * sizeof(BiasType));

        for (std::size_t k = 0; k < kPSQTBuckets; ++k)
          accumulator.psqt_accumulation[c][0][k] = 0;

        for (const auto index : active)
        {
          const IndexType offset = kHalfDimensions * index;

          for (IndexType j = 0; j < kHalfDimensions; ++j)
            accumulator.accumulation[c][0][j] += weights_[offset + j];

          for (std::size_t k = 0; k < kPSQTBuckets; ++k)
            accumulator.psqt_accumulation[c][0][k] += psqt_weights_[index*kPSQTBuckets+k];
        }
  #endif
      }

  #if defined(USE_MMX)
      _mm_empty();
  #endif
    }

    using BiasType = std::int16_t;
    using WeightType = std::int16_t;
    using PSQTWeightType = std::int32_t;

    alignas(kCacheLineSize) BiasType biases_[kHalfDimensions];
    alignas(kCacheLineSize)
        WeightType weights_[kHalfDimensions * kInputDimensions];
    alignas(kCacheLineSize) std::int32_t psqt_weights_[kInputDimensions * kPSQTBuckets];
  };

}  // namespace Stockfish::Eval::NNUE

#endif // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
