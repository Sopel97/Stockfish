/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <utility>
#include <unordered_map>
#include <optional>

#include "../position.h"
#include "../types.h"
#include "nnue_accumulator.h"
#include "nnue_architecture.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE {

using BiasType       = std::int16_t;
using WeightType     = std::int16_t;
using PSQTWeightType = std::int32_t;

// If vector instructions are enabled, we update and refresh the
// accumulator tile by tile such that each tile fits in the CPU's
// vector registers.
#define VECTOR

static_assert(PSQTBuckets % 8 == 0,
              "Per feature PSQT values cannot be processed at granularity lower than 8 at a time.");

#ifdef USE_AVX512
using vec_t      = __m512i;
using psqt_vec_t = __m256i;
    #define vec_load(a) _mm512_load_si512(a)
    #define vec_store(a, b) _mm512_store_si512(a, b)
    #define vec_add_16(a, b) _mm512_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm512_sub_epi16(a, b)
    #define vec_mul_16(a, b) _mm512_mullo_epi16(a, b)
    #define vec_zero() _mm512_setzero_epi32()
    #define vec_set_16(a) _mm512_set1_epi16(a)
    #define vec_max_16(a, b) _mm512_max_epi16(a, b)
    #define vec_min_16(a, b) _mm512_min_epi16(a, b)
    // Inverse permuted at load time
    #define vec_msb_pack_16(a, b) \
        _mm512_packs_epi16(_mm512_srli_epi16(a, 7), _mm512_srli_epi16(b, 7))
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()
    #define NumRegistersSIMD 16
    #define MaxChunkSize 64

#elif USE_AVX2
using vec_t      = __m256i;
using psqt_vec_t = __m256i;
    #define vec_load(a) _mm256_load_si256(a)
    #define vec_store(a, b) _mm256_store_si256(a, b)
    #define vec_add_16(a, b) _mm256_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm256_sub_epi16(a, b)
    #define vec_mul_16(a, b) _mm256_mullo_epi16(a, b)
    #define vec_zero() _mm256_setzero_si256()
    #define vec_set_16(a) _mm256_set1_epi16(a)
    #define vec_max_16(a, b) _mm256_max_epi16(a, b)
    #define vec_min_16(a, b) _mm256_min_epi16(a, b)
    // Inverse permuted at load time
    #define vec_msb_pack_16(a, b) \
        _mm256_packs_epi16(_mm256_srli_epi16(a, 7), _mm256_srli_epi16(b, 7))
    #define vec_load_psqt(a) _mm256_load_si256(a)
    #define vec_store_psqt(a, b) _mm256_store_si256(a, b)
    #define vec_add_psqt_32(a, b) _mm256_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm256_sub_epi32(a, b)
    #define vec_zero_psqt() _mm256_setzero_si256()
    #define NumRegistersSIMD 16
    #define MaxChunkSize 32

#elif USE_SSE2
using vec_t      = __m128i;
using psqt_vec_t = __m128i;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) _mm_add_epi16(a, b)
    #define vec_sub_16(a, b) _mm_sub_epi16(a, b)
    #define vec_mul_16(a, b) _mm_mullo_epi16(a, b)
    #define vec_zero() _mm_setzero_si128()
    #define vec_set_16(a) _mm_set1_epi16(a)
    #define vec_max_16(a, b) _mm_max_epi16(a, b)
    #define vec_min_16(a, b) _mm_min_epi16(a, b)
    #define vec_msb_pack_16(a, b) _mm_packs_epi16(_mm_srli_epi16(a, 7), _mm_srli_epi16(b, 7))
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) _mm_add_epi32(a, b)
    #define vec_sub_psqt_32(a, b) _mm_sub_epi32(a, b)
    #define vec_zero_psqt() _mm_setzero_si128()
    #define NumRegistersSIMD (Is64Bit ? 16 : 8)
    #define MaxChunkSize 16

#elif USE_NEON
using vec_t      = int16x8_t;
using psqt_vec_t = int32x4_t;
    #define vec_load(a) (*(a))
    #define vec_store(a, b) *(a) = (b)
    #define vec_add_16(a, b) vaddq_s16(a, b)
    #define vec_sub_16(a, b) vsubq_s16(a, b)
    #define vec_mul_16(a, b) vmulq_s16(a, b)
    #define vec_zero() \
        vec_t { 0 }
    #define vec_set_16(a) vdupq_n_s16(a)
    #define vec_max_16(a, b) vmaxq_s16(a, b)
    #define vec_min_16(a, b) vminq_s16(a, b)
inline vec_t vec_msb_pack_16(vec_t a, vec_t b) {
    const int8x8_t  shifta    = vshrn_n_s16(a, 7);
    const int8x8_t  shiftb    = vshrn_n_s16(b, 7);
    const int8x16_t compacted = vcombine_s8(shifta, shiftb);
    return *reinterpret_cast<const vec_t*>(&compacted);
}
    #define vec_load_psqt(a) (*(a))
    #define vec_store_psqt(a, b) *(a) = (b)
    #define vec_add_psqt_32(a, b) vaddq_s32(a, b)
    #define vec_sub_psqt_32(a, b) vsubq_s32(a, b)
    #define vec_zero_psqt() \
        psqt_vec_t { 0 }
    #define NumRegistersSIMD 16
    #define MaxChunkSize 16

#else
    #undef VECTOR

#endif


#ifdef VECTOR

    // Compute optimal SIMD register count for feature transformer accumulation.

    // We use __m* types as template arguments, which causes GCC to emit warnings
    // about losing some attribute information. This is irrelevant to us as we
    // only take their size, so the following pragma are harmless.
    #if defined(__GNUC__)
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wignored-attributes"
    #endif

template<typename SIMDRegisterType, typename LaneType, int NumLanes, int MaxRegisters>
static constexpr int BestRegisterCount() {
    #define RegisterSize sizeof(SIMDRegisterType)
    #define LaneSize sizeof(LaneType)

    static_assert(RegisterSize >= LaneSize);
    static_assert(MaxRegisters <= NumRegistersSIMD);
    static_assert(MaxRegisters > 0);
    static_assert(NumRegistersSIMD > 0);
    static_assert(RegisterSize % LaneSize == 0);
    static_assert((NumLanes * LaneSize) % RegisterSize == 0);

    const int ideal = (NumLanes * LaneSize) / RegisterSize;
    if (ideal <= MaxRegisters)
        return ideal;

    // Look for the largest divisor of the ideal register count that is smaller than MaxRegisters
    for (int divisor = MaxRegisters; divisor > 1; --divisor)
        if (ideal % divisor == 0)
            return divisor;

    return 1;
}
    #if defined(__GNUC__)
        #pragma GCC diagnostic pop
    #endif
#endif

struct MoveKeyTypeHashFunc {
    uint64_t operator()(FeatureSet::MoveKeyType v) const noexcept { return splitmix64_hash(v); }
};

static constexpr FeatureSet::MoveKeyType MoveKeyTombstone    = ~uint64_t(0);
static constexpr size_t                  HashCapacity        = 1 << 16;
static constexpr size_t                  HashMaxSearchLength = 3;

class FeatureTransformerWeightCachePreanalyzer {
   public:
    FeatureTransformerWeightCachePreanalyzer() :
        dpHistogram(MoveKeyTombstone, HashCapacity, HashMaxSearchLength) {}

    void after_do_move(const Position& pos, Move move) {
        if (move.type_of() == CASTLING || type_of(pos.piece_on(move.to_sq())) == KING)
            return;

        const FeatureSet::MoveKeyType keys[COLOR_NB] = {
          FeatureSet::make_move_key<WHITE>(pos.square<KING>(WHITE), pos.state()->dirtyPiece),
          FeatureSet::make_move_key<BLACK>(pos.square<KING>(BLACK), pos.state()->dirtyPiece)};

        auto w = dpHistogram.find_or_emplace(keys[WHITE], 0);
        auto b = dpHistogram.find_or_emplace(keys[BLACK], 0);
        if (w)
            w->second += 1;
        if (b)
            b->second += 1;
    }

    std::vector<FeatureSet::MoveKeyType> get_top(size_t count) const {
        std::vector<FeatureSet::MoveKeyType> res;

        if (dpHistogram.size() <= count)
        {
            res = dpHistogram.get_populated_keys();
        }
        else
        {
            auto       v   = dpHistogram.get_populated();
            const auto nth = v.begin() + (count + 1);
            std::nth_element(v.begin(), nth, v.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.second > rhs.second;
            });
            res.reserve(count);
            for (size_t i = 0; i < count; ++i)
                res.emplace_back(v[i].first);
        }

        assert(res.size() <= count);

        return res;
    }

    void print() const {
        auto v = dpHistogram.get_populated();
        std::sort(v.begin(), v.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
        std::cout << v.size() << '\n';
        for (int i = 0; i < 32 && i < v.size(); ++i)
        {
            auto el = v[i];
            std::cout << el.first << ": " << el.second << '\n';
        }

        int total_moves = 0;
        for (int i = 0; i < v.size(); ++i)
        {
            auto el = v[i];
            total_moves += el.second;
        }

        int top_moves = 0;
        for (int i = 0; i < v.size(); ++i)
        {
            auto el = v[i];
            top_moves += el.second;
            double pct = double(top_moves) / total_moves * 100.0;
            if ((i & (i - 1)) == 0)
            {
                std::cout << i << ": " << pct << "%\n";
            }
        }
    }

   private:
    UnreliableInsertionHashTable<FeatureSet::MoveKeyType, uint64_t, MoveKeyTypeHashFunc>
      dpHistogram;
};

struct FeatureWeightPtrs {
    const WeightType*     baseWeights;
    const PSQTWeightType* basePsqtWeights;

    template<IndexType TransformedFeatureDimensions>
    FeatureWeightPtrs offset(IndexType i) const {
        return {baseWeights + i * TransformedFeatureDimensions, basePsqtWeights + i * PSQTBuckets};
    }
};

// Cache is separate as different searches require a different cache, but the network is shared.
template<IndexType TransformedFeatureDimensions>
class FeatureTransformerWeightCache {
   public:
    using PreanalyzerType = FeatureTransformerWeightCachePreanalyzer;

    template<typename Network>
    FeatureTransformerWeightCache(const FeatureTransformerWeightCachePreanalyzer& preanalyzer,
                                  const Network&                                  net,
                                  size_t                                          numEntries) :
        keyToWeightsIndex(MoveKeyTombstone, HashCapacity, HashMaxSearchLength),
        numEntries(numEntries) {
        // Overallocate for alignment.
        weightsBuffer     = std::make_unique<WeightType[]>(numEntries * TransformedFeatureDimensions
                                                       + CacheLineSize / sizeof(WeightType));
        psqtWeightsBuffer = std::make_unique<PSQTWeightType[]>(
          numEntries * PSQTBuckets + CacheLineSize / sizeof(PSQTWeightType));

        weights     = align_ptr_up<CacheLineSize>(weightsBuffer.get());
        psqtWeights = align_ptr_up<CacheLineSize>(psqtWeightsBuffer.get());

        assert(weights - weightsBuffer.get() < CacheLineSize);
        assert(psqtWeights - psqtWeightsBuffer.get() < CacheLineSize);

        IndexType index = 0;
        for (FeatureSet::MoveKeyType key : preanalyzer.get_top(numEntries))
        {
            auto v = keyToWeightsIndex.find_or_emplace(key, index);
            // Insertion may fail, though it shouldn't because we're using the same size
            if (v)
            {
                assert(v->second == index);
                fill_weights_for_feature(index, key, net);
                index += 1;
            }
        }

        assert(index <= numEntries);
    }

    std::optional<FeatureWeightPtrs> find(FeatureSet::MoveKeyType key) const {
        const auto it = keyToWeightsIndex.find(key);
        if (it == nullptr)
            return std::nullopt;

        const IndexType idx = it->second;
        assert(idx < numEntries);
        return FeatureWeightPtrs{weights, psqtWeights}.offset<TransformedFeatureDimensions>(idx);
    }

   private:
    UnreliableInsertionHashTable<FeatureSet::MoveKeyType, IndexType, MoveKeyTypeHashFunc>
      keyToWeightsIndex;

    std::unique_ptr<WeightType[]>     weightsBuffer;
    std::unique_ptr<PSQTWeightType[]> psqtWeightsBuffer;
    WeightType*                       weights;
    PSQTWeightType*                   psqtWeights;
    size_t numEntries;

    template<typename Network>
    void
    fill_weights_for_feature(IndexType index, FeatureSet::MoveKeyType key, const Network& net) {
        static_assert(TransformedFeatureDimensions
                      == Network::FeatureTransformerType::HalfDimensions);

        const auto& ft = net.get_feature_transformer();

        FeatureSet::IndexList removed, added;
        FeatureSet::decode_move_key(key, removed, added);

        // Difference calculation for the deactivated features
        const IndexType offset     = index * TransformedFeatureDimensions;
        const IndexType psqtOffset = index * PSQTBuckets;

        std::fill(weights + offset, weights + offset + TransformedFeatureDimensions, 0);
        std::fill(psqtWeights + psqtOffset, psqtWeights + psqtOffset + PSQTBuckets, 0);

        for (const auto idx : added)
        {
            assert(idx < FeatureSet::Dimensions);
            ft.add_feature_weights(idx, weights + offset, psqtWeights + psqtOffset);
        }

        for (const auto idx : removed)
        {
            assert(idx < FeatureSet::Dimensions);
            ft.sub_feature_weights(idx, weights + offset, psqtWeights + psqtOffset);
        }
    }
};

class TranslatedFeatureUpdateList {
   public:
    template<Color Perspective, IndexType TransformedFeatureDimensions>
    void append_changed_indices(
      Square                                                             ksq,
      const DirtyPiece&                                                  dp,
      FeatureWeightPtrs                                                  baseWeights,
      const FeatureTransformerWeightCache<TransformedFeatureDimensions>* cache) {
        FeatureSet::IndexList r, a;

        if (cache)
        {
            const FeatureSet::MoveKeyType key = FeatureSet::make_move_key<Perspective>(ksq, dp);
            auto                          e   = cache->find(key);
            if (false && e.has_value())
            {
                added.push_back(*e);
                // Early return, don't add them normally
                return;
            }
            else
            {
                FeatureSet::decode_move_key(key, r, a);
            }
        }
        else
        {
            FeatureSet::append_changed_indices<Perspective>(ksq, dp, r, a);
        }

        for (const auto& i : r)
        {
            assert(i < FeatureSet::Dimensions);
            removed.push_back(baseWeights.offset<TransformedFeatureDimensions>(i));
        }
        for (const auto& i : a)
        {
            assert(i < FeatureSet::Dimensions);
            added.push_back(baseWeights.offset<TransformedFeatureDimensions>(i));
        }
    }

    ValueList<FeatureWeightPtrs, FeatureSet::MaxActiveDimensions> removed;
    ValueList<FeatureWeightPtrs, FeatureSet::MaxActiveDimensions> added;

    static_assert(std::is_trivial_v<FeatureWeightPtrs>);
};

// Input feature converter
template<IndexType                                 TransformedFeatureDimensions,
         Accumulator<TransformedFeatureDimensions> StateInfo::*accPtr>
class FeatureTransformer {

   public:
    // Number of output dimensions for one side
    static constexpr IndexType HalfDimensions = TransformedFeatureDimensions;

   private:
#ifdef VECTOR
    static constexpr int NumRegs =
      BestRegisterCount<vec_t, WeightType, TransformedFeatureDimensions, NumRegistersSIMD>();
    static constexpr int NumPsqtRegs =
      BestRegisterCount<psqt_vec_t, PSQTWeightType, PSQTBuckets, NumRegistersSIMD>();

    static constexpr IndexType TileHeight     = NumRegs * sizeof(vec_t) / 2;
    static constexpr IndexType PsqtTileHeight = NumPsqtRegs * sizeof(psqt_vec_t) / 4;
    static_assert(HalfDimensions % TileHeight == 0, "TileHeight must divide HalfDimensions");
    static_assert(PSQTBuckets % PsqtTileHeight == 0, "PsqtTileHeight must divide PSQTBuckets");
#endif

   public:
    // Output type
    using OutputType = TransformedFeatureType;

    // Number of input/output dimensions
    static constexpr IndexType InputDimensions  = FeatureSet::Dimensions;
    static constexpr IndexType OutputDimensions = HalfDimensions;

    // Size of forward propagation buffer
    static constexpr std::size_t BufferSize = OutputDimensions * sizeof(OutputType);

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t get_hash_value() {
        return FeatureSet::HashValue ^ (OutputDimensions * 2);
    }

    static constexpr void order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // _mm512_packs_epi16 ordering
        uint64_t tmp0, tmp1;
        tmp0 = v[2], tmp1 = v[3];
        v[2] = v[8], v[3] = v[9];
        v[8] = v[4], v[9] = v[5];
        v[4] = tmp0, v[5] = tmp1;
        tmp0 = v[6], tmp1 = v[7];
        v[6] = v[10], v[7] = v[11];
        v[10] = v[12], v[11] = v[13];
        v[12] = tmp0, v[13] = tmp1;
#elif defined(USE_AVX2)  // _mm256_packs_epi16 ordering
        std::swap(v[2], v[4]);
        std::swap(v[3], v[5]);
#endif
    }

    static constexpr void inverse_order_packs([[maybe_unused]] uint64_t* v) {
#if defined(USE_AVX512)  // Inverse _mm512_packs_epi16 ordering
        uint64_t tmp0, tmp1;
        tmp0 = v[2], tmp1 = v[3];
        v[2] = v[4], v[3] = v[5];
        v[4] = v[8], v[5] = v[9];
        v[8] = tmp0, v[9] = tmp1;
        tmp0 = v[6], tmp1 = v[7];
        v[6] = v[12], v[7] = v[13];
        v[12] = v[10], v[13] = v[11];
        v[10] = tmp0, v[11] = tmp1;
#elif defined(USE_AVX2)  // Inverse _mm256_packs_epi16 ordering
        std::swap(v[2], v[4]);
        std::swap(v[3], v[5]);
#endif
    }

    void permute_weights([[maybe_unused]] void (*order_fn)(uint64_t*)) const {
#if defined(USE_AVX2)
    #if defined(USE_AVX512)
        constexpr IndexType di = 16;
    #else
        constexpr IndexType di = 8;
    #endif
        uint64_t* b = reinterpret_cast<uint64_t*>(const_cast<BiasType*>(&biases[0]));
        for (IndexType i = 0; i < HalfDimensions * sizeof(BiasType) / sizeof(uint64_t); i += di)
            order_fn(&b[i]);

        for (IndexType j = 0; j < InputDimensions; ++j)
        {
            uint64_t* w =
              reinterpret_cast<uint64_t*>(const_cast<WeightType*>(&weights[j * HalfDimensions]));
            for (IndexType i = 0; i < HalfDimensions * sizeof(WeightType) / sizeof(uint64_t);
                 i += di)
                order_fn(&w[i]);
        }
#endif
    }

    // Read network parameters
    bool read_parameters(std::istream& stream) {

        read_leb_128<BiasType>(stream, biases, HalfDimensions);
        read_leb_128<WeightType>(stream, weights, HalfDimensions * InputDimensions);
        read_leb_128<PSQTWeightType>(stream, psqtWeights, PSQTBuckets * InputDimensions);

        permute_weights(inverse_order_packs);
        return !stream.fail();
    }

    // Write network parameters
    bool write_parameters(std::ostream& stream) const {

        permute_weights(order_packs);

        write_leb_128<BiasType>(stream, biases, HalfDimensions);
        write_leb_128<WeightType>(stream, weights, HalfDimensions * InputDimensions);
        write_leb_128<PSQTWeightType>(stream, psqtWeights, PSQTBuckets * InputDimensions);

        permute_weights(inverse_order_packs);
        return !stream.fail();
    }

    // Convert input features
    std::int32_t
    transform(const Position&                                      pos,
              OutputType*                                          output,
              int                                                  bucket,
              bool                                                 psqtOnly,
              const FeatureTransformerWeightCache<HalfDimensions>* cache = nullptr) const {
        update_accumulator<WHITE>(pos, psqtOnly, cache);
        update_accumulator<BLACK>(pos, psqtOnly, cache);

        const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& psqtAccumulation = (pos.state()->*accPtr).psqtAccumulation;
        const auto  psqt =
          (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket])
          / 2;

        if (psqtOnly)
            return psqt;

        const auto& accumulation = (pos.state()->*accPtr).accumulation;

        for (IndexType p = 0; p < 2; ++p)
        {
            const IndexType offset = (HalfDimensions / 2) * p;

#if defined(VECTOR)

            constexpr IndexType OutputChunkSize = MaxChunkSize;
            static_assert((HalfDimensions / 2) % OutputChunkSize == 0);
            constexpr IndexType NumOutputChunks = HalfDimensions / 2 / OutputChunkSize;

            const vec_t Zero = vec_zero();
            const vec_t One  = vec_set_16(127);

            const vec_t* in0 = reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][0]));
            const vec_t* in1 =
              reinterpret_cast<const vec_t*>(&(accumulation[perspectives[p]][HalfDimensions / 2]));
            vec_t* out = reinterpret_cast<vec_t*>(output + offset);

            for (IndexType j = 0; j < NumOutputChunks; ++j)
            {
                const vec_t sum0a = vec_max_16(vec_min_16(in0[j * 2 + 0], One), Zero);
                const vec_t sum0b = vec_max_16(vec_min_16(in0[j * 2 + 1], One), Zero);
                const vec_t sum1a = vec_max_16(vec_min_16(in1[j * 2 + 0], One), Zero);
                const vec_t sum1b = vec_max_16(vec_min_16(in1[j * 2 + 1], One), Zero);

                const vec_t pa = vec_mul_16(sum0a, sum1a);
                const vec_t pb = vec_mul_16(sum0b, sum1b);

                out[j] = vec_msb_pack_16(pa, pb);
            }

#else

            for (IndexType j = 0; j < HalfDimensions / 2; ++j)
            {
                BiasType sum0 = accumulation[static_cast<int>(perspectives[p])][j + 0];
                BiasType sum1 =
                  accumulation[static_cast<int>(perspectives[p])][j + HalfDimensions / 2];
                sum0               = std::clamp<BiasType>(sum0, 0, 127);
                sum1               = std::clamp<BiasType>(sum1, 0, 127);
                output[offset + j] = static_cast<OutputType>(unsigned(sum0 * sum1) / 128);
            }

#endif
        }

        return psqt;
    }  // end of function transform()

    void
    hint_common_access(const Position&                                      pos,
                       bool                                                 psqtOnly,
                       const FeatureTransformerWeightCache<HalfDimensions>* cache = nullptr) const {
        hint_common_access_for_perspective<WHITE>(pos, psqtOnly, cache);
        hint_common_access_for_perspective<BLACK>(pos, psqtOnly, cache);
    }

    void add_feature_weights(IndexType index, WeightType* w, PSQTWeightType* pw) const {
        const IndexType offset     = index * HalfDimensions;
        const IndexType psqtOffset = index * PSQTBuckets;

        for (IndexType i = 0; i < HalfDimensions; ++i)
            w[i] += weights[offset + i];

        for (IndexType i = 0; i < PSQTBuckets; ++i)
            pw[i] += psqtWeights[psqtOffset + i];
    }

    void sub_feature_weights(IndexType index, WeightType* w, PSQTWeightType* pw) const {
        const IndexType offset     = index * HalfDimensions;
        const IndexType psqtOffset = index * PSQTBuckets;

        for (IndexType i = 0; i < HalfDimensions; ++i)
            w[i] -= weights[offset + i];

        for (IndexType i = 0; i < PSQTBuckets; ++i)
            pw[i] -= psqtWeights[psqtOffset + i];
    }

   private:
    template<Color Perspective>
    [[nodiscard]] std::pair<StateInfo*, StateInfo*>
    try_find_computed_accumulator(const Position& pos, bool psqtOnly) const {
        // Look for a usable accumulator of an earlier position. We keep track
        // of the estimated gain in terms of features to be added/subtracted.
        StateInfo *st = pos.state(), *next = nullptr;
        int        gain = FeatureSet::refresh_cost(pos);
        while (st->previous
               && (!(st->*accPtr).computedPSQT[Perspective]
                   || (!psqtOnly && !(st->*accPtr).computed[Perspective])))
        {
            // This governs when a full feature refresh is needed and how many
            // updates are better than just one full refresh.
            if (FeatureSet::requires_refresh(st, Perspective)
                || (gain -= FeatureSet::update_cost(st) + 1) < 0)
                break;
            next = st;
            st   = st->previous;
        }
        return {st, next};
    }

    // NOTE: The parameter states_to_update is an array of position states, ending with nullptr.
    //       All states must be sequential, that is states_to_update[i] must either be reachable
    //       by repeatedly applying ->previous from states_to_update[i+1] or
    //       states_to_update[i] == nullptr.
    //       computed_st must be reachable by repeatedly applying ->previous on
    //       states_to_update[0], if not nullptr.
    template<Color Perspective, size_t N>
    void update_accumulator_incremental(
      const Position&                                      pos,
      StateInfo*                                           computed_st,
      StateInfo*                                           states_to_update[N],
      bool                                                 psqtOnly,
      const FeatureTransformerWeightCache<HalfDimensions>* cache = nullptr) const {
        static_assert(N > 0);
        assert(states_to_update[N - 1] == nullptr);

#ifdef VECTOR
        // Gcc-10.2 unnecessarily spills AVX2 registers if this array
        // is defined in the VECTOR code below, once in each branch
        vec_t      acc[NumRegs];
        psqt_vec_t psqt[NumPsqtRegs];
#endif

        if (states_to_update[0] == nullptr)
            return;

        // Update incrementally going back through states_to_update.

        // Gather all features to be updated.
        const Square ksq = pos.square<KING>(Perspective);

        // The size must be enough to contain the largest possible update.
        // That might depend on the feature set and generally relies on the
        // feature set's update cost calculation to be correct and never allow
        // updates with more added/removed features than MaxActiveDimensions.
        TranslatedFeatureUpdateList featureChanges[N - 1];

        {
            int i =
              N
              - 2;  // Last potential state to update. Skip last element because it must be nullptr.
            while (states_to_update[i] == nullptr)
                --i;

            StateInfo* st2 = states_to_update[i];

            for (; i >= 0; --i)
            {
                (states_to_update[i]->*accPtr).computed[Perspective]     = !psqtOnly;
                (states_to_update[i]->*accPtr).computedPSQT[Perspective] = true;

                const StateInfo* end_state = i == 0 ? computed_st : states_to_update[i - 1];

                for (; st2 != end_state; st2 = st2->previous)
                    featureChanges[i].template append_changed_indices<Perspective>(
                      ksq, st2->dirtyPiece, FeatureWeightPtrs{weights, psqtWeights}, cache);
            }
        }

        StateInfo* st = computed_st;

        // Now update the accumulators listed in states_to_update[], where the last element is a sentinel.
#ifdef VECTOR

        if (states_to_update[1] == nullptr && featureChanges[0].removed.size() == 0
            && featureChanges[0].added.size() == 1)
        {
            assert(states_to_update[0]);

            if (!psqtOnly)
            {
                auto accIn =
                  reinterpret_cast<const vec_t*>(&(st->*accPtr).accumulation[Perspective][0]);
                auto accOut = reinterpret_cast<vec_t*>(
                  &(states_to_update[0]->*accPtr).accumulation[Perspective][0]);

                auto columnA =
                  reinterpret_cast<const vec_t*>(featureChanges[0].added[0].baseWeights);

                for (IndexType k = 0; k < HalfDimensions * sizeof(std::int16_t) / sizeof(vec_t);
                     ++k)
                    accOut[k] = vec_add_16(accIn[k], columnA[k]);
            }

            auto accPsqtIn =
              reinterpret_cast<const psqt_vec_t*>(&(st->*accPtr).psqtAccumulation[Perspective][0]);
            auto accPsqtOut = reinterpret_cast<psqt_vec_t*>(
              &(states_to_update[0]->*accPtr).psqtAccumulation[Perspective][0]);

            auto columnPsqtA =
              reinterpret_cast<const psqt_vec_t*>(featureChanges[0].added[0].basePsqtWeights);

            for (std::size_t k = 0; k < PSQTBuckets * sizeof(std::int32_t) / sizeof(psqt_vec_t);
                 ++k)
                accPsqtOut[k] = vec_add_psqt_32(accPsqtIn[k], columnPsqtA[k]);
        }
        else
        {
            if (!psqtOnly)
                for (IndexType j = 0; j < HalfDimensions / TileHeight; ++j)
                {
                    // Load accumulator
                    auto accTileIn = reinterpret_cast<const vec_t*>(
                      &(st->*accPtr).accumulation[Perspective][j * TileHeight]);
                    for (IndexType k = 0; k < NumRegs; ++k)
                        acc[k] = vec_load(&accTileIn[k]);

                    for (IndexType i = 0; states_to_update[i]; ++i)
                    {
                        // Difference calculation for the deactivated features
                        for (const auto [columnBase, _] : featureChanges[i].removed)
                        {
                            auto column =
                              reinterpret_cast<const vec_t*>(columnBase + j * TileHeight);
                            for (IndexType k = 0; k < NumRegs; ++k)
                                acc[k] = vec_sub_16(acc[k], column[k]);
                        }

                        // Difference calculation for the activated features
                        for (const auto [columnBase, _] : featureChanges[i].added)
                        {
                            auto column =
                              reinterpret_cast<const vec_t*>(columnBase + j * TileHeight);
                            for (IndexType k = 0; k < NumRegs; ++k)
                                acc[k] = vec_add_16(acc[k], column[k]);
                        }

                        // Store accumulator
                        auto accTileOut =
                          reinterpret_cast<vec_t*>(&(states_to_update[i]->*accPtr)
                                                      .accumulation[Perspective][j * TileHeight]);
                        for (IndexType k = 0; k < NumRegs; ++k)
                            vec_store(&accTileOut[k], acc[k]);
                    }
                }

            for (IndexType j = 0; j < PSQTBuckets / PsqtTileHeight; ++j)
            {
                // Load accumulator
                auto accTilePsqtIn = reinterpret_cast<const psqt_vec_t*>(
                  &(st->*accPtr).psqtAccumulation[Perspective][j * PsqtTileHeight]);
                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] = vec_load_psqt(&accTilePsqtIn[k]);

                for (IndexType i = 0; states_to_update[i]; ++i)
                {
                    // Difference calculation for the deactivated features
                    for (const auto [_, columnBase] : featureChanges[i].removed)
                    {
                        auto columnPsqt =
                          reinterpret_cast<const psqt_vec_t*>(columnBase + j * PsqtTileHeight);
                        for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                            psqt[k] = vec_sub_psqt_32(psqt[k], columnPsqt[k]);
                    }

                    // Difference calculation for the activated features
                    for (const auto [_, columnBase] : featureChanges[i].added)
                    {
                        auto columnPsqt =
                          reinterpret_cast<const psqt_vec_t*>(columnBase + j * PsqtTileHeight);
                        for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                            psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
                    }

                    // Store accumulator
                    auto accTilePsqtOut = reinterpret_cast<psqt_vec_t*>(
                      &(states_to_update[i]->*accPtr)
                         .psqtAccumulation[Perspective][j * PsqtTileHeight]);
                    for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                        vec_store_psqt(&accTilePsqtOut[k], psqt[k]);
                }
            }
        }
#else
        for (IndexType i = 0; states_to_update[i]; ++i)
        {
            if (!psqtOnly)
                std::memcpy((states_to_update[i]->*accPtr).accumulation[Perspective],
                            (st->*accPtr).accumulation[Perspective],
                            HalfDimensions * sizeof(BiasType));

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                (states_to_update[i]->*accPtr).psqtAccumulation[Perspective][k] =
                  (st->*accPtr).psqtAccumulation[Perspective][k];

            st = states_to_update[i];

            // Difference calculation for the deactivated features
            for (const auto [columnBase, columnPsqtBase] : featureChanges[i].removed)
            {
                if (!psqtOnly)
                {
                    for (IndexType j = 0; j < HalfDimensions; ++j)
                        (st->*accPtr).accumulation[Perspective][j] -= columnBase[j];
                }

                for (std::size_t k = 0; k < PSQTBuckets; ++k)
                    (st->*accPtr).psqtAccumulation[Perspective][k] -= columnPsqtBase[k];
            }

            // Difference calculation for the activated features
            for (const auto [columnBase, columnPsqtBase] : featureChanges[i].added)
            {
                if (!psqtOnly)
                {
                    for (IndexType j = 0; j < HalfDimensions; ++j)
                        (st->*accPtr).accumulation[Perspective][j] += columnBase[j];
                }

                for (std::size_t k = 0; k < PSQTBuckets; ++k)
                    (st->*accPtr).psqtAccumulation[Perspective][k] += columnPsqtBase[k];
            }
        }
#endif
    }

    template<Color Perspective>
    void update_accumulator_refresh(const Position& pos, bool psqtOnly) const {
#ifdef VECTOR
        // Gcc-10.2 unnecessarily spills AVX2 registers if this array
        // is defined in the VECTOR code below, once in each branch
        vec_t      acc[NumRegs];
        psqt_vec_t psqt[NumPsqtRegs];
#endif

        // Refresh the accumulator
        // Could be extracted to a separate function because it's done in 2 places,
        // but it's unclear if compilers would correctly handle register allocation.
        auto& accumulator                     = pos.state()->*accPtr;
        accumulator.computed[Perspective]     = !psqtOnly;
        accumulator.computedPSQT[Perspective] = true;
        FeatureSet::IndexList active;
        FeatureSet::append_active_indices<Perspective>(pos, active);

#ifdef VECTOR
        if (!psqtOnly)
            for (IndexType j = 0; j < HalfDimensions / TileHeight; ++j)
            {
                auto biasesTile = reinterpret_cast<const vec_t*>(&biases[j * TileHeight]);
                for (IndexType k = 0; k < NumRegs; ++k)
                    acc[k] = biasesTile[k];

                int i = 0;
                for (; i < int(active.size()) - 1; i += 2)
                {
                    IndexType       index0  = active[i];
                    IndexType       index1  = active[i + 1];
                    const IndexType offset0 = HalfDimensions * index0 + j * TileHeight;
                    const IndexType offset1 = HalfDimensions * index1 + j * TileHeight;
                    auto            column0 = reinterpret_cast<const vec_t*>(&weights[offset0]);
                    auto            column1 = reinterpret_cast<const vec_t*>(&weights[offset1]);

                    for (unsigned k = 0; k < NumRegs; ++k)
                        acc[k] = vec_add_16(acc[k], vec_add_16(column0[k], column1[k]));
                }
                for (; i < int(active.size()); ++i)
                {
                    IndexType       index  = active[i];
                    const IndexType offset = HalfDimensions * index + j * TileHeight;
                    auto            column = reinterpret_cast<const vec_t*>(&weights[offset]);

                    for (unsigned k = 0; k < NumRegs; ++k)
                        acc[k] = vec_add_16(acc[k], column[k]);
                }

                auto accTile =
                  reinterpret_cast<vec_t*>(&accumulator.accumulation[Perspective][j * TileHeight]);
                for (unsigned k = 0; k < NumRegs; k++)
                    vec_store(&accTile[k], acc[k]);
            }

        for (IndexType j = 0; j < PSQTBuckets / PsqtTileHeight; ++j)
        {
            for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                psqt[k] = vec_zero_psqt();

            int i = 0;
            for (; i < int(active.size()) - 1; i += 2)
            {
                IndexType       index0  = active[i];
                IndexType       index1  = active[i + 1];
                const IndexType offset0 = PSQTBuckets * index0 + j * PsqtTileHeight;
                const IndexType offset1 = PSQTBuckets * index1 + j * PsqtTileHeight;
                auto columnPsqt0 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset0]);
                auto columnPsqt1 = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset1]);

                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] =
                      vec_add_psqt_32(psqt[k], vec_add_psqt_32(columnPsqt0[k], columnPsqt1[k]));
            }
            for (; i < int(active.size()); ++i)
            {
                IndexType       index  = active[i];
                const IndexType offset = PSQTBuckets * index + j * PsqtTileHeight;
                auto columnPsqt        = reinterpret_cast<const psqt_vec_t*>(&psqtWeights[offset]);

                for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                    psqt[k] = vec_add_psqt_32(psqt[k], columnPsqt[k]);
            }

            auto accTilePsqt = reinterpret_cast<psqt_vec_t*>(
              &accumulator.psqtAccumulation[Perspective][j * PsqtTileHeight]);
            for (std::size_t k = 0; k < NumPsqtRegs; ++k)
                vec_store_psqt(&accTilePsqt[k], psqt[k]);
        }

#else
        if (!psqtOnly)
            std::memcpy(accumulator.accumulation[Perspective], biases,
                        HalfDimensions * sizeof(BiasType));

        for (std::size_t k = 0; k < PSQTBuckets; ++k)
            accumulator.psqtAccumulation[Perspective][k] = 0;

        for (const auto index : active)
        {
            if (!psqtOnly)
            {
                const IndexType offset = HalfDimensions * index;
                for (IndexType j = 0; j < HalfDimensions; ++j)
                    accumulator.accumulation[Perspective][j] += weights[offset + j];
            }

            for (std::size_t k = 0; k < PSQTBuckets; ++k)
                accumulator.psqtAccumulation[Perspective][k] +=
                  psqtWeights[index * PSQTBuckets + k];
        }
#endif
    }

    template<Color Perspective>
    void hint_common_access_for_perspective(
      const Position&                                      pos,
      bool                                                 psqtOnly,
      const FeatureTransformerWeightCache<HalfDimensions>* cache = nullptr) const {

        // Works like update_accumulator, but performs less work.
        // Updates ONLY the accumulator for pos.

        // Look for a usable accumulator of an earlier position. We keep track
        // of the estimated gain in terms of features to be added/subtracted.
        // Fast early exit.
        if ((pos.state()->*accPtr).computed[Perspective]
            || (psqtOnly && (pos.state()->*accPtr).computedPSQT[Perspective]))
            return;

        auto [oldest_st, _] = try_find_computed_accumulator<Perspective>(pos, psqtOnly);

        if ((oldest_st->*accPtr).computed[Perspective]
            || (psqtOnly && (oldest_st->*accPtr).computedPSQT[Perspective]))
        {
            // Only update current position accumulator to minimize work.
            StateInfo* states_to_update[2] = {pos.state(), nullptr};
            update_accumulator_incremental<Perspective, 2>(pos, oldest_st, states_to_update,
                                                           psqtOnly, cache);
        }
        else
            update_accumulator_refresh<Perspective>(pos, psqtOnly);
    }

    template<Color Perspective>
    void
    update_accumulator(const Position&                                      pos,
                       bool                                                 psqtOnly,
                       const FeatureTransformerWeightCache<HalfDimensions>* cache = nullptr) const {

        auto [oldest_st, next] = try_find_computed_accumulator<Perspective>(pos, psqtOnly);

        if ((oldest_st->*accPtr).computed[Perspective]
            || (psqtOnly && (oldest_st->*accPtr).computedPSQT[Perspective]))
        {
            if (next == nullptr)
                return;

            // Now update the accumulators listed in states_to_update[], where the last element is a sentinel.
            // Currently we update 2 accumulators.
            //     1. for the current position
            //     2. the next accumulator after the computed one
            // The heuristic may change in the future.
            StateInfo* states_to_update[3] = {next, next == pos.state() ? nullptr : pos.state(),
                                              nullptr};

            update_accumulator_incremental<Perspective, 3>(pos, oldest_st, states_to_update,
                                                           psqtOnly, cache);
        }
        else
            update_accumulator_refresh<Perspective>(pos, psqtOnly);
    }

    alignas(CacheLineSize) BiasType biases[HalfDimensions];
    alignas(CacheLineSize) WeightType weights[HalfDimensions * InputDimensions];
    alignas(CacheLineSize) PSQTWeightType psqtWeights[InputDimensions * PSQTBuckets];
};

}  // namespace Stockfish::Eval::NNUE

#endif  // #ifndef NNUE_FEATURE_TRANSFORMER_H_INCLUDED
