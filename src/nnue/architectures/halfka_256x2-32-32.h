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

// Definition of input features and network structure used in NNUE evaluation function

#ifndef NNUE_HALFKA_256X2_32_32_H_INCLUDED
#define NNUE_HALFKA_256X2_32_32_H_INCLUDED

#include "../../misc.h"

#include "../features/feature_set.h"
#include "../features/half_ka.h"

#include "../layers/affine_transform.h"
#include "../layers/clipped_relu.h"

namespace Stockfish::Eval::NNUE {

// Input features used in evaluation function
using RawFeatures = Features::FeatureSet<
    Features::HalfKA<Features::Side::kFriend>>;

// Number of input feature dimensions after conversion
constexpr IndexType kTransformedFeatureDimensions = 1024;
constexpr IndexType kPSQTBuckets = 8;
constexpr IndexType kLayerStacks = 8;

struct Network
{
  static constexpr int FC_0_OUTPUTS = 8;
  static constexpr int FC_1_OUTPUTS = 32;

  Layers::AffineTransform<kTransformedFeatureDimensions * 2, FC_0_OUTPUTS> fc_0;
  Layers::ClippedReLU<FC_0_OUTPUTS> ac_0;
  Layers::AffineTransform<FC_0_OUTPUTS, FC_1_OUTPUTS> fc_1;
  Layers::ClippedReLU<FC_1_OUTPUTS> ac_1;
  Layers::AffineTransform<FC_1_OUTPUTS, 1> fc_2;

  // Hash value embedded in the evaluation file
  static constexpr std::uint32_t GetHashValue() {
    // input slice hash
    std::uint32_t hash_value = 0xEC42E90Du;
    hash_value ^= kTransformedFeatureDimensions * 2;

    hash_value = decltype(fc_0)::GetHashValue(hash_value);
    hash_value = decltype(ac_0)::GetHashValue(hash_value);
    hash_value = decltype(fc_1)::GetHashValue(hash_value);
    hash_value = decltype(ac_1)::GetHashValue(hash_value);
    hash_value = decltype(fc_2)::GetHashValue(hash_value);

    return hash_value;
  }

  // Read network parameters
  bool ReadParameters(std::istream& stream) {
    if (!fc_0.ReadParameters(stream)) return false;
    if (!ac_0.ReadParameters(stream)) return false;
    if (!fc_1.ReadParameters(stream)) return false;
    if (!ac_1.ReadParameters(stream)) return false;
    if (!fc_2.ReadParameters(stream)) return false;
    return true;
  }

  std::int32_t Propagate(const TransformedFeatureType* transformed_features)
  {
    constexpr uint64_t alignment = kCacheLineSize;

    struct Buffer
    {
      alignas(kCacheLineSize) decltype(fc_0)::OutputBuffer fc_0_out;
      alignas(kCacheLineSize) decltype(ac_0)::OutputBuffer ac_0_out;
      alignas(kCacheLineSize) decltype(fc_1)::OutputBuffer fc_1_out;
      alignas(kCacheLineSize) decltype(ac_1)::OutputBuffer ac_1_out;
      alignas(kCacheLineSize) decltype(fc_2)::OutputBuffer fc_2_out;
    };

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
    char buffer_raw[sizeof(Buffer) + alignment];
    char* buffer_raw_aligned = align_ptr_up<alignment>(&buffer_raw[0]);
    Buffer& buffer = *(new (buffer_raw_aligned) Buffer);
#else
    alignas(alignment) Buffer buffer;
#endif

    fc_0.Propagate(transformed_features, buffer.fc_0_out);
    ac_0.Propagate(buffer.fc_0_out, buffer.ac_0_out);
    fc_1.Propagate(buffer.ac_0_out, buffer.fc_1_out);
    ac_1.Propagate(buffer.fc_1_out, buffer.ac_1_out);
    fc_2.Propagate(buffer.ac_1_out, buffer.fc_2_out);

    std::uint32_t output_value = buffer.fc_2_out[0];

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
    buffer.~Buffer();
#endif

    return output_value;
  }
};

}  // namespace Stockfish::Eval::NNUE

#endif // #ifndef NNUE_HALFKA_256X2_32_32_H_INCLUDED
