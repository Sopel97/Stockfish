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

// Definition of layer AffineTransform of NNUE evaluation function

#ifndef NNUE_LAYERS_INNER_PRODUCT_H_INCLUDED
#define NNUE_LAYERS_INNER_PRODUCT_H_INCLUDED

#include <iostream>
#include "../nnue_common.h"

#include <string>
#include <type_traits>
#include <cstdint>

namespace Eval::NNUE::Layers {

  // Affine transformation layer
  template <typename PreviousLayer>
  class InnerProduct {
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = std::int32_t;
    static_assert(std::is_same<InputType, std::int16_t>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions =
        PreviousLayer::kOutputDimensions;
    static constexpr IndexType kOutputDimensions = 1;
    static constexpr IndexType kPaddedInputDimensions =
        CeilToMultiple<IndexType>(kInputDimensions, kMaxSimdWidth);

    // Size of forward propagation buffer used in this layer
    static constexpr std::size_t kSelfBufferSize =
        CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

    // Size of the forward propagation buffer used from the input layer to this layer
    static constexpr std::size_t kBufferSize =
        PreviousLayer::kBufferSize + kSelfBufferSize;

    static constexpr int kLayerIndex = PreviousLayer::kLayerIndex + 1;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {
      std::uint32_t hash_value = 0xBB03DAE4u;
      hash_value += kOutputDimensions;
      hash_value ^= PreviousLayer::GetHashValue() >> 1;
      hash_value ^= PreviousLayer::GetHashValue() << 31;
      return hash_value;
    }

    static std::string get_name() {
        return "InnerProduct[" +
            std::to_string(kOutputDimensions) + "<-" +
            std::to_string(kInputDimensions) + "]";
    }

    // A string that represents the structure from the input layer to this layer
    static std::string get_structure_string() {
        return get_name() + "(" +
            PreviousLayer::get_structure_string() + ")";
    }

    static std::string get_layers_info() {
        std::string info = PreviousLayer::get_layers_info();
        info += "\n  - ";
        info += std::to_string(kLayerIndex);
        info += " - ";
        info += get_name();
        return info;
    }

   // Read network parameters
    bool ReadParameters(std::istream& stream) {
      if (!previous_layer_.ReadParameters(stream)) return false;
      bias_ = read_little_endian<BiasType>(stream);
      for (std::size_t i = 0; i < kPaddedInputDimensions; ++i)
        weights_[i] = read_little_endian<WeightType>(stream);
      return !stream.fail();
    }

    // write parameters
    bool WriteParameters(std::ostream& stream) const {
        if (!previous_layer_.WriteParameters(stream))
            return false;

        stream.write(reinterpret_cast<const char*>(&bias_),
            sizeof(BiasType));

        stream.write(reinterpret_cast<const char*>(weights_),
            kPaddedInputDimensions *
            sizeof(WeightType));

        return !stream.fail();
    }

    // Forward propagation
    const OutputType* Propagate(
        const TransformedFeatureType* transformed_features, char* buffer) const {
      const auto input = previous_layer_.Propagate(
          transformed_features, buffer + kSelfBufferSize);

      auto output = reinterpret_cast<OutputType*>(buffer);

      OutputType sum = bias_;
      for (IndexType j = 0; j < kInputDimensions; ++j) {
        sum += weights_[j] * input[j];
      }
      *output = sum;

      return output;
    }

   private:
    using BiasType = OutputType;
    using WeightType = std::int8_t;

    // Make the learning class a friend
    friend class Trainer<InnerProduct>;

    PreviousLayer previous_layer_;

    alignas(kCacheLineSize) BiasType bias_;
    alignas(kCacheLineSize)
        WeightType weights_[kPaddedInputDimensions];
  };

}  // namespace Eval::NNUE::Layers

#endif // #ifndef NNUE_LAYERS_AFFINE_TRANSFORM_H_INCLUDED
