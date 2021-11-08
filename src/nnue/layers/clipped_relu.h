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
  template <typename PreviousLayer>
  class ClippedReLU {
   public:
    // Input/output type
    using InputType = typename PreviousLayer::OutputType;
    using OutputType = std::uint8_t;
    static_assert(std::is_same<InputType, std::int32_t>::value, "");

    // Number of input/output dimensions
    static constexpr IndexType kInputDimensions =
        PreviousLayer::kOutputDimensions;
    static constexpr IndexType kOutputDimensions = kInputDimensions;

    // Size of forward propagation buffer used in this layer
    static constexpr std::size_t kSelfBufferSize =
        CeilToMultiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

    // Size of the forward propagation buffer used from the input layer to this layer
    static constexpr std::size_t kBufferSize =
        PreviousLayer::kBufferSize + kSelfBufferSize;

    // Hash value embedded in the evaluation file
    static constexpr std::uint32_t GetHashValue() {
      std::uint32_t hash_value = 0x538D24C7u;
      hash_value += PreviousLayer::GetHashValue();
      return hash_value;
    }

    // Read network parameters
    bool ReadParameters(std::istream& stream) {
      return previous_layer_.ReadParameters(stream);
    }

    // Forward propagation
    const OutputType* Propagate(
        const TransformedFeatureType* transformed_features, char* buffer) const {
      const auto input = previous_layer_.Propagate(
          transformed_features, buffer + kSelfBufferSize);
      const auto output = reinterpret_cast<OutputType*>(buffer);

      for (IndexType i = 0; i < kInputDimensions; ++i) {
        constexpr int prec = 4;
        std::int16_t v = std::min(std::abs(input[i]) >> (kWeightScaleBits - prec), 127 << prec) - (127 << prec);
        std::int16_t vv = v * v >> (8 + prec * 2);
        if (input[i] > 0)
            vv = 126 - vv;
        output[i] = static_cast<OutputType>(vv);
      }
      return output;
    }

   private:
    PreviousLayer previous_layer_;
  };

}  // namespace Stockfish::Eval::NNUE::Layers

#endif // NNUE_LAYERS_CLIPPED_RELU_H_INCLUDED
