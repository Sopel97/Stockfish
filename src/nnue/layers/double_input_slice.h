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

// NNUE evaluation function layer DoubleInputSlice definition

#ifndef NNUE_LAYERS_DOUBLE_INPUT_SLICE_H_INCLUDED
#define NNUE_LAYERS_DOUBLE_INPUT_SLICE_H_INCLUDED

#include "../nnue_common.h"

#include <string>
#include <cstdint>
#include <type_traits>

namespace Eval::NNUE::Layers {

    // Input layer
    template <IndexType HalfOutputDimensions, IndexType Stride, IndexType Offset = 0>
    class DoubleInputSlice {
    public:
        // Need to maintain alignment
        static_assert(Offset % kMaxSimdWidth == 0, "");
        static_assert(Stride % kMaxSimdWidth == 0, "");

        // Output type
        using OutputType = TransformedFeatureType;

        // Output dimensionality
        static constexpr IndexType kHalfOutputDimensions = HalfOutputDimensions;
        static constexpr IndexType kOutputDimensions = HalfOutputDimensions * 2;

        // Size of forward propagation buffer used from the input layer to this layer
        static constexpr std::size_t kBufferSize = kOutputDimensions * sizeof(OutputType);

        static constexpr int kLayerIndex = 1;

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t GetHashValue() {
            std::uint32_t hash_value = 0xDF31E90Du;
            hash_value ^= kOutputDimensions ^ (Offset << 10) ^ (Stride << 4);
            return hash_value;
        }

        static std::string GetName() {
            return "DoubleInputSlice[" + std::to_string(kOutputDimensions) + "((" +
                std::to_string(Offset) + ":" +
                std::to_string(Offset + kHalfOutputDimensions) + ")+(" +
                std::to_string(Offset + Stride) + ":" +
                std::to_string(Offset + Stride + kHalfOutputDimensions) + "))]";
        }

        // A string that represents the structure from the input layer to this layer
        static std::string GetStructureString() {
            return GetName();
        }

        // Read network parameters
        bool ReadParameters(std::istream& /*stream*/) {
            return true;
        }

        // write parameters
        bool WritePparameters(std::ostream& /*stream*/) const {
            return true;
        }

        // Forward propagation
        const OutputType* Propagate(
            const TransformedFeatureType* transformed_features,
            char* buffer) const {

            static_assert(std::is_same_v<OutputType, TransformedFeatureType>);

            std::memcpy(
                buffer,
                transformed_features + Offset,
                sizeof(TransformedFeatureType) * kHalfOutputDimensions);

            std::memcpy(
                buffer + sizeof(TransformedFeatureType) * kHalfOutputDimensions,
                transformed_features + Stride + Offset,
                sizeof(TransformedFeatureType) * kHalfOutputDimensions);

            return reinterpret_cast<OutputType*>(buffer);
        }

    private:
    };

}  // namespace Layers

#endif // #ifndef NNUE_LAYERS_DOUBLE_INPUT_SLICE_H_INCLUDED