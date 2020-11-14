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

#ifndef NNUE_LAYERS_SCALE_FACTOR_INTERPOLATE_H_INCLUDED
#define NNUE_LAYERS_SCALE_FACTOR_INTERPOLATE_H_INCLUDED

#include "nnue/nnue_common.h"

#include <string>
#include <cstdint>
#include <type_traits>

namespace Eval::NNUE::Layers {

    template <typename PreviousLayer>
    class ScaleFactorInterpolate {
    public:
        // Input/output type
        using InputType = typename PreviousLayer::OutputType;

        using OutputType = InputType;

        // Number of input/output dimensions
        static constexpr IndexType kInputDimensions =
            PreviousLayer::kOutputDimensions;

        static_assert(kInputDimensions % 2 == 0);

        static constexpr IndexType kOutputDimensions = kInputDimensions / 2;

        // Size of forward propagation buffer used in this layer
        static constexpr std::size_t kSelfBufferSize =
            ceil_to_multiple(kOutputDimensions * sizeof(OutputType), kCacheLineSize);

        // Size of the forward propagation buffer used from the input layer to this layer
        static constexpr std::size_t kBufferSize =
            PreviousLayer::kBufferSize + kSelfBufferSize;

        static constexpr int kLayerIndex = PreviousLayer::kLayerIndex + 1;

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t get_hash_value() {
            std::uint32_t hash_value = 0x6F8AB22Cu;
            hash_value += PreviousLayer::get_hash_value();
            return hash_value;
        }

        static std::string get_name() {
            return "ScaleFactorInterpolate[" + std::to_string(kInputDimensions) +
                "->" + std::to_string(kOutputDimensions) + "]";
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
        bool read_parameters(std::istream& stream) {
            return previous_layer_.read_parameters(stream);
        }

        // write parameters
        bool write_parameters(std::ostream& stream) const {
            return previous_layer_.write_parameters(stream);
        }

        // Forward propagation
        const OutputType* propagate(
            int64_t scale_factor, int64_t phase,
            const TransformedFeatureType* transformed_features, char* buffer) const {

            const auto input = reinterpret_cast<const InputType*>(previous_layer_.propagate(
                scale_factor, phase,
                transformed_features, buffer + kSelfBufferSize));
            const auto output = reinterpret_cast<OutputType*>(buffer);

            for (IndexType i = 0; i < kOutputDimensions; ++i)
            {
                int64_t mg = input[2 * i];
                int64_t eg = input[2 * i + 1];

                int64_t v =
                     mg * phase
                   + eg * (PHASE_MIDGAME - phase) * scale_factor / SCALE_FACTOR_NORMAL;
                v /= PHASE_MIDGAME;
                output[i] = v;
            }

            return output;
        }

    private:
        // Make the learning class a friend
        friend class Trainer<ScaleFactorInterpolate>;

        PreviousLayer previous_layer_;
    };

}  // namespace Eval::NNUE::Layers

#endif // NNUE_LAYERS_SCALE_FACTOR_INTERPOLATE_H_INCLUDED
