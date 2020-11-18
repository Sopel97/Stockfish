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

#ifndef NNUE_NETWORK_SET_H_INCLUDED
#define NNUE_NETWORK_SET_H_INCLUDED

#include "nnue_common.h"

#include "misc.h"
#include "types.h"

#include <memory>
#include <tuple>
#include <type_traits>
#include <string>
#include <iostream>
#include <string>
#include <fstream>
#include <set>
#include <cstring>
#include <utility>

#ifndef TRAINED_NET_ID
#define TRAINED_NET_ID 0
#endif

namespace Eval::NNUE {

    template <typename T>
    using AlignedPtr = std::unique_ptr<T, AlignedDeleter<T>>;

    template <typename T>
    using LargePagePtr = std::unique_ptr<T, LargePageDeleter<T>>;

    namespace Detail {

        template<typename T>
        constexpr T vmax(T&&t)
        {
          return std::forward<T>(t);
        }

        template<typename T0, typename T1, typename... Ts>
        constexpr typename std::common_type<T0, T1, Ts...>::type vmax(T0&& val1, T1&& val2, Ts&&... vs)
        {
            if (val2 > val1)
                return vmax(val2, std::forward<Ts>(vs)...);
            else
                return vmax(val1, std::forward<Ts>(vs)...);
        }

        // Initialize the evaluation function parameters
        template <typename T>
        void initialize(AlignedPtr<T>& pointer) {

            pointer.reset(reinterpret_cast<T*>(std_aligned_alloc(alignof(T), sizeof(T))));
            std::memset(pointer.get(), 0, sizeof(T));
        }

        template <typename T>
        void initialize(LargePagePtr<T>& pointer) {

            static_assert(alignof(T) <= 4096, "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");

            pointer.reset(reinterpret_cast<T*>(aligned_large_pages_alloc(sizeof(T))));
            std::memset(pointer.get(), 0, sizeof(T));
        }

        // Read evaluation function parameters
        template <typename T>
        bool read_parameters(std::istream& stream, T& reference) {

            std::uint32_t header;
            header = read_little_endian<std::uint32_t>(stream);

            if (!stream || header != T::get_hash_value())
                return false;

            return reference.read_parameters(stream);
        }

        // write evaluation function parameters
        template <typename T>
        bool write_parameters(std::ostream& stream, const AlignedPtr<T>& pointer) {
            constexpr std::uint32_t header = T::get_hash_value();

            stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

            return pointer->write_parameters(stream);
        }

        template <typename T>
        bool write_parameters(std::ostream& stream, const LargePagePtr<T>& pointer) {
            constexpr std::uint32_t header = T::get_hash_value();

            stream.write(reinterpret_cast<const char*>(&header), sizeof(header));

            return pointer->write_parameters(stream);
        }
    }  // namespace Detail

    template <typename... NetworksTs>
    struct NetworkSet
    {
        template <int I>
        using NetworkTypeById = typename std::tuple_element<I, std::tuple<NetworksTs...>>::type;

        using FirstNet = NetworkTypeById<0>;

        using OutputType = typename FirstNet::OutputType;

        static constexpr int kOutputDimensions = FirstNet::kOutputDimensions;

        static constexpr int kBufferSize = Detail::vmax(NetworksTs::kBufferSize...);

        static_assert((std::is_same_v<OutputType, typename NetworksTs::OutputType> && ...));
        static_assert(((kOutputDimensions == NetworksTs::kOutputDimensions) && ...));

        // Hash value embedded in the evaluation file
        static constexpr std::uint32_t get_hash_value() {
            return ((NetworksTs::get_hash_value() ^ 0x55555555) ^ ...);
        }

        static std::string get_structure_string()
        {
            return std::string("NetworkSet<") + ((NetworksTs::get_structure_string() + ",") + ...) + ">";
        }

        static std::string get_layers_info()
        {
            return ((NetworksTs::get_layers_info() + "\n\n") + ...);
        }

        // Initialize the evaluation function parameters
        void initialize()
        {
            (Detail::initialize(std::get<AlignedPtr<NetworksTs>>(networks)), ...);
        }

        int read_parameters(std::istream& stream)
        {
            return (Detail::read_parameters(stream, *std::get<AlignedPtr<NetworksTs>>(networks)) && ...);
        }

        int write_parameters(std::ostream& ostream)
        {
            return (Detail::write_parameters(ostream, std::get<AlignedPtr<NetworksTs>>(networks)) && ...);
        }

        template <int I>
        Value evaluate(const TransformedFeatureType* transformed_features) const
        {
            using NetworkType = NetworkTypeById<I>;
            alignas(kCacheLineSize) char buffer[NetworkType::kBufferSize];
            return evaluate<I>(buffer, transformed_features);
        }

        template <int I>
        const auto& get() const
        {
            auto& ptr = std::get<I>(networks);
            assert(ptr);
            return *ptr;
        }

        template <int I>
        auto& get()
        {
            auto& ptr = std::get<I>(networks);
            assert(ptr);
            return *ptr;
        }

        Value evaluate(
            const TransformedFeatureType* transformed_features,
            int i) const {

            alignas(kCacheLineSize) char buffer[kBufferSize];
            return evaluate_selected_dynamic<0>(buffer, transformed_features, i);
        }

    private:
        std::tuple<AlignedPtr<NetworksTs>...> networks;

        template <int I>
        Value evaluate_selected_dynamic(
            char* buffer,
            const TransformedFeatureType* transformed_features,
            int i) const {

            if constexpr (I >= sizeof...(NetworksTs))
            {
                return Value(0);
            }
            else
            {
                if (i == I)
                {
                    return evaluate<I>(buffer, transformed_features);
                }
                else
                {
                    return evaluate_selected_dynamic<I+1>(buffer, transformed_features, i);
                }
            }
        }

        template <int I>
        Value evaluate(char* buffer, const TransformedFeatureType* transformed_features) const
        {
            const auto& network = std::get<I>(networks);
            const auto output = network->propagate(transformed_features, buffer);
            return static_cast<Value>(output[0] / FV_SCALE);
        }
    };

}

#endif
