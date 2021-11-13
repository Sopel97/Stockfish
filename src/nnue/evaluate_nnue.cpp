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

// Code for calculating NNUE evaluation function

#include <iostream>
#include <set>

#include "../evaluate.h"
#include "../position.h"
#include "../misc.h"
#include "../uci.h"
#include "../types.h"

#include "evaluate_nnue.h"

namespace Stockfish::Eval::NNUE {

  // Input feature converter
  LargePagePtr<FeatureTransformer> feature_transformer;

  // Evaluation function
  AlignedPtr<Network> network[kLayerStacks];

  // Evaluation function file name
  std::string fileName;

  namespace Detail {

  // Initialize the evaluation function parameters
  template <typename T>
  void Initialize(AlignedPtr<T>& pointer) {

    pointer.reset(reinterpret_cast<T*>(std_aligned_alloc(alignof(T), sizeof(T))));
    std::memset(pointer.get(), 0, sizeof(T));
  }

  template <typename T>
  void Initialize(LargePagePtr<T>& pointer) {

    static_assert(alignof(T) <= 4096, "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");
    pointer.reset(reinterpret_cast<T*>(aligned_large_pages_alloc(sizeof(T))));
    std::memset(pointer.get(), 0, sizeof(T));
  }

  // Read evaluation function parameters
  template <typename T>
  bool ReadParameters(std::istream& stream, T& reference) {

    std::uint32_t header;
    header = read_little_endian<std::uint32_t>(stream);
    if (!stream || header != T::GetHashValue()) {
      std::cerr << "Invalid header hash value " << header << ". Expected " << T::GetHashValue() << '\n';
      return false;
    }
    return reference.ReadParameters(stream);
  }

  }  // namespace Detail

  // Initialize the evaluation function parameters
  void Initialize() {

    Detail::Initialize(feature_transformer);
    for (std::size_t i = 0; i < kLayerStacks; ++i)
      Detail::Initialize(network[i]);
  }

  // Read network header
  bool ReadHeader(std::istream& stream, std::uint32_t* hash_value, std::string* architecture)
  {
    std::uint32_t version, size;

    version     = read_little_endian<std::uint32_t>(stream);
    *hash_value = read_little_endian<std::uint32_t>(stream);
    size        = read_little_endian<std::uint32_t>(stream);
    if (!stream || version != kVersion) {
      std::cerr << "Invalid version " << version << ". Expected " << kVersion << '\n';
      return false;
    }
    architecture->resize(size);
    stream.read(&(*architecture)[0], size);
    return !stream.fail();
  }

  // Read network parameters
  bool ReadParameters(std::istream& stream) {

    std::uint32_t hash_value;
    std::string architecture;
    if (!ReadHeader(stream, &hash_value, &architecture)) {
      std::cerr << "Invalid header\n";
      return false;
    }
    if (hash_value != kHashValue) {
      std::cerr << "Invalid network hash value " << hash_value << ". Expected " << kHashValue << '\n';
      return false;
    }
    if (!Detail::ReadParameters(stream, *feature_transformer)) {
      std::cerr << "Failed reading feature transformer parameters\n";
      return false;
    }
    for (std::size_t i = 0; i < kLayerStacks; ++i) {
      if (!Detail::ReadParameters(stream, *(network[i]))) {
        std::cerr << "Failed reading network parameters\n";
        return false;
      }
    }
    if (!(stream && stream.peek() == std::ios::traits_type::eof())) {
      std::cerr << "Expected end of stream.\n";
      return false;
    }
    return true;
  }

  // Evaluation function. Perform differential calculation.
  Value evaluate(const Position& pos) {

    // We manually align the arrays on the stack because with gcc < 9.3
    // overaligning stack variables with alignas() doesn't work correctly.

    constexpr uint64_t alignment = kCacheLineSize;

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
    TransformedFeatureType transformed_features_unaligned[
      FeatureTransformer::kBufferSize + alignment / sizeof(TransformedFeatureType)];
    char buffer_unaligned[Network::kBufferSize + alignment];

    auto* transformed_features = align_ptr_up<alignment>(&transformed_features_unaligned[0]);
    auto* buffer = align_ptr_up<alignment>(&buffer_unaligned[0]);
#else
    alignas(alignment)
      TransformedFeatureType transformed_features[FeatureTransformer::kBufferSize];
    alignas(alignment) char buffer[Network::kBufferSize];
#endif

    ASSERT_ALIGNED(transformed_features, alignment);
    ASSERT_ALIGNED(buffer, alignment);

    constexpr int idx_by_pc[33] = {
        -1, -1,
        0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6,
        7, 7
    };
    const std::size_t bucket = idx_by_pc[popcount(pos.pieces())];
    std::int32_t psqt = 0;
    feature_transformer->Transform(pos, transformed_features, psqt, bucket);
    const auto output = network[bucket]->Propagate(transformed_features, buffer);

    return static_cast<Value>((output[0] + psqt) / FV_SCALE);
  }

  // Load eval, from a file stream or a memory stream
  bool load_eval(std::string name, std::istream& stream) {

    Initialize();
    fileName = name;
    return ReadParameters(stream);
  }

} // namespace Stockfish::Eval::NNUE
