﻿#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_PAWN_CONV_9_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_PAWN_CONV_9_H_

#include "factorizer.h"

#include "nnue/features/pawn_conv_9.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKA
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKA
    template <Side AssociatedKing>
    class Factorizer<PawnConv9<AssociatedKing>> {
    private:
        using FeatureType = PawnConv9<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesPawnConv9,
            kFeaturesConv9,
            kFeaturesPawn,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesPawnConv9
            {true, FeatureType::kDimensions},
            // kFeaturesConv9
            {true, FeatureType::kConvStates},
            // kFeaturesPawn
            {true, SQUARE_NB * 2 * 24},
        };

        static_assert(get_array_length(kProperties) == kNumTrainingFeatureTypes, "");

    public:
        static constexpr std::string get_name() {
            return std::string("Factorizer<") + FeatureType::kName + "> -> " + "Conv9, Pawn";
        }

        static constexpr std::string get_factorizers_string() {
            return "  - " + get_name();
        }

        // Get the dimensionality of the learning feature
        static constexpr IndexType get_dimensions() {
            return get_active_dimensions(kProperties);
        }

        // Get index of learning feature and scale of learning rate
        static void append_training_features(
            IndexType base_index, std::vector<TrainingFeature>* training_features) {

            // kFeaturesHalfA
            IndexType index_offset = append_base_feature<FeatureType>(
                kProperties[kFeaturesPawnConv9], base_index, training_features);

            const auto& factors = FeatureType::factors_lookup[base_index];

            {
                const auto& properties = kProperties[kFeaturesConv9];
                if (properties.active) {
                    training_features->emplace_back(index_offset + factors.conv_id);
                    index_offset += properties.dimensions;
                }
            }

            {
                const auto& properties = kProperties[kFeaturesPawn];
                if (properties.active) {
                    for (IndexType i = 0; i < factors.num_piece_ids; ++i)
                        training_features->emplace_back(index_offset + factors.piece_ids[i]);

                    index_offset += properties.dimensions;
                }
            }

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<PawnConv9<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif // #ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_PAWN_CONV_9_H_
