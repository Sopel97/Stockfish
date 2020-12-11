#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KAE5_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KAE5_H_

#include "factorizer.h"

#include "nnue/features/half_kae5.h"

#include "factorizer_half_ka.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKAE5
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKAE4
    template <Side AssociatedKing>
    class Factorizer<HalfKAE5<AssociatedKing>> {
    private:
        using FeatureType = HalfKAE5<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesHalfKAE5,
            kFeaturesHalfKA,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesHalfKAE5
            {true, FeatureType::kDimensions},
            // kFeaturesHalfKA
            {true, Factorizer<HalfKA<AssociatedKing>>::get_dimensions()},
        };

        static_assert(get_array_length(kProperties) == kNumTrainingFeatureTypes, "");

    public:
        static constexpr std::string get_name() {
            return std::string("Factorizer<") + FeatureType::kName + "> -> " + "HalfKA -> A, HalfRelativeKA";
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

            // kFeaturesHalfKAE5
            IndexType index_offset = append_base_feature<FeatureType>(
                kProperties[kFeaturesHalfKAE5], base_index, training_features);

            IndexType index_HalfKA = base_index % (static_cast<IndexType>(PS_END2) * static_cast<IndexType>(SQUARE_NB));

            index_offset +=
                inherit_features_if_required<HalfKA<AssociatedKing>>(
                    index_offset, kProperties[kFeaturesHalfKA], index_HalfKA, training_features);

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<HalfKAE5<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif
