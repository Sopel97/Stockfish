#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_CONSTRAINED_HALF_NA_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_CONSTRAINED_HALF_NA_H_

#include "factorizer.h"

#include "nnue/features/constrained_half_na.h"
#include "nnue/features/a.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKA
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKA
    template <Side AssociatedKing>
    class Factorizer<ConstrainedHalfNA<AssociatedKing>> {
    private:
        using FeatureType = ConstrainedHalfNA<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesConstrainedHalfNA,
            kFeaturesA,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesConstrainedHalfNA
            {true, FeatureType::kDimensions},
            // kFeaturesA
            {true, Factorizer<A>::get_dimensions()},
        };

        static_assert(get_array_length(kProperties) == kNumTrainingFeatureTypes, "");

    public:
        static constexpr std::string get_name() {
            return std::string("Factorizer<") + FeatureType::kName + "> -> " + "A";
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
                kProperties[kFeaturesConstrainedHalfNA], base_index, training_features);

            const auto sq_k = static_cast<Square>(base_index / PS_END2);
            const auto a = static_cast<IndexType>(base_index % PS_END2);

            // kFeaturesA
            index_offset += inherit_features_if_required<A>(
                index_offset, kProperties[kFeaturesA], a, training_features);

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<ConstrainedHalfNA<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif // #ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_QA_H_
