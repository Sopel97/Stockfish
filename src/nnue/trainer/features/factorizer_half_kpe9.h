#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE9_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE9_H_

#include "factorizer.h"

#include "nnue/features/half_kpe9.h"
#include "nnue/features/p.h"
#include "nnue/features/half_relative_kp.h"

#include "factorizer_half_kp.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKPE4
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKPE4
    template <Side AssociatedKing>
    class Factorizer<HalfKPE9<AssociatedKing>> {
    private:
        using FeatureType = HalfKPE9<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesHalfKPE9,
            kFeaturesHalfKP,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesHalfKPE9
            {true, FeatureType::kDimensions},
            // kFeaturesHalfKP
            {true, Factorizer<HalfKP<AssociatedKing>>::get_dimensions()},
        };

        static_assert(get_array_length(kProperties) == kNumTrainingFeatureTypes, "");

    public:
        static constexpr std::string get_name() {
            return std::string("Factorizer<") + FeatureType::kName + "> -> " + "HalfKP -> HalfK, P, HalfRelativeKP";
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

            // kFeaturesHalfKP
            IndexType index_offset = append_base_feature<FeatureType>(
                kProperties[kFeaturesHalfKPE9], base_index, training_features);

            IndexType index_HalfKP = base_index % (static_cast<IndexType>(PS_END) * static_cast<IndexType>(SQUARE_NB));

            index_offset +=
                inherit_features_if_required<HalfKP<AssociatedKing>>(
                    index_offset, kProperties[kFeaturesHalfKP], index_HalfKP, training_features);

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<HalfKPE9<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif
