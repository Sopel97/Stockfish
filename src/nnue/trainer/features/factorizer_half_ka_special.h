#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KA_SPECIAL_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KA_SPECIAL_H_

#include "factorizer.h"

#include "nnue/features/half_ka.h"
#include "nnue/features/half_ka_special.h"
#include "nnue/features/a.h"
#include "nnue/features/half_relative_ka.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKASpecial
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKASpecial
    template <Side AssociatedKing>
    class Factorizer<HalfKASpecial<AssociatedKing>> {
    private:
        using FeatureType = HalfKASpecial<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesHalfKASpecial,
            kFeaturesHalfKA,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesHalfKASpecial
            {true, FeatureType::kDimensions},
            // kFeaturesHalfKa
            {true, Factorizer<HalfKA<AssociatedKing>>::get_dimensions()}
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

            // kFeaturesHalfA
            IndexType index_offset = append_base_feature<FeatureType>(
                kProperties[kFeaturesHalfKA], base_index, training_features);

            const auto halfka = base_index % (PS_END2 * SQUARE_NB);

            // kFeaturesA
            index_offset += inherit_features_if_required<HalfKA<AssociatedKing>>(
                index_offset, kProperties[kFeaturesHalfKA], halfka, training_features);

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<HalfKASpecial<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif // #ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KA_SPECIAL_H_
