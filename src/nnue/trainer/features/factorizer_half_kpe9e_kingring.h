#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE9E_KINGRING_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE9E_KINGRING_H_

#include "factorizer.h"

#include "nnue/features/half_kpe9e_kingring.h"
#include "nnue/features/pe.h"
#include "nnue/features/half_relative_kp.h"

#include "factorizer_half_kpe.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKPE4
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKPE4
    template <Side AssociatedKing>
    class Factorizer<HalfKPE9E_KingRing<AssociatedKing>> {
    private:
        using FeatureType = HalfKPE9E_KingRing<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesHalfKPE9_KingRing,
            kFeaturesHalfKPE,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesHalfKPE9_KingRing
            {true, FeatureType::kDimensions},
            // kFeaturesHalfKPE
            {true, Factorizer<HalfKPE<AssociatedKing>>::get_dimensions()},
        };

        static_assert(get_array_length(kProperties) == kNumTrainingFeatureTypes, "");

    public:
        static constexpr std::string get_name() {
            return std::string("Factorizer<") + FeatureType::kName + "> -> " + "HalfKPE -> HalfK, PE";
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

            // kFeaturesHalfKPE9_KingRing
            IndexType index_offset = append_base_feature<FeatureType>(
                kProperties[kFeaturesHalfKPE9_KingRing], base_index, training_features);

            IndexType index_HalfKPE = base_index % (static_cast<IndexType>(PS_END3) * static_cast<IndexType>(SQUARE_NB));
            IndexType mobility_index = base_index / (static_cast<IndexType>(PS_END3) * static_cast<IndexType>(SQUARE_NB));

            IndexType our_mobility_index = mobility_index / 3;
            IndexType their_mobility_index = mobility_index % 3;

            // If something is attacked twice also add a feature
            // for when it's attacked once.
            if (our_mobility_index == 2)
            {
                const IndexType new_mobility_index =
                    (our_mobility_index - 1) * 3
                    + their_mobility_index;

                const IndexType index =
                    index_HalfKPE
                    + (static_cast<IndexType>(PS_END3) * static_cast<IndexType>(SQUARE_NB)) * new_mobility_index;

                append_base_feature<FeatureType>(
                    kProperties[kFeaturesHalfKPE9_KingRing], index, training_features);
            }

            if (their_mobility_index == 2)
            {
                const IndexType new_mobility_index =
                    our_mobility_index * 3
                    + (their_mobility_index - 1);

                const IndexType index =
                    index_HalfKPE
                    + (static_cast<IndexType>(PS_END3) * static_cast<IndexType>(SQUARE_NB)) * new_mobility_index;

                append_base_feature<FeatureType>(
                    kProperties[kFeaturesHalfKPE9_KingRing], index, training_features);
            }

            if (their_mobility_index == 2 && our_mobility_index == 2)
            {
                const IndexType new_mobility_index =
                    (our_mobility_index - 1) * 3
                    + (their_mobility_index - 1);

                const IndexType index =
                    index_HalfKPE
                    + (static_cast<IndexType>(PS_END3) * static_cast<IndexType>(SQUARE_NB)) * new_mobility_index;

                append_base_feature<FeatureType>(
                    kProperties[kFeaturesHalfKPE9_KingRing], index, training_features);
            }

            index_offset +=
                inherit_features_if_required<HalfKPE<AssociatedKing>>(
                    index_offset, kProperties[kFeaturesHalfKPE], index_HalfKPE, training_features);

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<HalfKPE9E_KingRing<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif
