#ifndef _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE_H_
#define _NNUE_TRAINER_FEATURES_FACTORIZER_HALF_KPE_H_

#include "factorizer.h"

#include "nnue/features/half_kpe.h"
#include "nnue/features/pe.h"

// Specialization of NNUE evaluation function feature conversion class template for HalfKP
namespace Eval::NNUE::Features {

    // Class template that converts input features into learning features
    // Specialization for HalfKP
    template <Side AssociatedKing>
    class Factorizer<HalfKPE<AssociatedKing>> {
    private:
        using FeatureType = HalfKPE<AssociatedKing>;

        // The maximum value of the number of indexes whose value is 1 at the same time among the feature values
        static constexpr IndexType kMaxActiveDimensions =
            FeatureType::kMaxActiveDimensions;

        // Type of learning feature
        enum TrainingFeatureType {
            kFeaturesHalfKPE,
            kFeaturesHalfK,
            kFeaturesPE,
            kNumTrainingFeatureTypes,
        };

        // Learning feature information
        static constexpr FeatureProperties kProperties[] = {
            // kFeaturesHalfKPE
            {true, FeatureType::kDimensions},
            // kFeaturesHalfK
            {true, SQUARE_NB},
            // kFeaturesPE
            {true, Factorizer<PE>::get_dimensions()},
        };

        static_assert(get_array_length(kProperties) == kNumTrainingFeatureTypes, "");

    public:
        static constexpr std::string get_name() {
            return std::string("Factorizer<") + FeatureType::kName + "> -> " + "HalfK, PE";
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

            // kFeaturesHalfKPE
            IndexType index_offset = append_base_feature<FeatureType>(
                kProperties[kFeaturesHalfKPE], base_index, training_features);

            const auto sq_k = static_cast<Square>(base_index / PS_END3);
            const auto pe = static_cast<IndexType>(base_index % PS_END3);

            // kFeaturesHalfK
            {
                const auto& properties = kProperties[kFeaturesHalfK];
                if (properties.active) {
                    training_features->emplace_back(index_offset + sq_k);
                    index_offset += properties.dimensions;
                }
            }

            // kFeaturesPE
            index_offset += inherit_features_if_required<PE>(
                index_offset, kProperties[kFeaturesPE], pe, training_features);

            assert(index_offset == get_dimensions());
        }
    };

    template <Side AssociatedKing>
    constexpr FeatureProperties Factorizer<HalfKPE<AssociatedKing>>::kProperties[];

}  // namespace Eval::NNUE::Features

#endif
