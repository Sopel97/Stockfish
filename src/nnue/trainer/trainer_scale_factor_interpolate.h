#ifndef _NNUE_TRAINER_SCALE_FACTOR_INTERPOLATE_H_
#define _NNUE_TRAINER_SCALE_FACTOR_INTERPOLATE_H_

#include "trainer.h"

#include "learn/learn.h"

#include "nnue/layers/scale_factor_interpolate.h"

#include "thread.h"

// Specialization of NNUE evaluation function learning class template for ScaleFactorInterpolate
namespace Eval::NNUE {

    // Learning: Affine transformation layer
    template <typename PreviousLayer>
    class Trainer<Layers::ScaleFactorInterpolate<PreviousLayer>> {
    private:
        // Type of layer to learn
        using LayerType = Layers::ScaleFactorInterpolate<PreviousLayer>;

    public:
        // factory function
        static std::shared_ptr<Trainer> create(
            LayerType* target_layer, FeatureTransformer* ft) {

            return std::shared_ptr<Trainer>(
                new Trainer(target_layer, ft));
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
            previous_layer_trainer_->send_message(message);
            if (receive_message("check_health", message)) {
                check_health();
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            previous_layer_trainer_->initialize(rng);
        }

        // forward propagation
        const LearnFloatType* propagate(ThreadPool& thread_pool, const std::vector<Example>& batch) {
            if (output_.size() < kOutputDimensions * batch.size()) {
              output_.resize(kOutputDimensions * batch.size());
              gradients_.resize(kInputDimensions * batch.size());
            }

            const auto input = previous_layer_trainer_->propagate(thread_pool, batch);

            batch_size_ = static_cast<IndexType>(batch.size());
            batch_ = &batch;

            for (IndexType b = 0; b < batch_size_; ++b) {
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType indexout = kOutputDimensions * b + i;
                    const IndexType indexmg = kInputDimensions * b + 2 * i;
                    const IndexType indexeg = indexmg + 1;
                    float mg = input[indexmg];
                    float eg = input[indexeg];

                    float v =
                         mg * batch[b].phase
                       + eg * (PHASE_MIDGAME - batch[b].phase) * batch[b].scale_factor / SCALE_FACTOR_NORMAL;
                    v /= PHASE_MIDGAME;

                    output_[indexout] = v;
                }
            }

            return output_.data();
        }

        // backpropagation
        void backpropagate(ThreadPool& thread_pool,
                           const LearnFloatType* gradients,
                           LearnFloatType learning_rate) {

            for (IndexType b = 0; b < batch_size_; ++b) {
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType indexout = kOutputDimensions * b + i;
                    const IndexType indexmg = kInputDimensions * b + 2 * i;
                    const IndexType indexeg = indexmg + 1;
                    const float scalemg = float((*batch_)[b].phase) / PHASE_MIDGAME;
                    const float scaleeg = float(PHASE_MIDGAME - (*batch_)[b].phase) * (*batch_)[b].scale_factor / SCALE_FACTOR_NORMAL / PHASE_MIDGAME;
                    gradients_[indexmg] = gradients[indexout] * scalemg;
                    gradients_[indexeg] = gradients[indexout] * scaleeg;
                }
            }

            previous_layer_trainer_->backpropagate(thread_pool, gradients_.data(), learning_rate);
        }

    private:
        // constructor
        Trainer(LayerType* target_layer, FeatureTransformer* ft) :
            batch_(nullptr),
            batch_size_(0),
            previous_layer_trainer_(Trainer<PreviousLayer>::create(
                &target_layer->previous_layer_, ft)),
            target_layer_(target_layer) {

            reset_stats();
        }

        void reset_stats() {
        }

        // Check if there are any problems with learning
        void check_health() {

            reset_stats();
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions = LayerType::kInputDimensions;
        static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

        const std::vector<Example>* batch_;

        // number of samples in mini-batch
        IndexType batch_size_;

        // Trainer of the previous layer
        const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

        // layer to learn
        LayerType* const target_layer_;

        // Forward propagation buffer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;

        // buffer for back propagation
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;
    };

}  // namespace Eval::NNUE

#endif
