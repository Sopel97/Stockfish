#ifndef NNUE_SHARED_INPUT_TRAINER_H
#define NNUE_SHARED_INPUT_TRAINER_H

#include <memory>
#include <cassert>

#include "trainer.h"

#include "learn/learn.h"

#include "extra/stockfish_blas.h"

#include "thread.h"

namespace Eval::NNUE {

    // Learning: Input layer
    class SharedInputTrainer {
    public:
        // factory function
        static std::shared_ptr<SharedInputTrainer> create(
            FeatureTransformer* ft) {

            static std::shared_ptr<SharedInputTrainer> instance;

            if (!instance) {
                instance.reset(new SharedInputTrainer(ft));
            }

            ++instance->num_referrers_;

            return instance;
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
            if (num_calls_ == 0) {
                current_operation_ = Operation::kSendMessage;
                feature_transformer_trainer_->send_message(message);
            }

            assert(current_operation_ == Operation::kSendMessage);

            if (++num_calls_ == num_referrers_) {
                num_calls_ = 0;
                current_operation_ = Operation::kNone;
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            if (num_calls_ == 0) {
                current_operation_ = Operation::kInitialize;
                feature_transformer_trainer_->initialize(rng);
            }

            assert(current_operation_ == Operation::kInitialize);

            if (++num_calls_ == num_referrers_) {
                num_calls_ = 0;
                current_operation_ = Operation::kNone;
            }
        }

        // forward propagation
        const LearnFloatType* propagate(ThreadPool& thread_pool, const std::vector<Example>& batch) {
            if (gradients_.size() < kInputDimensions * batch.size()) {
                gradients_.resize(kInputDimensions * batch.size());
            }

            batch_size_ = static_cast<IndexType>(batch.size());

            if (num_calls_ == 0) {
                current_operation_ = Operation::kPropagate;
                output_ = feature_transformer_trainer_->propagate(thread_pool, batch);
            }

            assert(current_operation_ == Operation::kPropagate);

            if (++num_calls_ == num_referrers_) {
                num_calls_ = 0;
                current_operation_ = Operation::kNone;
            }

            return output_;
        }

        // backpropagation
        void backpropagate(ThreadPool& thread_pool,
                           const LearnFloatType* gradients,
                           LearnFloatType learning_rate) {

            if (num_referrers_ == 1) {
                feature_transformer_trainer_->backpropagate(thread_pool, gradients, learning_rate);
                return;
            }

            if (num_calls_ == 0) {
                current_operation_ = Operation::kBackPropagate;
                for (IndexType b = 0; b < batch_size_; ++b) {
                    const IndexType batch_offset = kInputDimensions * b;
                    for (IndexType i = 0; i < kInputDimensions; ++i) {
                        gradients_[batch_offset + i] = static_cast<LearnFloatType>(0.0);
                    }
                }
            }

            assert(current_operation_ == Operation::kBackPropagate);

            for (IndexType b = 0; b < batch_size_; ++b) {
                const IndexType batch_offset = kInputDimensions * b;
                for (IndexType i = 0; i < kInputDimensions; ++i) {
                    gradients_[batch_offset + i] += gradients[batch_offset + i];
                }
            }

            if (++num_calls_ == num_referrers_) {
                feature_transformer_trainer_->backpropagate(
                    thread_pool, gradients_.data(), learning_rate);
                num_calls_ = 0;
                current_operation_ = Operation::kNone;
            }
        }

    private:
        // constructor
        SharedInputTrainer(FeatureTransformer* ft) :
            batch_size_(0),
            num_referrers_(0),
            num_calls_(0),
            current_operation_(Operation::kNone),
            feature_transformer_trainer_(Trainer<FeatureTransformer>::create(
                ft)),
            output_(nullptr) {
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions =
            FeatureTransformer::kOutputDimensions;

        // type of processing
        enum class Operation {
            kNone,
            kSendMessage,
            kInitialize,
            kPropagate,
            kBackPropagate,
        };

        // number of samples in mini-batch
        IndexType batch_size_;

        // number of layers sharing this layer as input
        std::uint32_t num_referrers_;

        // Number of times the current process has been called
        std::uint32_t num_calls_;

        // current processing type
        Operation current_operation_;

        // Trainer of input feature converter
        const std::shared_ptr<Trainer<FeatureTransformer>>
            feature_transformer_trainer_;

        // pointer to output shared for forward propagation
        const LearnFloatType* output_;

        // buffer for back propagation
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;
    };
}

#endif
