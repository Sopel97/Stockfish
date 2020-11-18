#ifndef _NNUE_TRAINER_DOUBLE_INPUT_SLICE_H_
#define _NNUE_TRAINER_DOUBLE_INPUT_SLICE_H_

#include "shared_input_trainer.h"

#include "nnue/layers/double_input_slice.h"

#include "trainer.h"

#include "extra/stockfish_blas.h"

#include "learn/learn.h"

#include "nnue/layers/double_input_slice.h"

#include "thread.h"

// Specialization of NNUE evaluation function learning class template for InputSlice
namespace Eval::NNUE {

    // Learning: Input layer
    template <IndexType HalfOutputDimensions, IndexType Stride, IndexType Offset>
    class Trainer<Layers::DoubleInputSlice<HalfOutputDimensions, Stride, Offset>> {
    private:
        // Type of layer to learn
        using LayerType = Layers::DoubleInputSlice<HalfOutputDimensions, Stride, Offset>;

    public:
        // factory function
        static std::shared_ptr<Trainer> create(
            LayerType* /*target_layer*/, FeatureTransformer* ft) {

            return std::shared_ptr<Trainer>(new Trainer(ft));
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
            shared_input_trainer_->send_message(message);
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            shared_input_trainer_->initialize(rng);
        }

        // forward propagation
        const LearnFloatType* propagate(ThreadPool& thread_pool,const std::vector<Example>& batch) {
            if (output_.size() < kOutputDimensions * batch.size()) {
              output_.resize(kOutputDimensions * batch.size());
              gradients_.resize(kInputDimensions * batch.size());
            }

            batch_size_ = static_cast<IndexType>(batch.size());

            const auto input = shared_input_trainer_->propagate(thread_pool, batch);
            for (IndexType b = 0; b < batch_size_; ++b) {
                const IndexType input_offset = kInputDimensions * b;
                const IndexType output_offset = kOutputDimensions * b;

#if defined(USE_BLAS)

                cblas_scopy(
                    kHalfOutputDimensions, &input[input_offset + Offset], 1,
                    &output_[output_offset], 1
                );
                cblas_scopy(
                    kHalfOutputDimensions, &input[input_offset + Stride + Offset], 1,
                    &output_[output_offset + kHalfOutputDimensions], 1
                );
#else

                Blas::scopy(
                    thread_pool,
                    kHalfOutputDimensions, &input[input_offset + Offset], 1,
                    &output_[output_offset], 1
                );
                Blas::scopy(
                    thread_pool,
                    kHalfOutputDimensions, &input[input_offset + Stride + Offset], 1,
                    &output_[output_offset + kHalfOutputDimensions], 1
                );

#endif
            }

            return output_.data();
        }

        // backpropagation
        void backpropagate(ThreadPool& thread_pool,
                           const LearnFloatType* gradients,
                           LearnFloatType learning_rate) {

            thread_pool.for_each_index_with_workers(
                0, batch_size_,
                [&](Thread&, int b) {
                    const IndexType input_offset = kInputDimensions * b;
                    const IndexType output_offset = kOutputDimensions * b;

                    IndexType i = 0;
                    for (; i < Offset; ++i) {
                        gradients_[input_offset + i] = static_cast<LearnFloatType>(0.0);
                    }

                    for (; i < Offset + kHalfOutputDimensions; ++i) {
                        gradients_[input_offset + i] = gradients[output_offset + i - Offset];
                    }

                    for (; i < Stride + Offset; ++i)
                    {
                        gradients_[input_offset + i] = static_cast<LearnFloatType>(0.0);
                    }

                    for (; i < Stride + Offset + kHalfOutputDimensions; ++i) {
                        gradients_[input_offset + i] = gradients[output_offset + kHalfOutputDimensions + i - Offset - Stride];
                    }

                    for (; i < kInputDimensions; ++i)
                    {
                        gradients_[input_offset + i] = static_cast<LearnFloatType>(0.0);
                    }
                }
            );
            thread_pool.wait_for_workers_finished();

            shared_input_trainer_->backpropagate(thread_pool, gradients_.data(), learning_rate);
        }

    private:
        // constructor
        Trainer(FeatureTransformer* ft):
            batch_size_(0),
            shared_input_trainer_(SharedInputTrainer::create(ft)) {
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions =
            FeatureTransformer::kOutputDimensions;

        static constexpr IndexType kHalfOutputDimensions = HalfOutputDimensions;
        static constexpr IndexType kOutputDimensions = HalfOutputDimensions * 2;
        static_assert(Offset + Stride + kHalfOutputDimensions <= kInputDimensions, "");
        static_assert(Offset + kHalfOutputDimensions <= Stride);

        // number of samples in mini-batch
        IndexType batch_size_;

        // Trainer of shared input layer
        const std::shared_ptr<SharedInputTrainer> shared_input_trainer_;

        // Forward propagation buffer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;

        // buffer for back propagation
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;
    };

}  // namespace Eval::NNUE

#endif
