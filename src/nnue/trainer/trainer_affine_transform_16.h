﻿#ifndef _NNUE_TRAINER_AFFINE_TRANSFORM_16_H_
#define _NNUE_TRAINER_AFFINE_TRANSFORM_16_H_

#include "trainer.h"

#include "extra/stockfish_blas.h"

#include "learn/learn.h"

#include "nnue/layers/affine_transform_16.h"

#include "thread.h"

#include <random>

// Specialization of NNUE evaluation function learning class template for AffineTransform
namespace Eval::NNUE {

    // Learning: Affine transformation layer
    template <typename PreviousLayer, IndexType OutputDimensions>
    class Trainer<Layers::AffineTransform16<PreviousLayer, OutputDimensions>> {
    private:
        // Type of layer to learn
        using LayerType = Layers::AffineTransform16<PreviousLayer, OutputDimensions>;

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

            if (receive_message("momentum", message)) {
                momentum_ = static_cast<LearnFloatType>(std::stod(message->value));
            }

            if (receive_message("learning_rate_scale", message)) {
                learning_rate_scale_ =
                    static_cast<LearnFloatType>(std::stod(message->value));
            }

            if (receive_message("reset", message)) {
                dequantize_parameters();
            }

            if (receive_message("quantize_parameters", message)) {
                quantize_parameters();
            }

            if (receive_message("check_health", message)) {
                check_health();
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            previous_layer_trainer_->initialize(rng);

            if (kIsOutputLayer) {
                // Initialize output layer with 0
                std::fill(std::begin(biases_), std::end(biases_),
                          static_cast<LearnFloatType>(0.0));
                std::fill(std::begin(weights_), std::end(weights_),
                          static_cast<LearnFloatType>(0.0));
            }
            else {
                // Assuming that the input distribution is unit-mean 0.5, equal variance,
                // Initialize the output distribution so that each unit has a mean of 0.5 and the same variance as the input
                const double kSigma = 1.0 / std::sqrt(kInputDimensions);
                auto distribution = std::normal_distribution<double>(0.0, kSigma);

                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    double sum = 0.0;
                      for (IndexType j = 0; j < kInputDimensions; ++j) {
                          const auto weight = static_cast<LearnFloatType>(distribution(rng));
                          weights_[kInputDimensions * i + j] = weight;
                          sum += weight;
                      }

                    biases_[i] = static_cast<LearnFloatType>(0.5 - 0.5 * sum);
                }
            }

            quantize_parameters();
        }

        const LearnFloatType* step_start(ThreadPool& thread_pool, std::vector<Example>::const_iterator batch_begin, std::vector<Example>::const_iterator batch_end)
        {
            const auto size = batch_end - batch_begin;

            if ((long)output_.size() < (long)kOutputDimensions * size) {
                output_.resize(kOutputDimensions * size);
                gradients_.resize(kInputDimensions * size);
            }

            if (thread_states_.size() < thread_pool.size())
            {
                thread_states_.resize(thread_pool.size());
            }

            combined_batch_size_ = size;
            combined_batch_input_ = previous_layer_trainer_->step_start(thread_pool, batch_begin, batch_end);

            auto& main_thread_state = thread_states_[0];

#if defined(USE_BLAS)

            // update
            cblas_sscal(
                kOutputDimensions, momentum_, main_thread_state.biases_diff_, 1
            );

#else

            Blas::sscal(
                kOutputDimensions, momentum_, main_thread_state.biases_diff_, 1
            );

#endif

            for (IndexType i = 1; i < thread_states_.size(); ++i)
                thread_states_[i].reset_biases();

            return output_.data();
        }

        // forward propagation
        void propagate(Thread& th, const uint64_t offset, const uint64_t count) {

            previous_layer_trainer_->propagate(th, offset, count);

#if defined(USE_BLAS)

            for (IndexType b = offset; b < offset + count; ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                cblas_scopy(
                    kOutputDimensions, biases_, 1, &output_[batch_offset], 1
                );
            }

            cblas_sgemm(
                CblasColMajor, CblasTrans, CblasNoTrans,
                kOutputDimensions, count, kInputDimensions,
                1.0,
                weights_, kInputDimensions,
                combined_batch_input_ + offset * kInputDimensions, kInputDimensions,
                1.0,
                &output_[offset * kOutputDimensions], kOutputDimensions
            );
#else

            for (IndexType b = offset; b < offset + count; ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                Blas::scopy(
                    kOutputDimensions, biases_, 1, &output_[batch_offset], 1
                );
            }

            Blas::sgemm(
                Blas::MatrixLayout::ColMajor, Blas::MatrixTranspose::Trans, Blas::MatrixTranspose::NoTrans,
                kOutputDimensions, count, kInputDimensions,
                1.0,
                weights_, kInputDimensions,
                combined_batch_input_ + offset * kInputDimensions, kInputDimensions,
                1.0,
                &output_[offset * kOutputDimensions], kOutputDimensions
            );

#endif
        }

        // backpropagation
        void backpropagate(Thread& th,
                           const LearnFloatType* gradients,
                           uint64_t offset,
                           uint64_t count) {

            auto& thread_state = thread_states_[th.thread_idx()];
            const auto momentum = th.thread_idx() == 0 ? momentum_ : 0.0f;
#if defined(USE_BLAS)

            cblas_sgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                kInputDimensions, count, kOutputDimensions,
                1.0,
                weights_, kInputDimensions,
                gradients + offset * kOutputDimensions, kOutputDimensions,
                0.0,
                &gradients_[offset * kInputDimensions], kInputDimensions
            );

            for (IndexType b = offset; b < offset + count; ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                cblas_saxpy(
                    kOutputDimensions, 1.0,
                    &gradients[batch_offset], 1, thread_state.biases_diff_, 1
                );
            }

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                kOutputDimensions, kInputDimensions, count,
                1.0,
                gradients + offset * kOutputDimensions, kOutputDimensions,
                combined_batch_input_ + offset * kInputDimensions, kInputDimensions,
                momentum,
                thread_state.weights_diff_, kInputDimensions
            );

#else

            // backpropagate
            Blas::sgemm(
                Blas::MatrixLayout::ColMajor, Blas::MatrixTranspose::NoTrans, Blas::MatrixTranspose::NoTrans,
                kInputDimensions, count, kOutputDimensions,
                1.0,
                weights_, kInputDimensions,
                gradients + offset * kOutputDimensions, kOutputDimensions,
                0.0,
                &gradients_[offset * kInputDimensions], kInputDimensions
            );

            for (IndexType b = offset; b < offset + count; ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                Blas::saxpy(kOutputDimensions, 1.0,
                          &gradients[batch_offset], 1, thread_state.biases_diff_, 1);
            }

            Blas::sgemm(
                Blas::MatrixLayout::RowMajor, Blas::MatrixTranspose::Trans, Blas::MatrixTranspose::NoTrans,
                kOutputDimensions, kInputDimensions, count,
                1.0,
                gradients + offset * kOutputDimensions, kOutputDimensions,
                combined_batch_input_ + offset * kInputDimensions, kInputDimensions,
                momentum,
                thread_state.weights_diff_, kInputDimensions
            );

#endif

            previous_layer_trainer_->backpropagate(th, gradients_.data(), offset, count);
        }

        void reduce_thread_state()
        {
            for (IndexType i = 1; i < thread_states_.size(); ++i)
            {
                thread_states_[0] += thread_states_[i];
            }
        }

        void step_end(ThreadPool& thread_pool, LearnFloatType learning_rate)
        {
            const LearnFloatType local_learning_rate =
                learning_rate * learning_rate_scale_;

            reduce_thread_state();

            auto& main_thread_state = thread_states_[0];

            for (IndexType i = 0; i < kOutputDimensions; ++i) {
                const double d = local_learning_rate * main_thread_state.biases_diff_[i];
                biases_[i] -= d;
                abs_biases_diff_sum_ += std::abs(d);
            }
            num_biases_diffs_ += kOutputDimensions;

            for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
                const double d = local_learning_rate * main_thread_state.weights_diff_[i];
                weights_[i] -= d;
                abs_weights_diff_sum_ += std::abs(d);
            }
            num_weights_diffs_ += kOutputDimensions * kInputDimensions;

            previous_layer_trainer_->step_end(thread_pool, learning_rate);
        }

    private:
        // constructor
        Trainer(LayerType* target_layer, FeatureTransformer* ft) :
            combined_batch_size_(0),
            combined_batch_input_(nullptr),
            previous_layer_trainer_(Trainer<PreviousLayer>::create(
                &target_layer->previous_layer_, ft)),
            target_layer_(target_layer),
            biases_(),
            weights_(),
            momentum_(0.2),
            learning_rate_scale_(1.0) {

            dequantize_parameters();
        }

        void reset_stats() {
            abs_biases_diff_sum_ = 0.0;
            abs_weights_diff_sum_ = 0.0;
            num_biases_diffs_ = 0;
            num_weights_diffs_ = 0;
        }

        void check_health() {

            double abs_bias_sum = 0.0;
            double abs_weight_sum = 0.0;

            for(auto b : biases_)
                abs_bias_sum += std::abs(b);

            for(auto w : weights_)
                abs_weight_sum += std::abs(w);

            auto out = sync_region_cout.new_region();

            out << "INFO (check_health):"
                << " layer " << LayerType::kLayerIndex
                << " - " << LayerType::get_name()
                << std::endl;

            out << "  - avg_abs_bias        = " << abs_bias_sum / std::size(biases_) << std::endl;
            out << "  - avg_abs_bias_diff   = " << abs_biases_diff_sum_ / num_biases_diffs_ << std::endl;
            out << "  - avg_abs_weight      = " << abs_weight_sum / std::size(weights_) << std::endl;
            out << "  - avg_abs_weight_diff = " << abs_weights_diff_sum_ / num_weights_diffs_ << std::endl;

            out.unlock();

            reset_stats();
        }

        // Weight saturation and parameterization
        void quantize_parameters() {
            for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i) {
                weights_[i] = std::max(-kMaxWeightMagnitude,
                                       std::min(+kMaxWeightMagnitude, weights_[i]));
            }

            for (IndexType i = 0; i < kOutputDimensions; ++i) {
                target_layer_->biases_[i] =
                    round<typename LayerType::BiasType>(biases_[i] * kBiasScale);
            }

            for (IndexType i = 0; i < kOutputDimensions; ++i) {
                const auto offset = kInputDimensions * i;
                const auto padded_offset = LayerType::kPaddedInputDimensions * i;
                for (IndexType j = 0; j < kInputDimensions; ++j) {
                    target_layer_->weights_[padded_offset + j] =
                        round<typename LayerType::WeightType>(
                            weights_[offset + j] * kWeightScale);
                }
            }
        }

        // read parameterized integer
        void dequantize_parameters() {
            for (IndexType i = 0; i < kOutputDimensions; ++i) {
                biases_[i] = static_cast<LearnFloatType>(
                    target_layer_->biases_[i] / kBiasScale);
            }

            for (IndexType i = 0; i < kOutputDimensions; ++i) {
                const auto offset = kInputDimensions * i;
                const auto padded_offset = LayerType::kPaddedInputDimensions * i;
                for (IndexType j = 0; j < kInputDimensions; ++j) {
                    weights_[offset + j] = static_cast<LearnFloatType>(
                        target_layer_->weights_[padded_offset + j] / kWeightScale);
                }
            }

            for (auto& state : thread_states_)
            {
                state.reset_weights();
                state.reset_biases();
            }


            reset_stats();
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions = LayerType::kInputDimensions;
        static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;

        // If the output dimensionality is 1, the output layer
        static constexpr bool kIsOutputLayer = kOutputDimensions == 1;

        // Coefficient used for parameterization
        static constexpr LearnFloatType kActivationScale =
            std::numeric_limits<std::int8_t>::max();

        static constexpr LearnFloatType kBiasScale = kIsOutputLayer ?
            (kPonanzaConstant * FV_SCALE) :
            ((1 << kWeightScaleBits) * kActivationScale);

        static constexpr LearnFloatType kWeightScale = kBiasScale / kActivationScale;

        // Upper limit of absolute value of weight used to prevent overflow when parameterizing integers
        static constexpr LearnFloatType kMaxWeightMagnitude =
            std::numeric_limits<typename LayerType::WeightType>::max() / kWeightScale;

        // number of samples in mini-batch
        IndexType combined_batch_size_;

        double abs_biases_diff_sum_;
        double abs_weights_diff_sum_;
        uint64_t num_biases_diffs_;
        uint64_t num_weights_diffs_;

        // Input mini batch
        const LearnFloatType* combined_batch_input_;

        // Trainer of the previous layer
        const std::shared_ptr<Trainer<PreviousLayer>> previous_layer_trainer_;

        // layer to learn
        LayerType* const target_layer_;

        // parameter
        struct alignas(kCacheLineSize) ThreadState
        {
            // Buffer used for updating parameters
            alignas(kCacheLineSize) LearnFloatType biases_diff_[kOutputDimensions];
            alignas(kCacheLineSize) LearnFloatType weights_diff_[kOutputDimensions * kInputDimensions];

            ThreadState() { reset_weights(); reset_biases(); }

            ThreadState& operator+=(const ThreadState& other)
            {
                for (IndexType i = 0; i < kOutputDimensions; ++i)
                {
                    biases_diff_[i] += other.biases_diff_[i];
                }

                for (IndexType i = 0; i < kOutputDimensions * kInputDimensions; ++i)
                {
                    weights_diff_[i] += other.weights_diff_[i];
                }

                return *this;
            }

            void reset_weights()
            {
                std::fill(std::begin(weights_diff_), std::end(weights_diff_), 0.0f);
            }

            void reset_biases()
            {
                std::fill(std::begin(biases_diff_), std::end(biases_diff_), 0.0f);
            }
        };

        alignas(kCacheLineSize) LearnFloatType biases_[kOutputDimensions];
        alignas(kCacheLineSize) LearnFloatType weights_[kOutputDimensions * kInputDimensions];

        std::vector<ThreadState, CacheLineAlignedAllocator<ThreadState>> thread_states_;

        // Forward propagation buffer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;

        // buffer for back propagation
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;

        // hyper parameter
        LearnFloatType momentum_;
        LearnFloatType learning_rate_scale_;
    };

}  // namespace Eval::NNUE

#endif
