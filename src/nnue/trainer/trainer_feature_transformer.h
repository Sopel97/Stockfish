#ifndef _NNUE_TRAINER_FEATURE_TRANSFORMER_H_
#define _NNUE_TRAINER_FEATURE_TRANSFORMER_H_

#include "trainer.h"

#include "extra/stockfish_blas.h"

#include "features/factorizer_feature_set.h"

#include "learn/learn.h"

#include "nnue/nnue_feature_transformer.h"

#include "thread.h"

#include <array>
#include <bitset>
#include <numeric>
#include <random>
#include <set>

// Specialization for feature transformer of learning class template of NNUE evaluation function
namespace Eval::NNUE {

    // Learning: Input feature converter
    template <>
    class Trainer<FeatureTransformer> {
    private:
        // Type of layer to learn
        using LayerType = FeatureTransformer;

    public:
        template <typename T>
        friend struct AlignedDeleter;

        template <typename T, typename... ArgumentTypes>
        friend std::shared_ptr<T> make_aligned_shared_ptr(ArgumentTypes&&... arguments);

        // factory function
        static std::shared_ptr<Trainer> create(LayerType* target_layer) {
            return make_aligned_shared_ptr<Trainer>(target_layer);
        }

        // Set options such as hyperparameters
        void send_message(Message* message) {
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

            if (receive_message("clear_unobserved_feature_weights", message)) {
                clear_unobserved_feature_weights();
            }

            if (receive_message("check_health", message)) {
                check_health();
            }
        }

        // Initialize the parameters with random numbers
        template <typename RNG>
        void initialize(RNG& rng) {
            {
                std::fill(std::begin(weights_), std::end(weights_), +kZero);

                const double kSigma = 0.1 / std::sqrt(RawFeatures::kMaxActiveDimensions);
                auto distribution = std::normal_distribution<double>(0.0, kSigma);

                for (IndexType i = 0; i < kHalfDimensions * RawFeatures::kDimensions; ++i) {
                    const auto weight = static_cast<LearnFloatType>(distribution(rng));
                    weights_[i] = weight;
                }

                for (IndexType i = 0; i < kHalfDimensions; ++i) {
                    biases_[i] = static_cast<LearnFloatType>(0.5);
                }
            }

            {
                std::fill(std::begin(extra_weights_), std::end(extra_weights_), +kZero);

                const double kSigma = 0.1 / std::sqrt(RawExtraFeatures::kMaxActiveDimensions);
                auto distribution = std::normal_distribution<double>(0.0, kSigma);

                for (IndexType i = 0; i < kHalfExtraDimensions * RawExtraFeatures::kDimensions; ++i) {
                    const auto weight = static_cast<LearnFloatType>(distribution(rng));
                    extra_weights_[i] = weight;
                }

                for (IndexType i = 0; i < kHalfExtraDimensions; ++i) {
                    extra_biases_[i] = static_cast<LearnFloatType>(0.5);
                }
            }

            quantize_parameters();
        }

        // forward propagation
        const LearnFloatType* propagate(ThreadPool& thread_pool, const std::vector<Example>& batch) {
            if (output_.size() < kOutputTotalDimensions * batch.size()) {
                output_.resize(kOutputTotalDimensions * batch.size());
                gradients_.resize(kOutputTotalDimensions * batch.size());
            }

            (void)thread_pool;

            batch_ = &batch;
            // affine transform
            thread_pool.for_each_index_with_workers(
                0, batch.size(),
                [&](Thread&, int b) {
                    const IndexType batch_offset = kOutputTotalDimensions * b;
                    for (IndexType c = 0; c < 2; ++c) {
                        const IndexType output_offset = batch_offset + kHalfDimensions * c;

#if defined(USE_BLAS)

                        cblas_scopy(
                            kHalfDimensions, biases_, 1, &output_[output_offset], 1
                        );

                        for (const auto& feature : batch[b].training_features[c]) {
                            const IndexType weights_offset = kHalfDimensions * feature.get_index();
                            cblas_saxpy(
                                kHalfDimensions, (float)feature.get_count(),
                                &weights_[weights_offset], 1, &output_[output_offset], 1
                            );
                        }

#else

                        Blas::scopy(
                            kHalfDimensions, biases_, 1, &output_[output_offset], 1
                        );
                        for (const auto& feature : batch[b].training_features[c]) {
                            const IndexType weights_offset = kHalfDimensions * feature.get_index();
                            Blas::saxpy(
                                kHalfDimensions, (float)feature.get_count(),
                                &weights_[weights_offset], 1, &output_[output_offset], 1
                            );
                        }

#endif
                    }

                    for (IndexType c = 0; c < 2; ++c) {
                        const IndexType output_offset = batch_offset + kOutputDimensions + kHalfExtraDimensions * c;

#if defined(USE_BLAS)

                        cblas_scopy(
                            kHalfExtraDimensions, extra_biases_, 1, &output_[output_offset], 1
                        );

                        for (const auto& feature : batch[b].extra_training_features[c]) {
                            const IndexType weights_offset = kHalfExtraDimensions * feature.get_index();
                            cblas_saxpy(
                                kHalfExtraDimensions, (float)feature.get_count(),
                                &extra_weights_[weights_offset], 1, &output_[output_offset], 1
                            );
                        }

#else

                        Blas::scopy(
                            kHalfExtraDimensions, extra_biases_, 1, &output_[output_offset], 1
                        );
                        for (const auto& feature : batch[b].extra_training_features[c]) {
                            const IndexType weights_offset = kHalfExtraDimensions * feature.get_index();
                            Blas::saxpy(
                                kHalfExtraDimensions, (float)feature.get_count(),
                                &extra_weights_[weights_offset], 1, &output_[output_offset], 1
                            );
                        }

#endif
                    }
                }
            );
            thread_pool.wait_for_workers_finished();

#if defined (USE_SSE2)

            {
                static_assert(kHalfDimensions % 16 == 0, "This implementation assumes that it can process 16 floats at a time");
                static_assert(kHalfExtraDimensions % 16 == 0, "This implementation assumes that it can process 16 floats at a time");

                auto m128_hmin_ps = [](__m128 x3210) {
                    __m128 x0032 = _mm_shuffle_ps(x3210, x3210, _MM_SHUFFLE(0, 0, 3, 2));
                    __m128 min_x_x_13_20 = _mm_min_ps(x3210, x0032);
                    // a = [ # , # , min(x[1], x[3]) , min(x[2], x[0]) ]
                    __m128 min_x_x_20_13 = _mm_shuffle_ps(min_x_x_13_20, min_x_x_13_20, _MM_SHUFFLE(0, 0, 0, 1));
                    return _mm_cvtss_f32(_mm_min_ps(min_x_x_13_20, min_x_x_20_13));
                };

                auto m128_hmax_ps = [](__m128 x3210) {
                    __m128 x0032 = _mm_shuffle_ps(x3210, x3210, _MM_SHUFFLE(0, 0, 3, 2));
                    __m128 max_x_x_13_20 = _mm_max_ps(x3210, x0032);
                    // a = [ # , # , max(x[1], x[3]) , max(x[2], x[0]) ]
                    __m128 max_x_x_20_13 = _mm_shuffle_ps(max_x_x_13_20, max_x_x_13_20, _MM_SHUFFLE(0, 0, 0, 1));
                    return _mm_cvtss_f32(_mm_max_ps(max_x_x_13_20, max_x_x_20_13));
                };

                const __m128 kZero4 = _mm_set1_ps(+kZero);
                const __m128 kOne4 = _mm_set1_ps(+kOne);

                __m128 min_pre_activation0 = _mm_set1_ps(min_pre_activation_);
                __m128 min_pre_activation1 = _mm_set1_ps(min_pre_activation_);
                __m128 max_pre_activation0 = _mm_set1_ps(max_pre_activation_);
                __m128 max_pre_activation1 = _mm_set1_ps(max_pre_activation_);
                __m128 min_extra_pre_activation0 = _mm_set1_ps(min_extra_pre_activation_);
                __m128 min_extra_pre_activation1 = _mm_set1_ps(min_extra_pre_activation_);
                __m128 max_extra_pre_activation0 = _mm_set1_ps(max_extra_pre_activation_);
                __m128 max_extra_pre_activation1 = _mm_set1_ps(max_extra_pre_activation_);

                for (IndexType b = 0; b < batch.size(); ++b)
                {
                    const IndexType batch_offset = kOutputTotalDimensions * b;
                    for (IndexType i = 0; i < kOutputDimensions; i += 16)
                    {
                        __m128 out0 = _mm_loadu_ps(&output_[i +  0 + batch_offset]);
                        __m128 out1 = _mm_loadu_ps(&output_[i +  4 + batch_offset]);
                        __m128 out2 = _mm_loadu_ps(&output_[i +  8 + batch_offset]);
                        __m128 out3 = _mm_loadu_ps(&output_[i + 12 + batch_offset]);

                        __m128 min01 = _mm_min_ps(out0, out1);
                        __m128 min23 = _mm_min_ps(out2, out3);

                        __m128 max01 = _mm_max_ps(out0, out1);
                        __m128 max23 = _mm_max_ps(out2, out3);

                        min_pre_activation0 = _mm_min_ps(min_pre_activation0, min01);
                        min_pre_activation1 = _mm_min_ps(min_pre_activation1, min23);
                        max_pre_activation0 = _mm_max_ps(max_pre_activation0, max01);
                        max_pre_activation1 = _mm_max_ps(max_pre_activation1, max23);

                        out0 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out0));
                        out1 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out1));
                        out2 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out2));
                        out3 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out3));

                        _mm_storeu_ps(&output_[i +  0 + batch_offset], out0);
                        _mm_storeu_ps(&output_[i +  4 + batch_offset], out1);
                        _mm_storeu_ps(&output_[i +  8 + batch_offset], out2);
                        _mm_storeu_ps(&output_[i + 12 + batch_offset], out3);
                    }

                    for (IndexType i = kOutputDimensions; i < kOutputTotalDimensions; i += 16)
                    {
                        __m128 out0 = _mm_loadu_ps(&output_[i +  0 + batch_offset]);
                        __m128 out1 = _mm_loadu_ps(&output_[i +  4 + batch_offset]);
                        __m128 out2 = _mm_loadu_ps(&output_[i +  8 + batch_offset]);
                        __m128 out3 = _mm_loadu_ps(&output_[i + 12 + batch_offset]);

                        __m128 min01 = _mm_min_ps(out0, out1);
                        __m128 min23 = _mm_min_ps(out2, out3);

                        __m128 max01 = _mm_max_ps(out0, out1);
                        __m128 max23 = _mm_max_ps(out2, out3);

                        min_extra_pre_activation0 = _mm_min_ps(min_extra_pre_activation0, min01);
                        min_extra_pre_activation1 = _mm_min_ps(min_extra_pre_activation1, min23);
                        max_extra_pre_activation0 = _mm_max_ps(max_extra_pre_activation0, max01);
                        max_extra_pre_activation1 = _mm_max_ps(max_extra_pre_activation1, max23);

                        out0 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out0));
                        out1 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out1));
                        out2 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out2));
                        out3 = _mm_max_ps(kZero4, _mm_min_ps(kOne4, out3));

                        _mm_storeu_ps(&output_[i +  0 + batch_offset], out0);
                        _mm_storeu_ps(&output_[i +  4 + batch_offset], out1);
                        _mm_storeu_ps(&output_[i +  8 + batch_offset], out2);
                        _mm_storeu_ps(&output_[i + 12 + batch_offset], out3);
                    }
                }

                min_pre_activation_ = m128_hmin_ps(_mm_min_ps(min_pre_activation0, min_pre_activation1));
                max_pre_activation_ = m128_hmax_ps(_mm_max_ps(max_pre_activation0, max_pre_activation1));
                min_extra_pre_activation_ = m128_hmin_ps(_mm_min_ps(min_extra_pre_activation0, min_extra_pre_activation1));
                max_extra_pre_activation_ = m128_hmax_ps(_mm_max_ps(max_extra_pre_activation0, max_extra_pre_activation1));

                for (IndexType b = 0; b < batch.size(); ++b)
                {
                    const IndexType batch_offset = kOutputTotalDimensions * b;

                    for (IndexType half = 0; half < 2; ++half)
                    {
                        const IndexType half_offset = batch_offset + half * kHalfDimensions;
                        for (IndexType i = 0; i < kHalfDimensions; i += 16)
                        {
                            const __m128 out0 = _mm_loadu_ps(&output_[i +  0 + half_offset]);
                            const __m128 out1 = _mm_loadu_ps(&output_[i +  4 + half_offset]);
                            const __m128 out2 = _mm_loadu_ps(&output_[i +  8 + half_offset]);
                            const __m128 out3 = _mm_loadu_ps(&output_[i + 12 + half_offset]);

                            __m128 minact0 = _mm_loadu_ps(&min_activations_[i +  0]);
                            __m128 minact1 = _mm_loadu_ps(&min_activations_[i +  4]);
                            __m128 minact2 = _mm_loadu_ps(&min_activations_[i +  8]);
                            __m128 minact3 = _mm_loadu_ps(&min_activations_[i + 12]);

                            __m128 maxact0 = _mm_loadu_ps(&max_activations_[i +  0]);
                            __m128 maxact1 = _mm_loadu_ps(&max_activations_[i +  4]);
                            __m128 maxact2 = _mm_loadu_ps(&max_activations_[i +  8]);
                            __m128 maxact3 = _mm_loadu_ps(&max_activations_[i + 12]);

                            minact0 = _mm_min_ps(out0, minact0);
                            minact1 = _mm_min_ps(out1, minact1);
                            minact2 = _mm_min_ps(out2, minact2);
                            minact3 = _mm_min_ps(out3, minact3);

                            maxact0 = _mm_max_ps(out0, maxact0);
                            maxact1 = _mm_max_ps(out1, maxact1);
                            maxact2 = _mm_max_ps(out2, maxact2);
                            maxact3 = _mm_max_ps(out3, maxact3);

                            _mm_storeu_ps(&min_activations_[i +  0], minact0);
                            _mm_storeu_ps(&min_activations_[i +  4], minact1);
                            _mm_storeu_ps(&min_activations_[i +  8], minact2);
                            _mm_storeu_ps(&min_activations_[i + 12], minact3);

                            _mm_storeu_ps(&max_activations_[i +  0], maxact0);
                            _mm_storeu_ps(&max_activations_[i +  4], maxact1);
                            _mm_storeu_ps(&max_activations_[i +  8], maxact2);
                            _mm_storeu_ps(&max_activations_[i + 12], maxact3);
                        }
                    }

                    for (IndexType half = 0; half < 2; ++half)
                    {
                        const IndexType half_offset = batch_offset + kOutputDimensions + half * kHalfExtraDimensions;
                        for (IndexType i = 0; i < kHalfExtraDimensions; i += 16)
                        {
                            const __m128 out0 = _mm_loadu_ps(&output_[i +  0 + half_offset]);
                            const __m128 out1 = _mm_loadu_ps(&output_[i +  4 + half_offset]);
                            const __m128 out2 = _mm_loadu_ps(&output_[i +  8 + half_offset]);
                            const __m128 out3 = _mm_loadu_ps(&output_[i + 12 + half_offset]);

                            __m128 extra_minact0 = _mm_loadu_ps(&min_extra_activations_[i +  0]);
                            __m128 extra_minact1 = _mm_loadu_ps(&min_extra_activations_[i +  4]);
                            __m128 extra_minact2 = _mm_loadu_ps(&min_extra_activations_[i +  8]);
                            __m128 extra_minact3 = _mm_loadu_ps(&min_extra_activations_[i + 12]);

                            __m128 extra_maxact0 = _mm_loadu_ps(&max_extra_activations_[i +  0]);
                            __m128 extra_maxact1 = _mm_loadu_ps(&max_extra_activations_[i +  4]);
                            __m128 extra_maxact2 = _mm_loadu_ps(&max_extra_activations_[i +  8]);
                            __m128 extra_maxact3 = _mm_loadu_ps(&max_extra_activations_[i + 12]);

                            extra_minact0 = _mm_min_ps(out0, extra_minact0);
                            extra_minact1 = _mm_min_ps(out1, extra_minact1);
                            extra_minact2 = _mm_min_ps(out2, extra_minact2);
                            extra_minact3 = _mm_min_ps(out3, extra_minact3);

                            extra_maxact0 = _mm_max_ps(out0, extra_maxact0);
                            extra_maxact1 = _mm_max_ps(out1, extra_maxact1);
                            extra_maxact2 = _mm_max_ps(out2, extra_maxact2);
                            extra_maxact3 = _mm_max_ps(out3, extra_maxact3);

                            _mm_storeu_ps(&min_extra_activations_[i +  0], extra_minact0);
                            _mm_storeu_ps(&min_extra_activations_[i +  4], extra_minact1);
                            _mm_storeu_ps(&min_extra_activations_[i +  8], extra_minact2);
                            _mm_storeu_ps(&min_extra_activations_[i + 12], extra_minact3);

                            _mm_storeu_ps(&max_extra_activations_[i +  0], extra_maxact0);
                            _mm_storeu_ps(&max_extra_activations_[i +  4], extra_maxact1);
                            _mm_storeu_ps(&max_extra_activations_[i +  8], extra_maxact2);
                            _mm_storeu_ps(&max_extra_activations_[i + 12], extra_maxact3);
                        }
                    }
                }
            }

#else

            // clipped ReLU
            for (IndexType b = 0; b < batch.size(); ++b) {
                const IndexType batch_offset = kOutputTotalDimensions * b;
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    min_pre_activation_ = std::min(min_pre_activation_, output_[index]);
                    max_pre_activation_ = std::max(max_pre_activation_, output_[index]);
                    output_[index] = std::max(+kZero, std::min(+kOne, output_[index]));
                    const IndexType t = i % kHalfDimensions;
                    min_activations_[t] = std::min(min_activations_[t], output_[index]);
                    max_activations_[t] = std::max(max_activations_[t], output_[index]);
                }

                for (IndexType i = 0; i < kOutputExtraDimensions; ++i) {
                    const IndexType index = batch_offset + kOutputDimensions + i;
                    min_extra_pre_activation_ = std::min(min_extra_pre_activation_, output_[index]);
                    max_extra_pre_activation_ = std::max(max_extra_pre_activation_, output_[index]);
                    output_[index] = std::max(+kZero, std::min(+kOne, output_[index]));
                    const IndexType t = i % kHalfExtraDimensions;
                    min_extra_activations_[t] = std::min(min_extra_activations_[t], output_[index]);
                    max_extra_activations_[t] = std::max(max_extra_activations_[t], output_[index]);
                }
            }

#endif

            return output_.data();
        }

        // backpropagation
        void backpropagate(ThreadPool& thread_pool,
                           const LearnFloatType* gradients,
                           LearnFloatType learning_rate) {

            (void)thread_pool;

            const LearnFloatType local_learning_rate =
                learning_rate * learning_rate_scale_;

#if defined (USE_SSE2)

            {
                static_assert(kHalfDimensions % 16 == 0, "This implementation assumes that it can process 16 floats at a time");
                static_assert(kHalfExtraDimensions % 16 == 0, "This implementation assumes that it can process 16 floats at a time");

                const __m128 kZero4 = _mm_set1_ps(+kZero);
                const __m128 kOne4 = _mm_set1_ps(+kOne);

                for (IndexType b = 0; b < batch_->size(); ++b)
                {
                    const IndexType batch_offset = kOutputTotalDimensions * b;
                    for (IndexType i = 0; i < kOutputDimensions; i += 16)
                    {
                        __m128 out0 = _mm_loadu_ps(&output_[i + 0 + batch_offset]);
                        __m128 out1 = _mm_loadu_ps(&output_[i + 4 + batch_offset]);
                        __m128 out2 = _mm_loadu_ps(&output_[i + 8 + batch_offset]);
                        __m128 out3 = _mm_loadu_ps(&output_[i + 12 + batch_offset]);

                        __m128 clipped0 = _mm_or_ps(_mm_cmple_ps(out0, kZero4), _mm_cmpge_ps(out0, kOne4));
                        __m128 clipped1 = _mm_or_ps(_mm_cmple_ps(out1, kZero4), _mm_cmpge_ps(out1, kOne4));
                        __m128 clipped2 = _mm_or_ps(_mm_cmple_ps(out2, kZero4), _mm_cmpge_ps(out2, kOne4));
                        __m128 clipped3 = _mm_or_ps(_mm_cmple_ps(out3, kZero4), _mm_cmpge_ps(out3, kOne4));

                        __m128 grad0 = _mm_loadu_ps(&gradients[i + 0 + batch_offset]);
                        __m128 grad1 = _mm_loadu_ps(&gradients[i + 4 + batch_offset]);
                        __m128 grad2 = _mm_loadu_ps(&gradients[i + 8 + batch_offset]);
                        __m128 grad3 = _mm_loadu_ps(&gradients[i + 12 + batch_offset]);

                        grad0 = _mm_andnot_ps(clipped0, grad0);
                        grad1 = _mm_andnot_ps(clipped1, grad1);
                        grad2 = _mm_andnot_ps(clipped2, grad2);
                        grad3 = _mm_andnot_ps(clipped3, grad3);

                        _mm_storeu_ps(&gradients_[i + 0 + batch_offset], grad0);
                        _mm_storeu_ps(&gradients_[i + 4 + batch_offset], grad1);
                        _mm_storeu_ps(&gradients_[i + 8 + batch_offset], grad2);
                        _mm_storeu_ps(&gradients_[i + 12 + batch_offset], grad3);

                        const int clipped_mask =
                            (_mm_movemask_ps(clipped0) << 0)
                            | (_mm_movemask_ps(clipped1) << 4)
                            | (_mm_movemask_ps(clipped2) << 8)
                            | (_mm_movemask_ps(clipped3) << 12);

                        num_clipped_ += popcount(clipped_mask);
                    }

                    for (IndexType i = kOutputDimensions; i < kOutputTotalDimensions; i += 16)
                    {
                        __m128 out0 = _mm_loadu_ps(&output_[i + 0 + batch_offset]);
                        __m128 out1 = _mm_loadu_ps(&output_[i + 4 + batch_offset]);
                        __m128 out2 = _mm_loadu_ps(&output_[i + 8 + batch_offset]);
                        __m128 out3 = _mm_loadu_ps(&output_[i + 12 + batch_offset]);

                        __m128 extra_clipped0 = _mm_or_ps(_mm_cmple_ps(out0, kZero4), _mm_cmpge_ps(out0, kOne4));
                        __m128 extra_clipped1 = _mm_or_ps(_mm_cmple_ps(out1, kZero4), _mm_cmpge_ps(out1, kOne4));
                        __m128 extra_clipped2 = _mm_or_ps(_mm_cmple_ps(out2, kZero4), _mm_cmpge_ps(out2, kOne4));
                        __m128 extra_clipped3 = _mm_or_ps(_mm_cmple_ps(out3, kZero4), _mm_cmpge_ps(out3, kOne4));

                        __m128 grad0 = _mm_loadu_ps(&gradients[i + 0 + batch_offset]);
                        __m128 grad1 = _mm_loadu_ps(&gradients[i + 4 + batch_offset]);
                        __m128 grad2 = _mm_loadu_ps(&gradients[i + 8 + batch_offset]);
                        __m128 grad3 = _mm_loadu_ps(&gradients[i + 12 + batch_offset]);

                        grad0 = _mm_andnot_ps(extra_clipped0, grad0);
                        grad1 = _mm_andnot_ps(extra_clipped1, grad1);
                        grad2 = _mm_andnot_ps(extra_clipped2, grad2);
                        grad3 = _mm_andnot_ps(extra_clipped3, grad3);

                        _mm_storeu_ps(&gradients_[i + 0 + batch_offset], grad0);
                        _mm_storeu_ps(&gradients_[i + 4 + batch_offset], grad1);
                        _mm_storeu_ps(&gradients_[i + 8 + batch_offset], grad2);
                        _mm_storeu_ps(&gradients_[i + 12 + batch_offset], grad3);

                        const int extra_clipped_mask =
                            (_mm_movemask_ps(extra_clipped0) << 0)
                            | (_mm_movemask_ps(extra_clipped1) << 4)
                            | (_mm_movemask_ps(extra_clipped2) << 8)
                            | (_mm_movemask_ps(extra_clipped3) << 12);

                        extra_num_clipped_ += popcount(extra_clipped_mask);
                    }
                }
            }

#else

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputDimensions * b;
                for (IndexType i = 0; i < kOutputDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    const bool clipped = (output_[index] <= kZero) | (output_[index] >= kOne);
                    gradients_[index] = gradients[index] * !clipped;
                    num_clipped_ += clipped;
                }
                for (IndexType i = kOutputDimensions; i < kOutputTotalDimensions; ++i) {
                    const IndexType index = batch_offset + i;
                    const bool clipped = (output_[index] <= kZero) | (output_[index] >= kOne);
                    gradients_[index] = gradients[index] * !clipped;
                    extra_num_clipped_ += clipped;
                }
            }

#endif

            num_total_ += batch_->size() * kOutputDimensions;
            extra_num_total_ += batch_->size() * kOutputExtraDimensions;

            // Since the weight matrix updates only the columns corresponding to the features that appeared in the input,
            // Correct the learning rate and adjust the scale without using momentum
            const LearnFloatType effective_learning_rate =
                static_cast<LearnFloatType>(local_learning_rate / (1.0 - momentum_));

#if defined(USE_BLAS)

            cblas_sscal(
                kHalfDimensions, momentum_, biases_diff_, 1
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputTotalDimensions * b;
                for (IndexType c = 0; c < 2; ++c) {
                    const IndexType output_offset = batch_offset + kHalfDimensions * c;
                    cblas_saxpy(
                        kHalfDimensions, 1.0,
                        &gradients_[output_offset], 1, biases_diff_, 1
                    );
                }
            }

            cblas_saxpy(
                kHalfDimensions, -local_learning_rate,
                biases_diff_, 1, biases_, 1
            );

#else

            Blas::sscal(
                thread_pool,
                kHalfDimensions, momentum_, biases_diff_, 1
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputTotalDimensions * b;
                for (IndexType c = 0; c < 2; ++c) {
                    const IndexType output_offset = batch_offset + kHalfDimensions * c;
                    Blas::saxpy(
                        thread_pool,
                        kHalfDimensions, 1.0,
                        &gradients_[output_offset], 1, biases_diff_, 1
                    );
                }
            }

            Blas::saxpy(
                thread_pool,
                kHalfDimensions, -local_learning_rate,
                biases_diff_, 1, biases_, 1
            );

#endif

#if defined(USE_BLAS)

            cblas_sscal(
                kHalfExtraDimensions, momentum_, extra_biases_diff_, 1
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputTotalDimensions * b;
                for (IndexType c = 0; c < 2; ++c) {
                    const IndexType output_offset = batch_offset + kOutputDimensions + kHalfExtraDimensions * c;
                    cblas_saxpy(
                        kHalfExtraDimensions, 1.0,
                        &gradients_[output_offset], 1, extra_biases_diff_, 1
                    );
                }
            }

            cblas_saxpy(
                kHalfExtraDimensions, -local_learning_rate,
                extra_biases_diff_, 1, extra_biases_, 1
            );

#else

            Blas::sscal(
                thread_pool,
                kHalfExtraDimensions, momentum_, extra_biases_diff_, 1
            );

            for (IndexType b = 0; b < batch_->size(); ++b) {
                const IndexType batch_offset = kOutputTotalDimensions * b;
                for (IndexType c = 0; c < 2; ++c) {
                    const IndexType output_offset = batch_offset + kOutputDimensions + kHalfExtraDimensions * c;
                    Blas::saxpy(
                        thread_pool,
                        kHalfExtraDimensions, 1.0,
                        &gradients_[output_offset], 1, extra_biases_diff_, 1
                    );
                }
            }

            Blas::saxpy(
                thread_pool,
                kHalfExtraDimensions, -local_learning_rate,
                extra_biases_diff_, 1, extra_biases_, 1
            );

#endif

            thread_pool.execute_with_workers(
                [&, num_threads = thread_pool.size()](Thread& th) {
                    const auto thread_index = th.thread_idx();

                    for (IndexType b = 0; b < batch_->size(); ++b) {
                        const IndexType batch_offset = kOutputTotalDimensions * b;
                        for (IndexType c = 0; c < 2; ++c) {
                            const IndexType output_offset = batch_offset + kHalfDimensions * c;
                            for (const auto& feature : (*batch_)[b].training_features[c]) {
                                const IndexType feature_index = feature.get_index();

                                // We assign each bucket a continuous range of bits at least
                                // of cache line size to prevent false sharing.
                                // For HalfKP this is enough to saturate about 80 threads.
                                const IndexType thread_bucket =
                                    (feature_index / BitsetType::best_concurrent_access_stride)
                                    % num_threads;

                                if (thread_bucket != thread_index)
                                    continue;

                                // This operation can be performed safely because
                                // each thread accesses a different memory location
                                // (even a different cache line)
                                observed_features.set(feature_index);

                                const IndexType weights_offset =
                                    kHalfDimensions * feature_index;

                                const auto scale = static_cast<LearnFloatType>(
                                    effective_learning_rate / feature.get_count());

#if defined (USE_BLAS)

                                cblas_saxpy(
                                    kHalfDimensions, -scale,
                                    &gradients_[output_offset], 1,
                                    &weights_[weights_offset], 1
                                );

#else

                                Blas::saxpy(
                                    kHalfDimensions, -scale,
                                    &gradients_[output_offset], 1,
                                    &weights_[weights_offset], 1
                                );

#endif
                            }
                        }
                    }

                    for (IndexType b = 0; b < batch_->size(); ++b) {
                        const IndexType batch_offset = kOutputTotalDimensions * b;
                        for (IndexType c = 0; c < 2; ++c) {
                            const IndexType output_offset = batch_offset + kOutputDimensions + kHalfExtraDimensions * c;
                            for (const auto& feature : (*batch_)[b].extra_training_features[c]) {
                                const IndexType feature_index = feature.get_index();

                                // We assign each bucket a continuous range of bits at least
                                // of cache line size to prevent false sharing.
                                // For HalfKP this is enough to saturate about 80 threads.
                                const IndexType thread_bucket =
                                    (feature_index / ExtraBitsetType::best_concurrent_access_stride)
                                    % num_threads;

                                if (thread_bucket != thread_index)
                                    continue;

                                // This operation can be performed safely because
                                // each thread accesses a different memory location
                                // (even a different cache line)
                                extra_observed_features.set(feature_index);

                                const IndexType weights_offset =
                                    kHalfExtraDimensions * feature_index;

                                const auto scale = static_cast<LearnFloatType>(
                                    effective_learning_rate / feature.get_count());

#if defined (USE_BLAS)

                                cblas_saxpy(
                                    kHalfExtraDimensions, -scale,
                                    &gradients_[output_offset], 1,
                                    &extra_weights_[weights_offset], 1
                                );

#else

                                Blas::saxpy(
                                    kHalfExtraDimensions, -scale,
                                    &gradients_[output_offset], 1,
                                    &extra_weights_[weights_offset], 1
                                );

#endif
                            }
                        }
                    }
                }
            );

            thread_pool.wait_for_workers_finished();
        }

    private:
        // constructor
        Trainer(LayerType* target_layer) :
            batch_(nullptr),
            target_layer_(target_layer),
            biases_(),
            weights_(),
            biases_diff_(),
            momentum_(0.2),
            learning_rate_scale_(1.0) {

            dequantize_parameters();
        }

        // Weight saturation and parameterization
        void quantize_parameters() {
            for (IndexType i = 0; i < kHalfDimensions; ++i) {
                target_layer_->biases_[i] =
                    round<typename LayerType::BiasType>(biases_[i] * kBiasScale);
            }
            for (IndexType i = 0; i < kHalfExtraDimensions; ++i) {
                target_layer_->extra_biases_[i] =
                    round<typename LayerType::BiasType>(extra_biases_[i] * kBiasScale);
            }

            std::vector<TrainingFeature> training_features;

            Threads.for_each_index_with_workers(
                0, RawFeatures::kDimensions,
                [this, training_features](Thread&, int j) mutable {
                    training_features.clear();
                    Features::Factorizer<RawFeatures>::append_training_features(
                        j, &training_features);

                    for (IndexType i = 0; i < kHalfDimensions; ++i) {
                        double sum = 0.0;
                        for (const auto& feature : training_features) {
                            sum += weights_[kHalfDimensions * feature.get_index() + i];
                        }

                        target_layer_->weights_[kHalfDimensions * j + i] =
                            round<typename LayerType::WeightType>(sum * kWeightScale);
                    }
                }
            );
            Threads.wait_for_workers_finished();

            std::vector<TrainingFeature> extra_training_features;
            Threads.for_each_index_with_workers(
                0, RawExtraFeatures::kDimensions,
                [this, extra_training_features](Thread&, int j) mutable {
                    extra_training_features.clear();
                    Features::Factorizer<RawExtraFeatures>::append_training_features(
                        j, &extra_training_features);

                    for (IndexType i = 0; i < kHalfExtraDimensions; ++i) {
                        double sum = 0.0;
                        for (const auto& feature : extra_training_features) {
                            sum += extra_weights_[kHalfExtraDimensions * feature.get_index() + i];
                        }

                        target_layer_->extra_weights_[kHalfExtraDimensions * j + i] =
                            round<typename LayerType::WeightType>(sum * kWeightScale);
                    }
                }
            );
            Threads.wait_for_workers_finished();
        }

        void reset_stats() {
            min_pre_activation_ = std::numeric_limits<LearnFloatType>::max();
            max_pre_activation_ = std::numeric_limits<LearnFloatType>::lowest();
            min_extra_pre_activation_ = std::numeric_limits<LearnFloatType>::max();
            max_extra_pre_activation_ = std::numeric_limits<LearnFloatType>::lowest();

            std::fill(std::begin(min_activations_), std::end(min_activations_),
                      std::numeric_limits<LearnFloatType>::max());
            std::fill(std::begin(max_activations_), std::end(max_activations_),
                      std::numeric_limits<LearnFloatType>::lowest());
            std::fill(std::begin(min_extra_activations_), std::end(min_extra_activations_),
                      std::numeric_limits<LearnFloatType>::max());
            std::fill(std::begin(max_extra_activations_), std::end(max_extra_activations_),
                      std::numeric_limits<LearnFloatType>::lowest());

            num_clipped_ = 0;
            num_total_ = 0;
            extra_num_clipped_ = 0;
            extra_num_total_ = 0;
        }

        // read parameterized integer
        void dequantize_parameters() {
            for (IndexType i = 0; i < kHalfDimensions; ++i) {
                biases_[i] = static_cast<LearnFloatType>(
                    target_layer_->biases_[i] / kBiasScale);
            }
            for (IndexType i = 0; i < kHalfExtraDimensions; ++i) {
                extra_biases_[i] = static_cast<LearnFloatType>(
                    target_layer_->extra_biases_[i] / kBiasScale);
            }

            std::fill(std::begin(weights_), std::end(weights_), +kZero);
            std::fill(std::begin(extra_weights_), std::end(extra_weights_), +kZero);

            for (IndexType i = 0; i < kHalfDimensions * RawFeatures::kDimensions; ++i) {
                weights_[i] = static_cast<LearnFloatType>(
                    target_layer_->weights_[i] / kWeightScale);
            }
            for (IndexType i = 0; i < kHalfExtraDimensions * RawExtraFeatures::kDimensions; ++i) {
                extra_weights_[i] = static_cast<LearnFloatType>(
                    target_layer_->extra_weights_[i] / kWeightScale);
            }

            std::fill(std::begin(biases_diff_), std::end(biases_diff_), +kZero);
            std::fill(std::begin(extra_biases_diff_), std::end(extra_biases_diff_), +kZero);

            reset_stats();
        }

        // Set the weight corresponding to the feature that does not appear in the learning data to 0
        void clear_unobserved_feature_weights() {
            for (IndexType i = 0; i < kInputDimensions; ++i) {
                if (!observed_features.test(i)) {
                    std::fill(std::begin(weights_) + kHalfDimensions * i,
                              std::begin(weights_) + kHalfDimensions * (i + 1), +kZero);
                }
            }
            for (IndexType i = 0; i < kInputExtraDimensions; ++i) {
                if (!extra_observed_features.test(i)) {
                    std::fill(std::begin(extra_weights_) + kHalfExtraDimensions * i,
                              std::begin(extra_weights_) + kHalfExtraDimensions * (i + 1), +kZero);
                }
            }

            quantize_parameters();
        }

        // Check if there are any problems with learning
        void check_health() {

            constexpr LearnFloatType kPreActivationLimit =
                std::numeric_limits<typename LayerType::WeightType>::max() /
                kWeightScale;

            const auto largest_min_activation = *std::max_element(
                std::begin(min_activations_), std::end(min_activations_));
            const auto smallest_max_activation = *std::min_element(
                std::begin(max_activations_), std::end(max_activations_));
            const auto largest_extra_min_activation = *std::max_element(
                std::begin(min_extra_activations_), std::end(min_extra_activations_));
            const auto smallest_extra_max_activation = *std::min_element(
                std::begin(max_extra_activations_), std::end(max_extra_activations_));

            double abs_bias_sum = 0.0;
            double abs_weight_sum = 0.0;
            double abs_extra_bias_sum = 0.0;
            double abs_extra_weight_sum = 0.0;

            for(auto b : biases_)
                abs_bias_sum += std::abs(b);

            for(auto b : extra_biases_)
                abs_extra_bias_sum += std::abs(b);

            for(auto w : weights_)
                abs_weight_sum += std::abs(w);

            for(auto w : extra_weights_)
                abs_extra_weight_sum += std::abs(w);

            auto out = sync_region_cout.new_region();

            out << "INFO (check_health):"
                << " layer " << LayerType::kLayerIndex
                << " - " << LayerType::get_name()
                << std::endl;

            out << "  - observed " << observed_features.count()
                << " (out of " << kInputDimensions << ") features"
                << std::endl;

            out << "  - observed " << extra_observed_features.count()
                << " (out of " << kInputExtraDimensions << ") extra features"
                << std::endl;

            out << "  - (min, max) of pre-activations = "
                << min_pre_activation_ << ", "
                << max_pre_activation_ << " (limit = "
                << kPreActivationLimit << ")"
                << std::endl;

            out << "  - (min, max) of extra pre-activations = "
                << min_extra_pre_activation_ << ", "
                << max_extra_pre_activation_ << " (limit = "
                << kPreActivationLimit << ")"
                << std::endl;

            out << "  - largest min activation = " << largest_min_activation
                << " , smallest max activation = " << smallest_max_activation
                << std::endl;

            out << "  - largest extra min activation = " << largest_extra_min_activation
                << " , smallest extra max activation = " << smallest_extra_max_activation
                << std::endl;

            out << "  - avg_abs_bias         = " << abs_bias_sum / std::size(biases_) << std::endl;
            out << "  - avg_abs_extra_bias   = " << abs_extra_bias_sum / std::size(extra_biases_) << std::endl;
            out << "  - avg_abs_weight       = " << abs_weight_sum / std::size(weights_) << std::endl;
            out << "  - avg_abs_extra_weight = " << abs_extra_weight_sum / std::size(extra_weights_) << std::endl;

            out << "  - clipped " << static_cast<double>(num_clipped_) / num_total_ * 100.0 << "% of outputs"
                << std::endl;

            out << "  - clipped " << static_cast<double>(extra_num_clipped_) / extra_num_total_ * 100.0 << "% of extra outputs"
                << std::endl;

            out.unlock();

            reset_stats();
        }

        // number of input/output dimensions
        static constexpr IndexType kInputDimensions =
            Features::Factorizer<RawFeatures>::get_dimensions();
        static constexpr IndexType kInputExtraDimensions =
            Features::Factorizer<RawExtraFeatures>::get_dimensions();
        static constexpr IndexType kOutputDimensions = LayerType::kOutputDimensions;
        static constexpr IndexType kOutputExtraDimensions = LayerType::kOutputExtraDimensions;
        static constexpr IndexType kOutputTotalDimensions = LayerType::kOutputTotalDimensions;
        static constexpr IndexType kHalfDimensions = LayerType::kHalfDimensions;
        static constexpr IndexType kHalfExtraDimensions = LayerType::kHalfExtraDimensions;

        // Coefficient used for parameterization
        static constexpr LearnFloatType kActivationScale =
            std::numeric_limits<std::int8_t>::max();
        static constexpr LearnFloatType kBiasScale = kActivationScale;
        static constexpr LearnFloatType kWeightScale = kActivationScale;

        // LearnFloatType constant
        static constexpr LearnFloatType kZero = static_cast<LearnFloatType>(0.0);
        static constexpr LearnFloatType kOne = static_cast<LearnFloatType>(1.0);

        // mini batch
        const std::vector<Example>* batch_;

        // layer to learn
        LayerType* const target_layer_;

        IndexType num_clipped_;
        IndexType num_total_;
        IndexType extra_num_clipped_;
        IndexType extra_num_total_;

        // parameter
        alignas(kCacheLineSize) LearnFloatType biases_[kHalfDimensions];
        alignas(kCacheLineSize) LearnFloatType extra_biases_[kHalfExtraDimensions];
        alignas(kCacheLineSize)
            LearnFloatType weights_[kHalfDimensions * kInputDimensions];
        alignas(kCacheLineSize)
            LearnFloatType extra_weights_[kHalfExtraDimensions * kInputExtraDimensions];

        // Buffer used for updating parameters
        alignas(kCacheLineSize) LearnFloatType biases_diff_[kHalfDimensions];
        alignas(kCacheLineSize) LearnFloatType extra_biases_diff_[kHalfExtraDimensions];
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> gradients_;

        // Forward propagation buffer
        std::vector<LearnFloatType, CacheLineAlignedAllocator<LearnFloatType>> output_;

        // Features that appeared in the training data
        using BitsetType = LargeBitset<kInputDimensions>;
        BitsetType observed_features;
        using ExtraBitsetType = LargeBitset<kInputExtraDimensions>;
        ExtraBitsetType extra_observed_features;

        // hyper parameter
        LearnFloatType momentum_;
        LearnFloatType learning_rate_scale_;

        // Health check statistics
        LearnFloatType min_pre_activation_;
        LearnFloatType max_pre_activation_;
        LearnFloatType min_extra_pre_activation_;
        LearnFloatType max_extra_pre_activation_;
        alignas(kCacheLineSize) LearnFloatType min_activations_[kHalfDimensions];
        alignas(kCacheLineSize) LearnFloatType max_activations_[kHalfDimensions];
        alignas(kCacheLineSize) LearnFloatType min_extra_activations_[kHalfExtraDimensions];
        alignas(kCacheLineSize) LearnFloatType max_extra_activations_[kHalfExtraDimensions];
    };

}  // namespace Eval::NNUE

#endif
