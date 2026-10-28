/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "hdr_mulaw.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training::kernels {

    namespace {
        constexpr int kThreads = 256;

        __global__ void mulaw_forward_kernel(const float* __restrict__ v,
                                             float* __restrict__ y,
                                             float* __restrict__ x_saved,
                                             const size_t n,
                                             const float inv_scale,
                                             const float mu,
                                             const float inv_log1p_mu) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const float x = v[i] * inv_scale;
            if (x_saved != nullptr)
                x_saved[i] = x;
            y[i] = x > 0.0f ? log1pf(mu * x) * inv_log1p_mu : 0.0f;
        }

        __global__ void mulaw_backward_kernel(float* __restrict__ grad,
                                              const float* __restrict__ x_saved,
                                              const size_t n,
                                              const float inv_scale,
                                              const float mu,
                                              const float inv_log1p_mu) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const float x = x_saved[i];
            // d y / d v = mu / ((1 + mu x) ln(1 + mu)) * (1 / scale); the clamp at 0 has no gradient below it.
            grad[i] = x >= 0.0f ? grad[i] * (mu * inv_log1p_mu * inv_scale / fmaf(mu, x, 1.0f)) : 0.0f;
        }

        unsigned int blocks_for(const size_t n) {
            return static_cast<unsigned int>((n + kThreads - 1) / kThreads);
        }

        void check_args(const float scale, const float mu) {
            if (!(scale > 0.0f) || !std::isfinite(scale))
                throw std::invalid_argument("hdr_mulaw: scale must be positive and finite");
            if (!(mu > 0.0f) || !std::isfinite(mu))
                throw std::invalid_argument("hdr_mulaw: mu must be positive and finite");
        }
    } // namespace

    lfs::core::Tensor hdr_mulaw_forward(const lfs::core::Tensor& v, const float scale, const float mu,
                                        lfs::core::Tensor* x_saved) {
        check_args(scale, mu);
        if (!v.is_valid() || v.device() != lfs::core::Device::CUDA || v.dtype() != lfs::core::DataType::Float32)
            throw std::invalid_argument("hdr_mulaw_forward: expects a Float32 CUDA tensor");
        const auto input = v.is_contiguous() ? v : v.contiguous();
        auto y = lfs::core::Tensor::empty_like(input);
        float* x_ptr = nullptr;
        if (x_saved != nullptr) {
            if (!x_saved->is_valid() || x_saved->shape() != input.shape() ||
                x_saved->dtype() != lfs::core::DataType::Float32 || x_saved->device() != lfs::core::Device::CUDA)
                *x_saved = lfs::core::Tensor::empty_like(input);
            x_ptr = x_saved->ptr<float>();
        }
        const size_t n = input.numel();
        if (n == 0)
            return y;
        const cudaStream_t stream = input.stream();
        y.set_stream(stream);
        if (x_saved != nullptr)
            x_saved->set_stream(stream);
        mulaw_forward_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
            input.ptr<float>(), y.ptr<float>(), x_ptr, n, 1.0f / scale, mu, 1.0f / std::log1p(mu));
        LFS_CUDA_LAUNCH_CHECK(stream, "training.hdr_mulaw.forward");
        return y;
    }

    void hdr_mulaw_backward_(lfs::core::Tensor& grad, const lfs::core::Tensor& x_saved, const float scale,
                             const float mu) {
        check_args(scale, mu);
        if (!grad.is_valid() || !x_saved.is_valid() || grad.shape() != x_saved.shape() ||
            grad.dtype() != lfs::core::DataType::Float32 || x_saved.dtype() != lfs::core::DataType::Float32 ||
            grad.device() != lfs::core::Device::CUDA || x_saved.device() != lfs::core::Device::CUDA)
            throw std::invalid_argument("hdr_mulaw_backward_: grad and x_saved must be matching Float32 CUDA tensors");
        if (!grad.is_contiguous() || !x_saved.is_contiguous())
            throw std::invalid_argument("hdr_mulaw_backward_: contiguous tensors required");
        const size_t n = grad.numel();
        if (n == 0)
            return;
        const cudaStream_t stream = grad.stream();
        x_saved.sync_to_stream(stream);
        mulaw_backward_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
            grad.ptr<float>(), x_saved.ptr<float>(), n, 1.0f / scale, mu, 1.0f / std::log1p(mu));
        LFS_CUDA_LAUNCH_CHECK(stream, "training.hdr_mulaw.backward");
    }

} // namespace lfs::training::kernels
