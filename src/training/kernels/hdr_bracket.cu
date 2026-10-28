/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"
#include "hdr_bracket.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training::kernels {

    namespace {
        constexpr int kThreads = 256;
        constexpr int kCodes = 256;
        constexpr float kLn2 = 0.69314718055994530942f;
        constexpr float kInv255 = 1.0f / 255.0f;

        unsigned int blocks_for(const size_t n) {
            return static_cast<unsigned int>((n + kThreads - 1) / kThreads);
        }

        // Segment k of a strictly increasing g[256] containing u: g[k] <= u < g[k+1], k in [0, 254];
        // k = 0 below g[0] and k = 254 at or above g[255] (the end segments extrapolate).
        __device__ __forceinline__ int lut_segment(const float* __restrict__ g, const float u) {
            int lo = 1, hi = kCodes; // first index j in [1, 255] with g[j] > u, else 256
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (g[mid] > u)
                    hi = mid;
                else
                    lo = mid + 1;
            }
            return min(max(lo - 1, 0), kCodes - 2);
        }

        template <typename GtT>
        __global__ void bracket_forward_kernel(const float* __restrict__ v,
                                               float* __restrict__ z_out,
                                               float* __restrict__ dzdu_out,
                                               const GtT* __restrict__ gt,
                                               const float* __restrict__ g_global,
                                               const float* __restrict__ b,
                                               const float* __restrict__ delta,
                                               const int frame,
                                               const size_t hw,
                                               const size_t n,
                                               const float gate_high,
                                               const float gate_low,
                                               const float gt_scale) {
            __shared__ float g[3 * kCodes];
            for (int i = threadIdx.x; i < 3 * kCodes; i += blockDim.x)
                g[i] = g_global[i];
            __syncthreads();

            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n)
                return;
            const int c = static_cast<int>(i / hw);
            const float* gc = g + c * kCodes;
            const float log2_gain = b[frame] + (delta != nullptr ? delta[frame] : 0.0f);
            const float x = v[i] * exp2f(log2_gain);
            const float u_min = gc[0] - 5.0f;
            float u = x > 0.0f ? logf(x) : u_min;
            bool valid = u > u_min; // below the clamp the stage is flat: no gradient (5 nats under the LUT's floor)
            u = fmaxf(u, u_min);
            const int k = lut_segment(gc, u);
            const float inv_seg = 1.0f / (gc[k + 1] - gc[k]);
            float z = (static_cast<float>(k) + (u - gc[k]) * inv_seg) * kInv255;
            float dzdu = inv_seg * kInv255;
            if (gt != nullptr) {
                // Thresholds are compared in the GT's own units (codes for UInt8) so 250 and 5 are exact.
                const float raw = static_cast<float>(gt[i]);
                const float q = raw * gt_scale;
                if ((raw >= gate_high && z > q) || (raw <= gate_low && z < q)) {
                    z = q;
                    valid = false;
                }
            }
            z_out[i] = z;
            if (dzdu_out != nullptr)
                dzdu_out[i] = valid ? dzdu : 0.0f;
        }

        __global__ void bracket_backward_kernel(float* __restrict__ grad,
                                                const float* __restrict__ v,
                                                const float* __restrict__ dzdu,
                                                float* __restrict__ grad_delta,
                                                const size_t n) {
            const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            float contrib = 0.0f;
            if (i < n) {
                const float d = dzdu[i];
                const float g_z = grad[i];
                const float vi = v[i];
                contrib = g_z * d; // d is 0 where the gate closed or u was clamped
                grad[i] = (vi > 0.0f && d != 0.0f) ? g_z * d / vi : 0.0f;
            }
            if (grad_delta == nullptr)
                return;
            // Block reduction of sum(grad * dzdu): warp shuffles, then one atomic per block.
            for (int offset = 16; offset > 0; offset >>= 1)
                contrib += __shfl_down_sync(0xffffffffu, contrib, offset);
            __shared__ float warp_sums[kThreads / 32];
            const int lane = threadIdx.x & 31;
            const int warp = threadIdx.x >> 5;
            if (lane == 0)
                warp_sums[warp] = contrib;
            __syncthreads();
            if (warp == 0) {
                float total = lane < (kThreads / 32) ? warp_sums[lane] : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1)
                    total += __shfl_down_sync(0xffffffffu, total, offset);
                if (lane == 0 && total != 0.0f)
                    atomicAdd(grad_delta, total * kLn2);
            }
        }

        __global__ void bracket_adam_kernel(float* __restrict__ delta,
                                            float* __restrict__ exp_avg,
                                            float* __restrict__ exp_avg_sq,
                                            float* __restrict__ grad,
                                            const int frame,
                                            const float lr,
                                            const float beta1,
                                            const float beta2,
                                            const float eps,
                                            const float prior,
                                            const float max_delta,
                                            const float bias1,
                                            const float bias2) {
            if (threadIdx.x != 0 || blockIdx.x != 0)
                return;
            const float d = delta[frame];
            const float g = grad[frame] + 2.0f * prior * d;
            const float m = beta1 * exp_avg[frame] + (1.0f - beta1) * g;
            const float s = beta2 * exp_avg_sq[frame] + (1.0f - beta2) * g * g;
            exp_avg[frame] = m;
            exp_avg_sq[frame] = s;
            const float m_hat = m / bias1;
            const float s_hat = s / bias2;
            float next = d - lr * m_hat / (sqrtf(s_hat) + eps);
            next = fminf(fmaxf(next, -max_delta), max_delta);
            delta[frame] = isfinite(next) ? next : d;
            grad[frame] = 0.0f;
        }

        void check_float_cuda(const lfs::core::Tensor& t, const char* what) {
            if (!t.is_valid() || t.device() != lfs::core::Device::CUDA || t.dtype() != lfs::core::DataType::Float32)
                throw std::invalid_argument(std::string("hdr_bracket: ") + what + " must be a Float32 CUDA tensor");
        }
    } // namespace

    lfs::core::Tensor hdr_bracket_forward(const lfs::core::Tensor& v,
                                          const HdrBracketForwardArgs& args,
                                          const lfs::core::Tensor* gt,
                                          lfs::core::Tensor* dzdu) {
        check_float_cuda(v, "render");
        if (v.ndim() != 3 || v.shape()[0] != 3)
            throw std::invalid_argument("hdr_bracket_forward: render must be [3,H,W]");
        if (args.log_irradiance == nullptr || args.b == nullptr)
            throw std::invalid_argument("hdr_bracket_forward: LUT and gains must be set");
        const auto input = v.is_contiguous() ? v : v.contiguous();
        auto z = lfs::core::Tensor::empty_like(input);
        const size_t n = input.numel();
        const size_t hw = n / 3;
        float* dzdu_ptr = nullptr;
        if (dzdu != nullptr) {
            if (!dzdu->is_valid() || dzdu->shape() != input.shape() ||
                dzdu->dtype() != lfs::core::DataType::Float32 || dzdu->device() != lfs::core::Device::CUDA)
                *dzdu = lfs::core::Tensor::empty_like(input);
            dzdu_ptr = dzdu->ptr<float>();
        }
        if (n == 0)
            return z;
        const cudaStream_t stream = input.stream();
        z.set_stream(stream);
        if (dzdu != nullptr)
            dzdu->set_stream(stream);
        lfs::core::Tensor gt_c;
        if (gt != nullptr) {
            if (!gt->is_valid() || gt->device() != lfs::core::Device::CUDA || gt->numel() != n)
                throw std::invalid_argument("hdr_bracket_forward: gt must be a CUDA tensor of the render's shape");
            if (gt->dtype() != lfs::core::DataType::Float32 && gt->dtype() != lfs::core::DataType::UInt8)
                throw std::invalid_argument("hdr_bracket_forward: gt must be Float32 or UInt8");
            gt_c = gt->is_contiguous() ? *gt : gt->contiguous();
            gt_c.sync_to_stream(stream);
        }
        const auto grid = blocks_for(n);
        if (gt_c.is_valid() && gt_c.dtype() == lfs::core::DataType::UInt8) {
            bracket_forward_kernel<unsigned char><<<grid, kThreads, 0, stream>>>(
                input.ptr<float>(), z.ptr<float>(), dzdu_ptr, gt_c.ptr<unsigned char>(), args.log_irradiance,
                args.b, args.delta, args.frame, hw, n, std::round(args.gate_high * 255.0f), std::round(args.gate_low * 255.0f), kInv255);
        } else {
            bracket_forward_kernel<float><<<grid, kThreads, 0, stream>>>(
                input.ptr<float>(), z.ptr<float>(), dzdu_ptr, gt_c.is_valid() ? gt_c.ptr<float>() : nullptr,
                args.log_irradiance, args.b, args.delta, args.frame, hw, n, args.gate_high, args.gate_low, 1.0f);
        }
        LFS_CUDA_LAUNCH_CHECK(stream, "training.hdr_bracket.forward");
        return z;
    }

    void hdr_bracket_backward_(lfs::core::Tensor& grad,
                               const lfs::core::Tensor& v,
                               const lfs::core::Tensor& dzdu,
                               float* grad_delta) {
        check_float_cuda(grad, "grad");
        check_float_cuda(v, "render");
        check_float_cuda(dzdu, "dzdu");
        if (grad.shape() != v.shape() || grad.shape() != dzdu.shape())
            throw std::invalid_argument("hdr_bracket_backward_: grad, render and dzdu must share a shape");
        if (!grad.is_contiguous() || !v.is_contiguous() || !dzdu.is_contiguous())
            throw std::invalid_argument("hdr_bracket_backward_: contiguous tensors required");
        const size_t n = grad.numel();
        if (n == 0)
            return;
        const cudaStream_t stream = grad.stream();
        v.sync_to_stream(stream);
        dzdu.sync_to_stream(stream);
        bracket_backward_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
            grad.ptr<float>(), v.ptr<float>(), dzdu.ptr<float>(), grad_delta, n);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.hdr_bracket.backward");
    }

    void hdr_bracket_adam_step(float* delta, float* exp_avg, float* exp_avg_sq, float* grad,
                               const int frame, const HdrBracketAdamArgs& args, const cudaStream_t stream) {
        if (delta == nullptr || exp_avg == nullptr || exp_avg_sq == nullptr || grad == nullptr)
            throw std::invalid_argument("hdr_bracket_adam_step: null state");
        const int t = std::max(1, args.step);
        const float bias1 = 1.0f - std::pow(args.beta1, static_cast<float>(t));
        const float bias2 = 1.0f - std::pow(args.beta2, static_cast<float>(t));
        bracket_adam_kernel<<<1, 32, 0, stream>>>(delta, exp_avg, exp_avg_sq, grad, frame, args.lr, args.beta1,
                                                  args.beta2, args.eps, args.prior, args.max_delta, bias1, bias2);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.hdr_bracket.adam");
    }

    void hdr_bracket_lut_host(const float* g, const double u_in, double* z_code, double* dz_code_du) {
        const double u_min = static_cast<double>(g[0]) - 5.0;
        const double u = std::max(u_in, u_min);
        int lo = 1, hi = kCodes;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (static_cast<double>(g[mid]) > u)
                hi = mid;
            else
                lo = mid + 1;
        }
        const int k = std::min(std::max(lo - 1, 0), kCodes - 2);
        const double inv_seg = 1.0 / (static_cast<double>(g[k + 1]) - static_cast<double>(g[k]));
        if (z_code)
            *z_code = k + (u - static_cast<double>(g[k])) * inv_seg;
        if (dz_code_du)
            *dz_code_du = u_in > u_min ? inv_seg : 0.0;
    }

} // namespace lfs::training::kernels
