/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_color_chain_cuda.hpp"
#include "kernel_utils.cuh"

#include <cuda_runtime.h>
#include <type_traits>
#include <vector>

namespace hdr_color_chain_test {
    namespace {
        using fast_lfs::rasterization::kernels::convert_sh_to_color;
        using fast_lfs::rasterization::kernels::convert_sh_to_color_backward_grads;
        using fast_lfs::rasterization::kernels::log_radiance_color;
        using fast_lfs::rasterization::kernels::shAt;
        using fast_lfs::rasterization::kernels::shSlotsForBases;

        template <int ACTIVE_SH_BASES>
        __global__ void chain_kernel(const float3* sh0, const float4* shN, const float3* means, const float3 cam,
                                     const float3* grad_c, const int n, const uint slots, const float color_max,
                                     const float log_color_max, float3* b_out, float3* ldr_out, float3* hdr_out,
                                     float3* grad_color_helper, float3* dsh0_out) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n)
                return;
            const uint p = static_cast<uint>(i);
            // Forward: the same call the preprocess kernel makes, then the activation step of each instantiation.
            const float3 b = convert_sh_to_color(sh0, shN, means[p], cam, p, ACTIVE_SH_BASES, slots);
            b_out[p] = b;
            ldr_out[p] = b; // LOG_RADIANCE = false: the preprocess kernel stores the sum unchanged
            const float3 c = log_radiance_color(b, color_max, log_color_max);
            hdr_out[p] = c;
            // Backward, as the blend kernel does it: cap mask (strict in HDR) and dL/db = c * dL/dc.
            const float3 g = grad_c[p];
            grad_color_helper[p] = make_float3(c.x < color_max ? c.x * g.x : 0.0f,
                                               c.y < color_max ? c.y * g.y : 0.0f,
                                               c.z < color_max ? c.z * g.z : 0.0f);
            float sh0_grads[3];
            convert_sh_to_color_backward_grads<ACTIVE_SH_BASES>(shN, grad_color_helper, means[p], cam, p, p, slots,
                                                                 sh0_grads);
            dsh0_out[p] = make_float3(sh0_grads[0], sh0_grads[1], sh0_grads[2]);
        }

        template <typename T>
        struct DeviceArray {
            T* ptr = nullptr;
            ~DeviceArray() {
                if (ptr)
                    cudaFree(ptr);
            }
            bool upload(const std::vector<T>& host) {
                if (cudaMalloc(&ptr, sizeof(T) * (host.empty() ? 1 : host.size())) != cudaSuccess)
                    return false;
                return host.empty() ||
                       cudaMemcpy(ptr, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice) == cudaSuccess;
            }
            bool alloc(size_t count) {
                return cudaMalloc(&ptr, sizeof(T) * (count == 0 ? 1 : count)) == cudaSuccess;
            }
            bool download(std::vector<T>& host, size_t count) const {
                host.resize(count);
                return count == 0 ||
                       cudaMemcpy(host.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost) == cudaSuccess;
            }
        };
    } // namespace

    bool eval_sh_color_chain(const ShChainInput& in, ShChainOutput& out) {
        const int n = in.n;
        if (n <= 0)
            return false;
        const uint slots = shSlotsForBases(static_cast<uint>(in.active_sh_bases));
        // Pack the canonical rest coefficients into the fp32 float4 swizzle the kernels read:
        // coefficient k channel ch -> linear idx = k*3+ch -> float4 slot idx/4, component idx%4, at shAt(p, slot).
        const size_t blocks = (static_cast<size_t>(n) + 31) / 32;
        std::vector<float> packed(blocks * static_cast<size_t>(slots) * 32 * 4, 0.0f);
        for (int p = 0; p < n; ++p) {
            for (int k = 0; k < 15; ++k) {
                for (int ch = 0; ch < 3; ++ch) {
                    const uint idx = static_cast<uint>(k * 3 + ch);
                    const uint slot = idx / 4u;
                    if (slot >= slots)
                        continue;
                    packed[static_cast<size_t>(shAt(static_cast<uint>(p), slot, slots)) * 4 + (idx % 4u)] =
                        in.shN[(static_cast<size_t>(p) * 15 + k) * 3 + ch];
                }
            }
        }
        DeviceArray<float> sh0, shN, means, grad_c, b, ldr, hdr, helper, dsh0;
        if (!sh0.upload(in.sh0) || !shN.upload(packed) || !means.upload(in.means) || !grad_c.upload(in.grad_c) ||
            !b.alloc(n * 3) || !ldr.alloc(n * 3) || !hdr.alloc(n * 3) || !helper.alloc(n * 3) || !dsh0.alloc(n * 3))
            return false;
        const float3 cam = make_float3(in.cam[0], in.cam[1], in.cam[2]);
        const float log_color_max = logf(in.color_max);
        const dim3 grid((n + 127) / 128), block(128);
        auto launch = [&](auto bases) {
            constexpr int B = decltype(bases)::value;
            chain_kernel<B><<<grid, block>>>(
                reinterpret_cast<const float3*>(sh0.ptr), reinterpret_cast<const float4*>(shN.ptr),
                reinterpret_cast<const float3*>(means.ptr), cam, reinterpret_cast<const float3*>(grad_c.ptr), n,
                slots, in.color_max, log_color_max, reinterpret_cast<float3*>(b.ptr),
                reinterpret_cast<float3*>(ldr.ptr), reinterpret_cast<float3*>(hdr.ptr),
                reinterpret_cast<float3*>(helper.ptr), reinterpret_cast<float3*>(dsh0.ptr));
        };
        switch (in.active_sh_bases) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 9: launch(std::integral_constant<int, 9>{}); break;
        case 16: launch(std::integral_constant<int, 16>{}); break;
        default: return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess)
            return false;
        const size_t count = static_cast<size_t>(n) * 3;
        return b.download(out.b, count) && ldr.download(out.ldr_color, count) && hdr.download(out.hdr_color, count) &&
               dsh0.download(out.hdr_dL_dsh0, count);
    }

} // namespace hdr_color_chain_test
