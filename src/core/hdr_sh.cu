/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The per-direction HDR kernel (core/cuda/hdr_sh_kernels.hpp). The tone operators are the viewer's own:
// display_color.inc is included here as the CUDA twin of the Vulkan compose shader, so a baked LDR export
// shows what the viewer showed at the same exposure and operator.

#include "core/cuda/hdr_sh_kernels.hpp"
#include "core/cuda_error.hpp"

#include <cmath>
#include <glm/glm.hpp>

#define LFS_COLOR_INLINE __device__ __forceinline__
#define LFS_COLOR_VEC3 glm::vec3
#define LFS_COLOR_UINT unsigned int
#define LFS_COLOR_MIX glm::mix
#include "display_color.inc"
#undef LFS_COLOR_INLINE
#undef LFS_COLOR_VEC3
#undef LFS_COLOR_UINT
#undef LFS_COLOR_MIX

namespace lfs::core::hdr::cuda {

    namespace {

        __device__ __forceinline__ float linear_radiance(const float b, const float color_max, const float log_color_max) {
            // Same strict comparison as fastgs log_radiance_color: at or above the cap the value IS the cap.
            return b < log_color_max ? expf(b) : color_max;
        }

        __device__ __forceinline__ float srgb_oetf(float x) {
            x = fminf(fmaxf(x, 0.0f), 1.0f);
            return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
        }

        __global__ void per_direction_kernel(float* __restrict__ values, const std::size_t n_splats,
                                             const std::size_t directions, const int mode, const float color_max,
                                             const float log_color_max, const float exposure, const unsigned int tone) {
            const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t total = n_splats * directions;
            if (idx >= total)
                return;
            const std::size_t splat = idx / directions;
            const std::size_t d = idx - splat * directions;
            float* const r = values + (splat * 3 + 0) * directions + d;
            float* const g = values + (splat * 3 + 1) * directions + d;
            float* const b = values + (splat * 3 + 2) * directions + d;

            if (mode == 1) {
                *r = logf(fmaxf(*r, 1e-6f));
                *g = logf(fmaxf(*g, 1e-6f));
                *b = logf(fmaxf(*b, 1e-6f));
                return;
            }
            glm::vec3 c = mode == 3 ? glm::vec3(*r, *g, *b)
                                    : glm::vec3(linear_radiance(*r, color_max, log_color_max),
                                                linear_radiance(*g, color_max, log_color_max),
                                                linear_radiance(*b, color_max, log_color_max));
            if (mode == 2 || mode == 3) {
                // Mode 0 of lfsDisplayTone returns the colour untouched (no exposure); a bake always
                // exposes, so "none" is "linear" here.
                c = lfsDisplayTone(c, tone == 0u ? 1u : tone, exposure);
                c = glm::vec3(srgb_oetf(c.x), srgb_oetf(c.y), srgb_oetf(c.z));
            }
            *r = c.x;
            *g = c.y;
            *b = c.z;
        }

    } // namespace

    void apply_per_direction(float* values, const std::size_t n_splats, const std::size_t directions, const int mode,
                             const float color_max, const float exposure, const int tone, cudaStream_t stream) {
        const std::size_t total = n_splats * directions;
        if (total == 0)
            return;
        const float log_color_max = color_max > 0.0f ? std::log(color_max) : INFINITY;
        constexpr int kBlock = 256;
        const unsigned int blocks = static_cast<unsigned int>((total + kBlock - 1) / kBlock);
        per_direction_kernel<<<blocks, kBlock, 0, stream>>>(values, n_splats, directions, mode, color_max,
                                                            log_color_max, exposure,
                                                            static_cast<unsigned int>(tone < 0 ? 0 : tone));
        LFS_CUDA_CHECK(cudaGetLastError());
    }

} // namespace lfs::core::hdr::cuda
