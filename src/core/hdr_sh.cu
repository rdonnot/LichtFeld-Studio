/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// The per-direction HDR kernel (core/cuda/hdr_sh_kernels.hpp). The tone operators are the viewer's own:
// display_color.inc is included here as the CUDA twin of the Vulkan compose shader, so a baked LDR export
// shows what the viewer showed at the same exposure and operator.

#include "core/cuda/hdr_sh_kernels.hpp"
#include "core/cuda_error.hpp"

#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

#define LFS_COLOR_INLINE __device__ __forceinline__
#define LFS_COLOR_VEC3 glm::vec3
#define LFS_COLOR_UINT std::uint32_t // one token: the .inc uses it as a functional cast
#define LFS_COLOR_MIX glm::mix
#define LFS_OCIO_LUT_PARAMS const float* __restrict__ lut,
#define LFS_OCIO_LUT_FETCH(i) glm::vec3(lut[3 * (i)], lut[3 * (i) + 1], lut[3 * (i) + 2])
using glm::clamp; // the .inc calls clamp() unqualified; CUDA has no scalar clamp of its own
#include "display_color.inc"
#undef LFS_OCIO_LUT_PARAMS
#undef LFS_OCIO_LUT_FETCH
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

        __global__ void per_direction_kernel(float* __restrict__ values, const std::size_t n_splats,
                                             const std::size_t directions, const int mode, const float color_max,
                                             const float log_color_max, const float exposure, const unsigned int tone,
                                             const float* __restrict__ dc, const std::size_t dc_stride,
                                             const OcioLutRef ocio_lut) {
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
                float fr = 1e-6f, fg = 1e-6f, fb = 1e-6f;
                if (dc != nullptr) {
                    const float* d = dc + splat * 3 * dc_stride;
                    fr = fmaxf(fr, 1e-2f * d[0]);
                    fg = fmaxf(fg, 1e-2f * d[dc_stride]);
                    fb = fmaxf(fb, 1e-2f * d[2 * dc_stride]);
                }
                *r = logf(fmaxf(*r, fr));
                *g = logf(fmaxf(*g, fg));
                *b = logf(fmaxf(*b, fb));
                return;
            }
            glm::vec3 c = mode == 3 ? glm::vec3(*r, *g, *b)
                                    : glm::vec3(linear_radiance(*r, color_max, log_color_max),
                                                linear_radiance(*g, color_max, log_color_max),
                                                linear_radiance(*b, color_max, log_color_max));
            if (mode == 2 || mode == 3) {
                // display_color.inc: the viewer's own HDR chain (exposure -> tone, "none" = "linear" ->
                // sRGB OETF), one definition for the viewer (M4) and this bake (M5).
                if (tone == 7u && ocio_lut.data != nullptr) {
                    // OCIO: the exposure is applied in linear BEFORE the view transform; the LUT holds the view.
                    c = lfsOcioLut(ocio_lut.data, max(c, glm::vec3(0.0f)) * exposure, ocio_lut.size, ocio_lut.log2_lo,
                                   ocio_lut.log2_hi);
                } else {
                    c = lfsDisplayHdr(c, tone == 7u ? 4u : tone, exposure);
                }
            }
            *r = c.x;
            *g = c.y;
            *b = c.z;
        }

    } // namespace

    void apply_per_direction(float* values, const std::size_t n_splats, const std::size_t directions, const int mode,
                             const float color_max, const float exposure, const int tone, cudaStream_t stream,
                             const float* dc, const std::size_t dc_stride, const OcioLutRef ocio_lut) {
        const std::size_t total = n_splats * directions;
        if (total == 0)
            return;
        const float log_color_max = color_max > 0.0f ? std::log(color_max) : INFINITY;
        constexpr int kBlock = 256;
        const unsigned int blocks = static_cast<unsigned int>((total + kBlock - 1) / kBlock);
        per_direction_kernel<<<blocks, kBlock, 0, stream>>>(values, n_splats, directions, mode, color_max,
                                                            log_color_max, exposure,
                                                            static_cast<unsigned int>(tone < 0 ? 0 : tone), dc, dc_stride,
                                                            ocio_lut);
        LFS_CUDA_CHECK(cudaGetLastError());
    }

} // namespace lfs::core::hdr::cuda
