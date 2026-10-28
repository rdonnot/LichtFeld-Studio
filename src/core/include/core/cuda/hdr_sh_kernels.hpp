/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// The per-direction half of the HDR SH projection (core/hdr_sh.hpp). `values` is a CUDA buffer laid out
// [n_splats * 3, directions] (row = splat * 3 + channel): the SH sum b evaluated on the sphere. In place:
//   mode 0  linear radiance:  v = v < ln(color_max) ? exp(v) : color_max   (== fastgs log_radiance_color)
//   mode 1  log:              v = ln(max(v, floor)), floor = max(1e-6, 1e-2 * dc[splat*3+ch]) when `dc` is given
//                             (the splat's DC radiance per channel, stride dc_stride floats between channels):
//                             a linear expansion that goes negative on part of the sphere must not drag the
//                             log fit down to ln 1e-6 = -13.8; 1 % of the splat's own mean keeps the fit near
//                             the positive side (docs/hdr.md)
//   mode 2  bake:             rgb = srgb_oetf(tone(exposure * linear(v))) clamped to [0, 1]
//   mode 3  display:          as mode 2 but v is ALREADY linear radiance (a rendered image: n_splats = pixels,
//                             directions = 1, HWC interleaved)
// color_max <= 0 means no cap. `tone` indexes the viewer's display_color.inc operators (0 = none);
// tone 7 (OCIO) samples the baked display LUT `ocio_lut` (device memory, core/ocio.hpp Lut3D layout,
// lut_size^3 nodes over the log2 shaper lut_lo..lut_hi) on exposure * linear(v) — null falls back to tone 4.
namespace lfs::core::hdr::cuda {

    struct OcioLutRef {
        const float* data = nullptr;
        int size = 0;
        float log2_lo = 0.0f;
        float log2_hi = 0.0f;
    };

    void apply_per_direction(float* values, std::size_t n_splats, std::size_t directions, int mode,
                             float color_max, float exposure, int tone, cudaStream_t stream,
                             const float* dc = nullptr, std::size_t dc_stride = 1, OcioLutRef ocio_lut = {});

} // namespace lfs::core::hdr::cuda
