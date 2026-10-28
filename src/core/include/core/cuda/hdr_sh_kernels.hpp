/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

// The per-direction half of the HDR SH projection (core/hdr_sh.hpp). `values` is a CUDA buffer laid out
// [n_splats * 3, directions] (row = splat * 3 + channel): the SH sum b evaluated on the sphere. In place:
//   mode 0  linear radiance:  v = v < ln(color_max) ? exp(v) : color_max   (== fastgs log_radiance_color)
//   mode 1  log:              v = ln(max(v, 1e-6))
//   mode 2  bake:             rgb = srgb_oetf(tone(exposure * linear(v))) clamped to [0, 1]
//   mode 3  display:          as mode 2 but v is ALREADY linear radiance (a rendered image: n_splats = pixels,
//                             directions = 1, HWC interleaved)
// color_max <= 0 means no cap. `tone` indexes the viewer's display_color.inc operators (0 = none).
namespace lfs::core::hdr::cuda {

    void apply_per_direction(float* values, std::size_t n_splats, std::size_t directions, int mode,
                             float color_max, float exposure, int tone, cudaStream_t stream);

} // namespace lfs::core::hdr::cuda
