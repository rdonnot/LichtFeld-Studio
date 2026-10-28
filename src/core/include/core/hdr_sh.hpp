/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// HDR SH conversions (PLAN.md M5). The model stores log radiance: in every direction w the fastgs
// kernels read radiance(w) = exp(min(b(w), ln cmax)) with b(w) = 0.5 + SH_C0 * sh0 + sum_i Y_i(w) c_i.
// A PLY for other applications needs the SAME radiance as a standard SH expansion. `exp` of an SH
// expansion is not an SH expansion, so every conversion here is a least-squares projection: the
// per-direction function is sampled on a Fibonacci sphere (D directions, >= 2x the coefficient count:
// 32 / 64 / 128 for degree 1 / 2 / 3) and projected back onto the degree-N real SH basis with the
// precomputed pseudo-inverse. Degree 0 has D = 1 and is exact.
//
// Layouts: sh0 [N, 1, 3] float32; shN CANONICAL [N, K, 3] (K = 3 / 8 / 15), invalid or empty for
// degree 0. Everything runs on CUDA in chunks (the Tensor library's matmul + one kernel).

#include "core/export.hpp"
#include "core/hdr_export_options.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

namespace lfs::core::hdr {

    inline constexpr float kShC0 = 0.28209479177387814f;

    struct ShPair {
        Tensor sh0; // [N, 1, 3]
        Tensor shN; // [N, K, 3] canonical, invalid when K == 0
    };

    // Number of projection directions used for a model of this SH degree (1, 32, 64, 128).
    LFS_CORE_API int projection_direction_count(int sh_degree);

    // Log SH -> linear radiance SH. color_max <= 0 means no cap.
    LFS_CORE_API ShPair log_sh_to_linear(const Tensor& sh0, const Tensor& shN_canonical, float color_max);

    // Linear radiance SH -> log SH: ln(max(radiance(w), 1e-6)) projected back.
    LFS_CORE_API ShPair linear_sh_to_log(const Tensor& sh0, const Tensor& shN_canonical);

    // Log SH -> display-referred sRGB-encoded colour in [0,1] as standard SH:
    // srgb(tone(exposure * exp(min(b, ln cmax)))) per direction, then the projection.
    // tone == Ocio: the display/view/look of `ocio` (empty fields = core::ocio::current()) baked as a LUT.
    LFS_CORE_API ShPair bake_log_sh_to_display(const Tensor& sh0, const Tensor& shN_canonical,
                                               float color_max, float exposure, ToneOperator tone,
                                               const ocio::Selection& ocio = {});

    // Display exposure for an HDR state: 2^ev / s (s = P99 luminance of the training images; the
    // scene's natural exposure is 1/s). An unknown s (0) counts as 1.
    LFS_CORE_API float display_exposure(const SplatData::HdrState& state, float exposure_ev);

    // A deep copy of `source` whose SH are re-expressed through `pair` (sh0 replaced, shN re-swizzled)
    // and whose HDR state is `state`. The copy drops q16 ShN quantisation (fp32 swizzled storage).
    LFS_CORE_API SplatData with_sh(const SplatData& source, const ShPair& pair, const SplatData::HdrState& state);

    // The baked LDR clone every writer can use: an LDR model (is_hdr() == false) that any viewer shows
    // at `exposure_ev` through `tone`. An LDR input is returned as a plain clone.
    LFS_CORE_API SplatData bake_hdr_to_ldr(const SplatData& source, float exposure_ev, ToneOperator tone,
                                           const ocio::Selection& ocio = {});

    // Linear-radiance clone for the PLY writer (is_hdr() == false on the copy: the values are no
    // longer log; the caller writes the lichtfeld_hdr header from the source's state).
    LFS_CORE_API SplatData to_linear_radiance(const SplatData& source);

    // A rendered LINEAR image [3, H, W] (CUDA float) -> display-referred sRGB in [0, 1], same shape:
    // srgb(tone(2^ev / s * radiance)) through the viewer's own curves. What the baked LDR export shows.
    LFS_CORE_API Tensor display_image(const Tensor& linear_chw, const SplatData::HdrState& state, float exposure_ev,
                                      ToneOperator tone, const ocio::Selection& ocio = {});

    // Evaluate the real SH basis in the fastgs ordering/signs for one unit direction:
    // out[0] = 1 (the DC carrier), out[1..K] = Y_1..Y_K. K in {0, 3, 8, 15}.
    LFS_CORE_API void sh_basis(float x, float y, float z, int K, double* out);

} // namespace lfs::core::hdr
