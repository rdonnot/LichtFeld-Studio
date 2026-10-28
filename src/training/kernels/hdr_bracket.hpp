/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // HDR bracket training (PLAN.md M3, docs/hdr-m3-design.md §3-§4): the fixed Debevec CRF stage.
    //
    //   x = v * 2^(b + delta)            v: linear render [3,H,W], b: the frame's EXIF log2 gain, delta: learnable
    //   u = ln(max(x, 1e-12)), clamped to u >= g_c[0] - 5
    //   g_c[k] <= u < g_c[k+1]  ->  z = (k + (u - g_c[k]) / (g_c[k+1] - g_c[k])) / 255        per channel c
    //
    // Beyond both ends z is extrapolated linearly in u with the end segment's slope — never clamped — so a render
    // the LUT cannot express still receives a gradient. dz/du = (1/255) / (g[k+1] - g[k]) (the end slopes outside),
    // dz/dv = dz/du / v, dz/ddelta = dz/du * ln 2.
    //
    // Saturation gate (training only, gt given): where the JPEG is clipped high (gt >= 250/255) a prediction above
    // it is replaced by gt, where clipped low (gt <= 5/255) a prediction below it is replaced by gt; the backward
    // slope of such a pixel is 0. The loss then sees a one-sided L1 and a consistent image for the SSIM.

    struct HdrBracketForwardArgs {
        const float* log_irradiance = nullptr; // device, [3][256], strictly increasing per channel
        const float* b = nullptr;              // device, per-frame log2 EXIF gain; indexed by frame
        const float* delta = nullptr;          // device, per-frame learnable log2 delta; nullptr = 0 (eval / frozen)
        int frame = 0;
        float gate_high = 250.0f / 255.0f;     // gt >= high: only "x >= x_high" is known (UInt8 gt: code >= high*255)
        float gate_low = 5.0f / 255.0f;        // gt <= low: only "x <= x_low" is known (UInt8 gt: code <= low*255)
    };

    // z (and the masked dz/du when dzdu is given) for v [3,H,W] Float32 CUDA. gt (optional) gates the prediction;
    // Float32 in [0,1] or UInt8 codes, same shape as v. Returns z as a new Float32 tensor.
    LFS_CUDA_API lfs::core::Tensor hdr_bracket_forward(const lfs::core::Tensor& v,
                                                       const HdrBracketForwardArgs& args,
                                                       const lfs::core::Tensor* gt,
                                                       lfs::core::Tensor* dzdu);

    // In place: grad (dL/dz) -> dL/dv = grad * dzdu / v (0 where v <= 0 or the gate closed). When grad_delta is
    // given, sum(grad * dzdu) * ln 2 is ADDED to *grad_delta (device float).
    LFS_CUDA_API void hdr_bracket_backward_(lfs::core::Tensor& grad,
                                            const lfs::core::Tensor& v,
                                            const lfs::core::Tensor& dzdu,
                                            float* grad_delta);

    struct HdrBracketAdamArgs {
        float lr = 2e-3f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float prior = 1.0f;     // loss += prior * delta^2 per VISITED frame (one frame per step): gradient 2 * prior * delta
        float max_delta = 3.0f; // |delta| clamp after the step
        int step = 1;           // this frame's Adam step count (bias correction), >= 1
    };

    // One Adam step on delta[frame] with the prior, then grad[frame] = 0. Runs on `stream`.
    LFS_CUDA_API void hdr_bracket_adam_step(float* delta, float* exp_avg, float* exp_avg_sq, float* grad,
                                            int frame, const HdrBracketAdamArgs& args, cudaStream_t stream);

    // Improvement round B2: Debevec's hat as a per-pixel weight. hat(z) = min(z, 255 - z) / 127.5 on the code z of
    // each channel (1 at mid-range, 0 at both ends; the gate's codes >= 250 / <= 5 weigh <= 0.04); the pixel's
    // weight is the channel mean of hat^p. p = 0 gives 1 everywhere. gt [3,H,W] UInt8 codes or Float32 in [0,1];
    // out becomes [H,W] Float32 CUDA (reallocated when its shape differs).
    LFS_CUDA_API void hdr_bracket_hat_weight(const lfs::core::Tensor& gt, float p, lfs::core::Tensor& out);
    // Host reference: hat(code)^p for one channel sample.
    LFS_CUDA_API double hdr_bracket_hat_host(double code, double p);

    // Host reference of the forward LUT for one channel (the same arithmetic as the kernel, double precision):
    // z in code units (0..255 inside the LUT, extrapolated outside) and dz/du in code units per nat.
    LFS_CUDA_API void hdr_bracket_lut_host(const float* log_irradiance_channel, double u, double* z_code, double* dz_code_du);

} // namespace lfs::training::kernels
