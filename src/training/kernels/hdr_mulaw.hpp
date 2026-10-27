/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::training::kernels {

    // HDR photometric domain (PLAN.md M2): the loss compares mu-law images of the P99-normalised radiance,
    //     x = v / scale,  y = log1p(mu * max(x, 0)) / log1p(mu)  in [0, 1] for x in [0, 1].
    // Forward returns y as a new Float32 tensor of v's shape; x_saved (optional) receives the SIGNED x so the
    // backward can zero the gradient where v < 0 (a render below black carries no mu-law gradient).
    LFS_CUDA_API lfs::core::Tensor hdr_mulaw_forward(
        const lfs::core::Tensor& v, float scale, float mu, lfs::core::Tensor* x_saved = nullptr);

    // In place: grad *= mu / ((1 + mu * x) * log1p(mu) * scale) for x >= 0, grad = 0 for x < 0.
    LFS_CUDA_API void hdr_mulaw_backward_(
        lfs::core::Tensor& grad, const lfs::core::Tensor& x_saved, float scale, float mu);

} // namespace lfs::training::kernels
