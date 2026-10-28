/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "rasterization_api.h"
namespace fast_lfs::rasterization {
    // color_max: the forward's blend cap (ColorActivation::color_max); the colours in ctx already carry
    // the activation, this only clamps them the way the blend did.
    void accumulate_pop_scores(const ForwardContext& ctx, int width, int height,
                               const float* background, const float* background_image,
                               double* scores, size_t n_primitives, float color_max, cudaStream_t stream);
}
