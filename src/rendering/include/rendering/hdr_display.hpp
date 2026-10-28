/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// HDR fork (PLAN.md M4): the display chain for consumers that turn an HDR model's colour into 8-bit
// pixels OUTSIDE the Vulkan splat renderer: point-cloud previews (colour from sh0) and the headless
// video render (a fastgs image in absolute radiance). Same maths as display_color.inc's lfsDisplayHdr
// with the "none" tone operator: radiance * 2^EV / s, clamped, sRGB OETF (approximated by 1/2.2 on
// tensors). The Vulkan viewer does not use this header.

#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::rendering {

    inline bool hdr_display_applies(const lfs::core::SplatData& model) {
        const auto& hdr = model.hdr();
        return model.is_hdr() && std::isfinite(hdr.scale) && hdr.scale > 0.0f &&
               std::isfinite(hdr.color_max) && hdr.color_max > 0.0f;
    }

    // Absolute linear radiance (any shape) -> display RGB in [0, 1].
    inline lfs::core::Tensor hdr_display_image(const lfs::core::Tensor& radiance,
                                               const lfs::core::SplatData::HdrState& hdr,
                                               const float exposure_ev = 0.0f) {
        const float gain = std::exp2(std::clamp(exposure_ev, -20.0f, 20.0f)) / hdr.scale;
        return (radiance * gain).clamp(0.0f, 1.0f).pow(1.0f / 2.2f);
    }

    // Per-splat display colour from the DC term alone: exp(min(0.5 + SH_C0 * sh0, ln cmax)) -> the chain above.
    inline lfs::core::Tensor hdr_point_colors(const lfs::core::SplatData& model, const float exposure_ev = 0.0f) {
        constexpr float SH_C0 = 0.28209479177387814f;
        const auto& hdr = model.hdr();
        const auto sh0 = model.sh0_raw().slice(1, 0, 1).squeeze(1);
        const auto radiance = (sh0 * SH_C0 + 0.5f).clamp_max(std::log(hdr.color_max)).exp();
        return hdr_display_image(radiance, hdr, exposure_ev);
    }

} // namespace lfs::rendering
