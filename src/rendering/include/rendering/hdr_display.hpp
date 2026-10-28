/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// HDR fork (PLAN.md M4): the display chain for consumers that turn an HDR model's colour into 8-bit
// pixels OUTSIDE the Vulkan splat renderer: point-cloud previews (colour from sh0), the headless video
// render (a fastgs image in absolute radiance) and the exporters (M5). Same maths as display_color.inc's
// lfsDisplayHdr with the "none" tone operator: radiance * 2^EV / s, clamped, the EXACT piecewise sRGB OETF
// (lfsSrgbOetfChannel, pinned by tests/test_hdr_display_chain.cpp). The Vulkan viewer does not use this
// header; rendering/display_color.hpp exposes both sides.

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

    // The sRGB OETF (IEC 61966-2-1) on a tensor of linear values, clamped to [0, 1] first:
    // c <= 0.0031308 ? 12.92 c : 1.055 c^(1/2.4) - 0.055. Element-wise identical to lfsSrgbOetfChannel.
    inline lfs::core::Tensor srgb_oetf(const lfs::core::Tensor& linear) {
        const auto c = linear.clamp(0.0f, 1.0f);
        const auto low = c * 12.92f;
        const auto high = c.pow(1.0f / 2.4f) * 1.055f - 0.055f;
        return lfs::core::Tensor::where(c <= 0.0031308f, low, high);
    }

    // Absolute linear radiance (any shape) -> display RGB in [0, 1], sRGB-encoded.
    inline lfs::core::Tensor hdr_display_image(const lfs::core::Tensor& radiance,
                                               const lfs::core::SplatData::HdrState& hdr,
                                               const float exposure_ev = 0.0f) {
        const float gain = std::exp2(std::clamp(exposure_ev, -20.0f, 20.0f)) / hdr.scale;
        return srgb_oetf(radiance * gain);
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
