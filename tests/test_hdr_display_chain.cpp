/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_hdr_display_chain.cpp
 * @brief HDR fork (PLAN.md M4): the viewer's HDR display chain, through the C++ twin of
 *        display_color.inc (the same text the Slang projection shader and the GLSL compose pass compile):
 *        normalised radiance * 2^EV -> tone operator -> sRGB OETF, the SH activation that feeds it, and
 *        the false-colour ramp, against hand-computed values. CPU only.
 */

#include "rendering/display_color.hpp"

#include "core/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

    constexpr float kShC0 = 0.28209479177387814f;

    // The projection shader's HDR activation (spherical_harmonics.slang compute_hdr_color_from_sh_coeffs)
    // on the DC term alone: exp(min(0.5 + SH_C0 * sh0 - ln s, ln(cmax / s))).
    float hdr_dc_color(const float sh0, const float scale, const float color_max) {
        const float b = 0.5f + kShC0 * sh0 - std::log(scale);
        return std::exp(std::min(b, std::log(color_max) - std::log(scale)));
    }

    float srgb_oetf_ref(const double c) {
        const double x = std::clamp(c, 0.0, 1.0);
        return static_cast<float>(x <= 0.0031308 ? 12.92 * x : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055);
    }

    TEST(HdrDisplayChainTest, ShActivationIsRadianceOverScaleAndCapped) {
        const float s = 2421.4f, cmax = 1000.0f * s;
        // A splat trained to the scene's P99 white: ln(s) = 0.5 + SH_C0 * sh0  ->  normalised radiance 1.
        const float sh0_white = (std::log(s) - 0.5f) / kShC0;
        EXPECT_NEAR(hdr_dc_color(sh0_white, s, cmax), 1.0f, 1e-5f);
        // Four stops above white.
        const float sh0_bright = (std::log(16.0f * s) - 0.5f) / kShC0;
        EXPECT_NEAR(hdr_dc_color(sh0_bright, s, cmax), 16.0f, 1e-3f);
        // Far above the cap: exactly the cap in normalised units (the blend clamp uses the same number).
        const float sh0_sun = (std::log(1e9f) - 0.5f) / kShC0;
        EXPECT_NEAR(hdr_dc_color(sh0_sun, s, cmax), 1000.0f, 1e-2f);
        // Never negative, however dark.
        EXPECT_GT(hdr_dc_color(-200.0f, s, cmax), 0.0f);
    }

    TEST(HdrDisplayChainTest, SrgbTransferFunctionsRoundTrip) {
        for (const float v : {0.0f, 0.001f, 0.0031308f, 0.01f, 0.18f, 0.5f, 0.9f, 1.0f}) {
            EXPECT_NEAR(lfs::rendering::lfsSrgbOetfChannel(v), srgb_oetf_ref(v), 1e-6f) << v;
            EXPECT_NEAR(lfs::rendering::lfsSrgbEotfChannel(lfs::rendering::lfsSrgbOetfChannel(v)), v, 2e-6f) << v;
        }
        // Hand values: 18 % grey encodes to ~0.4613, 50 % to ~0.7354.
        EXPECT_NEAR(lfs::rendering::lfsSrgbOetfChannel(0.18f), 0.46135f, 2e-4f);
        EXPECT_NEAR(lfs::rendering::lfsSrgbOetfChannel(0.5f), 0.73536f, 2e-4f);
        // Out of range input is clamped, never NaN.
        EXPECT_FLOAT_EQ(lfs::rendering::lfsSrgbOetfChannel(-3.0f), 0.0f);
        EXPECT_FLOAT_EQ(lfs::rendering::lfsSrgbOetfChannel(40.0f), 1.0f);
    }

    TEST(HdrDisplayChainTest, ExposureIsStopsAndNoneIsALinearClamp) {
        using lfs::rendering::lfsDisplayHdr;
        const glm::vec3 white(1.0f);
        // EV 0, tone "none" (0) and "linear" (1): white displays as 1.0 after the OETF.
        EXPECT_NEAR(lfsDisplayHdr(white, 0u, std::exp2(0.0f)).r, 1.0f, 1e-6f);
        EXPECT_NEAR(lfsDisplayHdr(white, 1u, std::exp2(0.0f)).r, 1.0f, 1e-6f);
        // -2 EV: white -> 0.25 linear -> sRGB 0.5371.
        EXPECT_NEAR(lfsDisplayHdr(white, 0u, std::exp2(-2.0f)).r, srgb_oetf_ref(0.25), 1e-5f);
        // +2 EV on a quarter-white: 1.0.
        EXPECT_NEAR(lfsDisplayHdr(glm::vec3(0.25f), 0u, std::exp2(2.0f)).g, 1.0f, 1e-6f);
        // A lamp 6.5 stops over white is clipped at EV 0 and recovered at -8 EV (6.5 - 8 = -1.5 stops).
        const glm::vec3 lamp(std::exp2(6.5f));
        EXPECT_NEAR(lfsDisplayHdr(lamp, 0u, 1.0f).b, 1.0f, 1e-6f);
        EXPECT_NEAR(lfsDisplayHdr(lamp, 0u, std::exp2(-8.0f)).b, srgb_oetf_ref(std::exp2(-1.5)), 1e-5f);
        // Negative radiance (overlay arithmetic) is clamped to black, not NaN.
        const auto dark = lfsDisplayHdr(glm::vec3(-0.1f), 0u, 1.0f);
        EXPECT_FLOAT_EQ(dark.r, 0.0f);
        EXPECT_TRUE(std::isfinite(dark.g));
    }

    TEST(HdrDisplayChainTest, ToneOperatorsSeeExposedLinearRadiance) {
        using lfs::rendering::lfsDisplayHdr;
        using lfs::rendering::lfsDisplayTone;
        // For every curve, the HDR chain equals OETF(lfsDisplayTone(radiance, mode, exposure)): the LDR
        // tone code is reused unchanged, only the input domain (linear radiance) and the OETF differ.
        for (std::uint32_t mode = 1u; mode <= 6u; ++mode) {
            for (const float radiance : {0.02f, 0.18f, 1.0f, 3.0f, 40.0f}) {
                for (const float ev : {-4.0f, 0.0f, 3.0f}) {
                    const glm::vec3 c(radiance, radiance * 0.5f, radiance * 0.25f);
                    const glm::vec3 want = lfs::rendering::lfsSrgbOetf(lfsDisplayTone(c, mode, std::exp2(ev)));
                    const glm::vec3 got = lfsDisplayHdr(c, mode, std::exp2(ev));
                    EXPECT_NEAR(got.r, want.r, 1e-6f) << mode << ' ' << radiance << ' ' << ev;
                    EXPECT_NEAR(got.g, want.g, 1e-6f) << mode << ' ' << radiance << ' ' << ev;
                    EXPECT_NEAR(got.b, want.b, 1e-6f) << mode << ' ' << radiance << ' ' << ev;
                    EXPECT_GE(got.r, 0.0f);
                    EXPECT_LE(got.r, 1.0f);
                }
            }
        }
        // ACES at EV 0 maps white below 1 (it has headroom), filmic likewise; "none" does not.
        EXPECT_LT(lfsDisplayHdr(glm::vec3(1.0f), 4u, 1.0f).r, 1.0f);
        EXPECT_LT(lfsDisplayHdr(glm::vec3(1.0f), 2u, 1.0f).r, 1.0f);
    }

    TEST(HdrDisplayChainTest, TensorOetfMatchesTheShaderOetf) {
        // rendering/hdr_display.hpp srgb_oetf (tensors, used by the point-cloud previews, the headless video
        // render and the exporters) must be the piecewise curve the viewer uses, not a 1/2.2 power: in the
        // shadows the two differ by up to ~8/255.
        std::vector<float> values;
        for (float v = -0.05f; v <= 1.2f; v += 0.0025f)
            values.push_back(v);
        values.insert(values.end(), {0.0f, 0.0031308f, 0.0031309f, 0.04045f, 0.18f, 0.5f, 1.0f});
        const auto linear = lfs::core::Tensor::from_vector(values, {values.size()}, lfs::core::Device::CPU);
        const auto encoded = lfs::rendering::srgb_oetf(linear).cpu().contiguous();
        const float* out = encoded.ptr<float>();
        ASSERT_NE(out, nullptr);
        for (std::size_t i = 0; i < values.size(); ++i) {
            EXPECT_NEAR(out[i], lfs::rendering::lfsSrgbOetfChannel(values[i]), 2e-6f) << values[i];
        }
        // And the difference to the old approximation is real where it matters.
        EXPECT_GT(std::abs(lfs::rendering::lfsSrgbOetfChannel(0.002f) - std::pow(0.002f, 1.0f / 2.2f)) * 255.0f, 4.0f);
    }

    TEST(HdrDisplayChainTest, FalseColorRampIsStopsRelativeToWhite) {
        using lfs::rendering::lfsHdrFalseColor;
        const float ln_cap = std::log(1000.0f);
        // No light: black. White (0 EV): the middle of the ramp. -8 EV: the dark end; +8 EV: the yellow end.
        EXPECT_EQ(lfsHdrFalseColor(0.0f, ln_cap), glm::vec3(0.0f));
        const glm::vec3 mid = lfsHdrFalseColor(1.0f, ln_cap);
        EXPECT_NEAR(mid.r, 0.128f, 1e-3f);
        EXPECT_NEAR(mid.g, 0.567f, 1e-3f);
        const glm::vec3 lo = lfsHdrFalseColor(std::exp2(-8.0f), ln_cap);
        EXPECT_NEAR(lo.r, 0.267f, 1e-3f);
        EXPECT_NEAR(lo.b, 0.329f, 1e-3f);
        const glm::vec3 hi = lfsHdrFalseColor(std::exp2(8.0f), ln_cap);
        EXPECT_NEAR(hi.r, 0.993f, 1e-3f);
        EXPECT_NEAR(hi.g, 0.906f, 1e-3f);
        // Below -8 EV clamps to the dark end, above +8 EV to the bright end (below the cap).
        EXPECT_EQ(lfsHdrFalseColor(std::exp2(-11.0f), ln_cap), lo);
        EXPECT_EQ(lfsHdrFalseColor(std::exp2(9.0f), ln_cap), hi);
        // At the cap: magenta. The test is on LUMINANCE (as the compose pass computes it), so a single
        // channel at the cap with the others dark is not flagged; the cap is a per-channel clamp in the
        // shader, the flag a luminance warning — by design, not an oversight.
        EXPECT_EQ(lfsHdrFalseColor(1000.0f, ln_cap), glm::vec3(1.0f, 0.0f, 1.0f));
        // Monotonic green channel through the ramp's lower half (a sanity check on the segment lookup).
        float previous = -1.0f;
        for (float ev = -8.0f; ev <= 0.0f; ev += 0.5f) {
            const float g = lfsHdrFalseColor(std::exp2(ev), ln_cap).g;
            EXPECT_GE(g, previous) << ev;
            previous = g;
        }
    }

} // namespace
