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

#include "core/hdr_export_options.hpp"
#include "core/hdr_sh.hpp"
#include "core/native_fs.hpp"
#include "core/ocio.hpp"
#include "core/tensor.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
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

    // ── OpenColorIO (docs/hdr.md "Colour management") ─────────────────────────────────────────────────────

    // The configs the OCIO tests run on: the built-in ACES studio config always; Houdini 22's CG config when
    // it is on this machine (a second, file-based config with the same colorspace names).
    std::vector<std::pair<std::string, const char*>> ocio_configs() {
        std::vector<std::pair<std::string, const char*>> out{{"", "built-in studio-config-latest"}};
        const char* houdini = "/opt/hfs22.0/packages/ocio/cg-config-v3.0.0_aces-v2.0_ocio-v2.5.ocio";
        if (lfs::core::native_fs::exists(houdini))
            out.emplace_back(houdini, "Houdini 22 CG config");
        return out;
    }

    // The sweep the viewer, the bake and the oracle are compared on: 20 000 normalised radiances, log-uniform
    // over 22 stops (2^-12 .. 2^10) with a random chroma, i.e. every value the HDR chain can meet.
    std::vector<float> radiance_sweep(const std::size_t n = 20000) {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> ev(-12.0f, 10.0f), chroma(0.2f, 1.0f);
        std::vector<float> rgb;
        rgb.reserve(3 * n + 12);
        for (std::size_t i = 0; i < n; ++i) {
            const float y = std::exp2(ev(rng));
            rgb.push_back(y * chroma(rng));
            rgb.push_back(y * chroma(rng));
            rgb.push_back(y * chroma(rng));
        }
        // The edges: black, white, the cap, a negative.
        for (const float v : {0.0f, 1.0f, 1000.0f, -0.5f})
            rgb.insert(rgb.end(), {v, v, v});
        return rgb;
    }

    struct SweepStats {
        double psnr = 0.0;
        double max_abs = 0.0;
    };
    SweepStats compare(const std::vector<float>& a, const std::vector<float>& b) {
        double se = 0.0, max_abs = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = std::clamp<double>(a[i], 0.0, 1.0) - std::clamp<double>(b[i], 0.0, 1.0);
            se += d * d;
            max_abs = std::max(max_abs, std::abs(d));
        }
        const double mse = se / static_cast<double>(a.size());
        return {mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 200.0, max_abs};
    }

    TEST(HdrDisplayChainTest, OcioToneOperatorIsNamedAndListed) {
        using lfs::core::hdr::ToneOperator;
        EXPECT_EQ(lfs::core::hdr::parse_tone_operator("ocio"), ToneOperator::Ocio);
        EXPECT_EQ(lfs::core::hdr::tone_operator_name(ToneOperator::Ocio), "ocio");
        EXPECT_NE(lfs::core::hdr::kToneOperatorNames.find("ocio"), std::string_view::npos);
        std::string error;
        ASSERT_TRUE(lfs::core::ocio::available({}, &error)) << error;
        const auto displays = lfs::core::ocio::displays();
        ASSERT_FALSE(displays.empty());
        ASSERT_FALSE(lfs::core::ocio::views({}, displays.front()).empty());
        // The working space resolves in the ACES configs (alias lin_srgb -> "Linear Rec.709 (sRGB)").
        EXPECT_EQ(lfs::core::ocio::working_space(), "Linear Rec.709 (sRGB)");
        // An unknown display is refused with the list of valid ones, not silently defaulted.
        EXPECT_FALSE(lfs::core::ocio::validate({"", "No Such Display", "", ""}, &error));
        EXPECT_NE(error.find("sRGB - Display"), std::string::npos) << error;
    }

    // The LUT the viewer samples (65^3 over a -14..+8 stop log2 shaper) against OCIO's own CPU processor: at
    // least 45 dB on every (display, view) the fork exercises, on both configs. The C++ twin IS the shader text.
    TEST(HdrDisplayChainTest, OcioLutMatchesTheCpuProcessorWithin45dB) {
        const auto sweep = radiance_sweep();
        for (const auto& [config, config_name] : ocio_configs()) {
            const std::string display = "sRGB - Display";
            for (const char* view : {"ACES 2.0 - SDR 100 nits (Rec.709)", "ACES 1.0 - SDR Video", "Un-tone-mapped", "Raw"}) {
                const lfs::core::ocio::Selection sel{config, display, view, ""};
                std::string error;
                if (!lfs::core::ocio::validate(sel, &error))
                    continue; // the CG config has no ACES 1.0 view; the studio config has both
                const auto lut = lfs::core::ocio::lut_for(sel, lfs::core::ocio::kDefaultLutSize, &error);
                ASSERT_TRUE(lut) << config_name << ' ' << view << ": " << error;
                EXPECT_EQ(lut->size, 65);
                EXPECT_EQ(lut->rgb.size(), 3u * 65u * 65u * 65u);
                std::vector<float> reference = sweep;
                ASSERT_TRUE(lfs::core::ocio::apply_cpu(sel, reference.data(), reference.size() / 3, &error)) << error;
                std::vector<float> got(sweep.size());
                for (std::size_t i = 0; i < sweep.size(); i += 3) {
                    const glm::vec3 c = lfs::core::ocio::sample(*lut, glm::vec3(sweep[i], sweep[i + 1], sweep[i + 2]));
                    got[i] = c.r, got[i + 1] = c.g, got[i + 2] = c.b;
                }
                const auto stats = compare(got, reference);
                EXPECT_GE(stats.psnr, 45.0) << config_name << " / " << view;
                // A tone-mapped view is smooth in log space: <= 2/255 anywhere. "Raw" (linear, clipped at 1) and
                // "Un-tone-mapped" (sRGB-encoded linear, clipped) have a hard knee at 1.0 that a 0.34-stop cell
                // straddles: the worst cell there is 5-11/255 (measured), the PSNR still 50-57 dB.
                const bool unbounded_view = std::string_view(view) == "Raw" || std::string_view(view) == "Un-tone-mapped";
                EXPECT_LE(stats.max_abs, (unbounded_view ? 12.0 : 2.0) / 255.0 + 1e-6) << config_name << " / " << view;
                // Node 0 is exactly f(0): black stays black.
                const glm::vec3 black = lfs::core::ocio::sample(*lut, glm::vec3(0.0f));
                std::vector<float> zero{0.0f, 0.0f, 0.0f};
                ASSERT_TRUE(lfs::core::ocio::apply_cpu(sel, zero.data(), 1, &error));
                EXPECT_NEAR(black.r, std::clamp(zero[0], 0.0f, 1.0f), 1e-6f) << view;
                // A negative input is black, never NaN.
                const glm::vec3 neg = lfs::core::ocio::sample(*lut, glm::vec3(-1.0f));
                EXPECT_TRUE(std::isfinite(neg.r) && neg.r == black.r) << view;
                std::printf("[ ocio ] %s / %s / %s: %.1f dB, max |e| %.2f/255\n", config_name, display.c_str(), view,
                            stats.psnr, stats.max_abs * 255.0);
            }
        }
    }

    // Exposure (EV) is applied in linear light BEFORE the view transform: LUT(x * 2^EV) equals the processor on
    // x * 2^EV, at every EV, within the same bound.
    TEST(HdrDisplayChainTest, OcioExposureIsAppliedBeforeTheView) {
        const lfs::core::ocio::Selection sel{"", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", ""};
        std::string error;
        ASSERT_TRUE(lfs::core::ocio::validate(sel, &error)) << error;
        const auto lut = lfs::core::ocio::lut_for(sel, lfs::core::ocio::kDefaultLutSize, &error);
        ASSERT_TRUE(lut) << error;
        const auto sweep = radiance_sweep(4000);
        for (const float ev : {-4.0f, 0.0f, 3.0f}) {
            std::vector<float> exposed = sweep;
            for (float& v : exposed)
                v *= std::exp2(ev);
            std::vector<float> reference = exposed;
            ASSERT_TRUE(lfs::core::ocio::apply_cpu(sel, reference.data(), reference.size() / 3, &error)) << error;
            std::vector<float> got(exposed.size());
            for (std::size_t i = 0; i < exposed.size(); i += 3) {
                const glm::vec3 c = lfs::core::ocio::sample(*lut, glm::vec3(exposed[i], exposed[i + 1], exposed[i + 2]));
                got[i] = c.r, got[i + 1] = c.g, got[i + 2] = c.b;
            }
            const auto stats = compare(got, reference);
            EXPECT_GE(stats.psnr, 45.0) << ev;
        }
        // And +1 EV on mid grey is brighter than 0 EV through the view (the LUT is monotonic in exposure).
        EXPECT_GT(lfs::core::ocio::sample(*lut, glm::vec3(0.36f)).g, lfs::core::ocio::sample(*lut, glm::vec3(0.18f)).g);
    }

    // The CUDA bake (core::hdr::display_image with tone Ocio: the kernel samples the same LUT with the same text)
    // equals the C++ twin pixel for pixel. One source of truth for the viewer, the bake and the tensor path.
    TEST(HdrDisplayChainTest, OcioCudaDisplayImageEqualsTheCpuTwin) {
        const lfs::core::ocio::Selection sel{"", "sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)", ""};
        std::string error;
        ASSERT_TRUE(lfs::core::ocio::validate(sel, &error)) << error;
        const auto lut = lfs::core::ocio::lut_for(sel, lfs::core::ocio::kDefaultLutSize, &error);
        ASSERT_TRUE(lut) << error;
        constexpr std::size_t H = 64, W = 64;
        const auto sweep = radiance_sweep(H * W - 4); // 3 * H * W floats, interleaved
        // Absolute radiance = normalised * s; display_image divides by s and applies 2^EV.
        const lfs::core::SplatData::HdrState state{true, 2421.4f, 2421400.0f};
        const float ev = 1.5f;
        std::vector<float> chw(3 * H * W);
        for (std::size_t p = 0; p < H * W; ++p)
            for (int c = 0; c < 3; ++c)
                chw[c * H * W + p] = sweep[3 * p + c] * state.scale;
        const auto linear = lfs::core::Tensor::from_vector(chw, {3, H, W}, lfs::core::Device::CUDA);
        const auto shown = lfs::core::hdr::display_image(linear, state, ev, lfs::core::hdr::ToneOperator::Ocio, sel)
                               .cpu()
                               .contiguous();
        const float* out = shown.ptr<float>();
        ASSERT_NE(out, nullptr);
        double max_abs = 0.0;
        for (std::size_t p = 0; p < H * W; ++p) {
            const glm::vec3 x = glm::vec3(sweep[3 * p], sweep[3 * p + 1], sweep[3 * p + 2]) * std::exp2(ev);
            const glm::vec3 want = lfs::core::ocio::sample(*lut, glm::max(x, glm::vec3(0.0f)));
            for (int c = 0; c < 3; ++c)
                max_abs = std::max(max_abs, std::abs(static_cast<double>(out[c * H * W + p]) - want[c]));
        }
        EXPECT_LE(max_abs, 1e-5) << "CUDA LUT sampling drifted from the C++ twin";
    }

    // A foreign linear colorspace maps to the working space through a 3x3; a non-linear one is refused.
    TEST(HdrDisplayChainTest, OcioLinearMatrixProbe) {
        std::string error;
        const auto m = lfs::core::ocio::linear_matrix("ACEScg", "lin_srgb", {}, &error);
        ASSERT_TRUE(m) << error;
        // AP1 -> Rec.709 (D60 -> D65 Bradford, the ACES convention): the well-known matrix to 1e-3.
        const glm::vec3 r = (*m) * glm::vec3(1.0f, 0.0f, 0.0f);
        EXPECT_NEAR(r.x, 1.7050f, 2e-3f);
        EXPECT_NEAR(r.y, -0.1302f, 2e-3f);
        EXPECT_NEAR(r.z, -0.0240f, 2e-3f);
        // Row sums of the matrix are 1 (white maps to white).
        const glm::vec3 w = (*m) * glm::vec3(1.0f);
        EXPECT_NEAR(w.x, 1.0f, 2e-3f);
        EXPECT_NEAR(w.y, 1.0f, 2e-3f);
        EXPECT_NEAR(w.z, 1.0f, 2e-3f);
        EXPECT_FALSE(lfs::core::ocio::linear_matrix("ACEScct", "lin_srgb", {}, &error)) << "ACEScct is a log encoding";
        EXPECT_FALSE(lfs::core::ocio::linear_matrix("No Such Space", "lin_srgb", {}, &error));
        const auto identity = lfs::core::ocio::linear_matrix("lin_srgb", "Linear Rec.709 (sRGB)", {}, &error);
        ASSERT_TRUE(identity) << error;
        EXPECT_NEAR((*identity)[0][0], 1.0f, 1e-6f);
        EXPECT_NEAR((*identity)[1][0], 0.0f, 1e-6f);
    }

} // namespace
