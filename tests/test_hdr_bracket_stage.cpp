/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_hdr_bracket_stage.cpp
 * @brief HDR fork (PLAN.md M3): the exposure + fixed Debevec LUT + saturation gate stage. The LUT forward against
 *        tools/eval/hdr_metrics.apply_crf (tests/data/hdr_bracket_crf_fixture.json, made with the tools venv),
 *        gradients on v and delta against central differences across the LUT and beyond both ends, the gate
 *        masks, exif == delta frozen, and the checkpoint round trip.
 */

#include "components/hdr_bracket_stage.hpp"
#include "core/tensor.hpp"
#include "kernels/hdr_bracket.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;
using lfs::core::TensorShape;
using lfs::training::HdrBracketCrf;
using lfs::training::HdrBracketStage;

namespace {
    struct Fixture {
        HdrBracketCrf crf;
        std::vector<double> x;                    // 1024 linear exposures (LUT units)
        std::vector<std::vector<int>> codes;      // [3][1024] apply_crf output
    };

    Fixture load_fixture() {
        const auto path = std::filesystem::path(PROJECT_ROOT_PATH) / "tests" / "data" / "hdr_bracket_crf_fixture.json";
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("missing fixture " + path.string());
        const auto j = nlohmann::json::parse(in);
        auto crf = HdrBracketCrf::from_curves(j.at("log_irradiance").get<std::vector<std::vector<double>>>(),
                                              j.at("exif_ev_ref").get<float>());
        if (!crf)
            throw std::runtime_error(crf.error());
        return Fixture{*crf, j.at("x").get<std::vector<double>>(), j.at("apply_crf_codes").get<std::vector<std::vector<int>>>()};
    }

    class HdrBracketStageTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            if (device_count == 0)
                GTEST_SKIP() << "No CUDA device available";
        }
    };

    double rel_error(const double got, const double want) {
        return std::abs(got - want) / std::max(1e-9, std::abs(want));
    }

    // Host double reference of the whole chain for one element: z (in [0,1] units) at render v, log2 gain b + delta.
    double z_host(const HdrBracketCrf& crf, const int c, const double v, const double log2_gain, double* dz_du = nullptr) {
        const double x = v * std::exp2(log2_gain);
        const double u = x > 0.0 ? std::log(x) : -1e30;
        double z_code = 0.0, d = 0.0;
        lfs::training::kernels::hdr_bracket_lut_host(crf.log_irradiance[static_cast<size_t>(c)].data(), u, &z_code, &d);
        if (dz_du)
            *dz_du = d / 255.0;
        return z_code / 255.0;
    }
} // namespace

// The GPU LUT forward reproduces tools/eval/hdr_metrics.apply_crf on 1024 exposures per channel (<= 0.5/255 where the
// Python output is not clipped), and extrapolates linearly in ln x beyond both ends instead of clamping.
TEST_F(HdrBracketStageTest, LutForwardMatchesPythonApplyCrf) {
    const auto fx = load_fixture();
    ASSERT_EQ(fx.x.size(), 1024u);
    HdrBracketStage stage(fx.crf, HdrBracketStage::Config{});
    std::vector<float> v(3 * 1024);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < 1024; ++i)
            v[static_cast<size_t>(c) * 1024 + i] = static_cast<float>(fx.x[i]);
    const auto z = stage.apply_eval(Tensor::from_vector(v, TensorShape({3, 32, 32}), Device::CUDA), 0.0f).to_vector();
    double worst = 0.0;
    int compared = 0, below = 0, above = 0;
    for (int c = 0; c < 3; ++c) {
        const auto& g = fx.crf.log_irradiance[static_cast<size_t>(c)];
        for (size_t i = 0; i < 1024; ++i) {
            const double got = 255.0 * z[static_cast<size_t>(c) * 1024 + i];
            const int want = fx.codes[static_cast<size_t>(c)][i];
            ASSERT_TRUE(std::isfinite(got));
            if (want > 0 && want < 255) {
                worst = std::max(worst, std::abs(got - want));
                EXPECT_LE(std::abs(got - want), 0.5 + 2e-3) << "channel " << c << " x=" << fx.x[i];
                ++compared;
            } else if (want == 0) {
                EXPECT_LE(got, 0.5 + 2e-3);
                if (std::log(fx.x[i]) < g[0]) {
                    ++below;
                    EXPECT_LT(got, 0.0) << "below the LUT z extrapolates negative, x=" << fx.x[i];
                }
            } else {
                EXPECT_GE(got, 254.5 - 2e-3);
                if (std::log(fx.x[i]) > g[255]) {
                    ++above;
                    EXPECT_GT(got, 255.0) << "above the LUT z extrapolates past 1, x=" << fx.x[i];
                }
            }
        }
    }
    EXPECT_GT(compared, 1000);
    EXPECT_GT(below, 50);
    EXPECT_GT(above, 50);
    // The knots map to their own code.
    std::vector<float> knots(3 * 256);
    for (int c = 0; c < 3; ++c)
        for (int code = 0; code < 256; ++code)
            knots[static_cast<size_t>(c) * 256 + static_cast<size_t>(code)] = static_cast<float>(fx.crf.exposure_of_code(c, code));
    const auto zk = stage.apply_eval(Tensor::from_vector(knots, TensorShape({3, 16, 16}), Device::CUDA), 0.0f).to_vector();
    for (size_t i = 0; i < zk.size(); ++i)
        EXPECT_NEAR(255.0 * zk[i], static_cast<double>(i % 256), 2e-3);
}

// dL/dv and dL/ddelta from backward_in_place against central differences of the double host reference (rel < 1e-4)
// and of the GPU forward itself (rel < 2e-3, float32 noise) — inside the LUT and beyond both ends. No gate here.
TEST_F(HdrBracketStageTest, GradientsMatchCentralDifferences) {
    const auto fx = load_fixture();
    HdrBracketStage stage(fx.crf, HdrBracketStage::Config{});
    constexpr int kUid = 7;
    constexpr float kB = -3.3f;
    constexpr float kDelta = 0.37f;
    stage.register_frame(kUid, kB);
    stage.finalize();
    stage.set_delta_for_test(kUid, kDelta);

    // 3 x 6 x 8 = 144 elements: u = ln x mid-segment from 6 nats under the floor to 3 nats above the ceiling.
    constexpr int kH = 6, kW = 8, kN = 3 * kH * kW;
    std::vector<float> v(kN), upstream(kN);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> gdist(-1.0f, 1.0f);
    for (int c = 0; c < 3; ++c) {
        const auto& g = fx.crf.log_irradiance[static_cast<size_t>(c)];
        for (int p = 0; p < kH * kW; ++p) {
            const double t = static_cast<double>(p) / (kH * kW - 1);
            double u = (g[0] - 6.0) + t * ((g[255] + 3.0) - (g[0] - 6.0));
            // Nudge off a knot so the central difference does not straddle a slope change.
            double zc = 0.0;
            lfs::training::kernels::hdr_bracket_lut_host(g.data(), u, &zc, nullptr);
            if (std::abs(zc - std::round(zc)) < 0.05)
                u += 0.003;
            const size_t i = static_cast<size_t>(c * kH * kW + p);
            v[i] = static_cast<float>(std::exp(u - (kB + kDelta) * std::log(2.0)));
            upstream[i] = gdist(rng);
        }
    }
    const auto gt = Tensor::full(TensorShape({3, kH, kW}), 0.5f, Device::CUDA); // never gated
    auto input = Tensor::from_vector(v, TensorShape({3, kH, kW}), Device::CUDA);

    const auto loss_gpu = [&](const std::vector<float>& vv, const float delta) {
        stage.set_delta_for_test(kUid, delta);
        const auto zz = stage.apply(Tensor::from_vector(vv, TensorShape({3, kH, kW}), Device::CUDA), kUid, gt).to_vector();
        double l = 0.0;
        for (int i = 0; i < kN; ++i)
            l += static_cast<double>(upstream[static_cast<size_t>(i)]) * zz[static_cast<size_t>(i)];
        return l;
    };

    // The analytic gradients.
    stage.set_delta_for_test(kUid, kDelta);
    (void)stage.apply(input, kUid, gt);
    auto grad = Tensor::from_vector(upstream, TensorShape({3, kH, kW}), Device::CUDA);
    stage.backward_in_place(grad, kUid);
    const auto grad_v = grad.to_vector();
    const double grad_delta = stage.grad_delta_for_test(kUid);

    // Reference: double host chain.
    double ref_delta = 0.0;
    int nonzero = 0, below_end = 0, above_end = 0;
    for (int i = 0; i < kN; ++i) {
        const int c = i / (kH * kW);
        const double vi = v[static_cast<size_t>(i)];
        const double h = 1e-6 * vi;
        const double dz_dv = (z_host(fx.crf, c, vi + h, kB + kDelta) - z_host(fx.crf, c, vi - h, kB + kDelta)) / (2 * h);
        const double want = upstream[static_cast<size_t>(i)] * dz_dv;
        const double hd = 1e-6;
        ref_delta += upstream[static_cast<size_t>(i)] *
                     (z_host(fx.crf, c, vi, kB + kDelta + hd) - z_host(fx.crf, c, vi, kB + kDelta - hd)) / (2 * hd);
        const double u = std::log(vi) + (kB + kDelta) * std::log(2.0);
        const auto& g = fx.crf.log_irradiance[static_cast<size_t>(c)];
        if (u > g[0] - 5.0) {
            EXPECT_LT(rel_error(grad_v[static_cast<size_t>(i)], want), 1e-4) << "element " << i << " u=" << u;
            EXPECT_NE(grad_v[static_cast<size_t>(i)], 0.0f) << "dead gradient at u=" << u;
            ++nonzero;
            if (u < g[0])
                ++below_end;
            if (u > g[255])
                ++above_end;
        } else {
            EXPECT_EQ(grad_v[static_cast<size_t>(i)], 0.0f) << "flat below the u clamp, u=" << u;
        }
    }
    EXPECT_GT(nonzero, 100);
    EXPECT_GT(below_end, 5) << "the sweep must cover x below the LUT";
    EXPECT_GT(above_end, 5) << "the sweep must cover x above the LUT";
    EXPECT_LT(rel_error(grad_delta, ref_delta), 1e-4) << grad_delta << " vs " << ref_delta;

    // GPU-only central differences (float32): delta, and a handful of v elements.
    const double l_plus = loss_gpu(v, kDelta + 1e-2f), l_minus = loss_gpu(v, kDelta - 1e-2f);
    EXPECT_LT(rel_error((l_plus - l_minus) / 2e-2, grad_delta), 1e-2); // float32 sums over 144 elements
    for (const int i : {5, 20, 47, 70, 95, 120, 143}) {
        auto vp = v, vm = v;
        const float h = 1e-2f * v[static_cast<size_t>(i)];
        vp[static_cast<size_t>(i)] += h;
        vm[static_cast<size_t>(i)] -= h;
        const double fd = (loss_gpu(vp, kDelta) - loss_gpu(vm, kDelta)) / (2.0 * h);
        if (grad_v[static_cast<size_t>(i)] != 0.0f)
            EXPECT_LT(rel_error(fd, grad_v[static_cast<size_t>(i)]), 3e-2) << "element " << i; // float32 z, h = 1 %
    }
}

// The one-sided gate: clipped-high JPEG pixels bound the prediction from below only, clipped-low from above only.
TEST_F(HdrBracketStageTest, SaturationGateIsOneSided) {
    const auto fx = load_fixture();
    HdrBracketStage stage(fx.crf, HdrBracketStage::Config{});
    stage.register_frame(1, 0.0f);
    stage.finalize();
    // Six pixels per channel: prediction code ~ 100 and ~ 254.9 (above 250/255) and ~ 2 (below 5/255) against
    // gt codes 255 / 250 / 128 / 5 / 0.
    const std::vector<unsigned char> gts = {255, 255, 128, 5, 5, 0};
    std::vector<float> v;
    for (int c = 0; c < 3; ++c) {
        const double x_mid = fx.crf.exposure_of_code(c, 100);
        const double x_hi = std::exp(fx.crf.log_irradiance[static_cast<size_t>(c)][255] + 0.5); // beyond -> z > 1
        const double x_lo = std::exp(fx.crf.log_irradiance[static_cast<size_t>(c)][0] - 0.5);   // below -> z < 0
        for (const double x : {x_hi, x_mid, x_hi, x_lo, x_mid, x_lo})
            v.push_back(static_cast<float>(x));
    }
    auto gt_codes = std::vector<unsigned char>{};
    for (int c = 0; c < 3; ++c)
        gt_codes.insert(gt_codes.end(), gts.begin(), gts.end());
    std::vector<float> gt_u8f(gt_codes.begin(), gt_codes.end());
    auto gt = Tensor::from_vector(gt_u8f, TensorShape({3, 2, 3}), Device::CUDA).to(DataType::UInt8);
    ASSERT_EQ(gt.dtype(), DataType::UInt8);
    auto input = Tensor::from_vector(v, TensorShape({3, 2, 3}), Device::CUDA);
    const auto z = stage.apply(input, 1, gt).to_vector();
    auto grad = Tensor::full(TensorShape({3, 2, 3}), 1.0f, Device::CUDA);
    stage.backward_in_place(grad, 1);
    const auto gv = grad.to_vector();
    for (int c = 0; c < 3; ++c) {
        const size_t o = static_cast<size_t>(c) * 6;
        // gt 255, prediction above: gated to 255/255, no gradient.
        EXPECT_FLOAT_EQ(z[o + 0], 1.0f);
        EXPECT_EQ(gv[o + 0], 0.0f);
        // gt 255, prediction below: identity (the model still owes brightness), gradient alive.
        EXPECT_NEAR(z[o + 1], 100.0 / 255.0, 2e-3);
        EXPECT_NE(gv[o + 1], 0.0f);
        // gt 128: never gated even beyond the LUT.
        EXPECT_GT(z[o + 2], 1.0f);
        EXPECT_NE(gv[o + 2], 0.0f);
        // gt 5, prediction below: gated to 5/255, no gradient.
        EXPECT_FLOAT_EQ(z[o + 3], 5.0f / 255.0f);
        EXPECT_EQ(gv[o + 3], 0.0f);
        // gt 5, prediction above: identity.
        EXPECT_NEAR(z[o + 4], 100.0 / 255.0, 2e-3);
        EXPECT_NE(gv[o + 4], 0.0f);
        // gt 0, prediction below (negative extrapolation): gated to 0.
        EXPECT_FLOAT_EQ(z[o + 5], 0.0f);
        EXPECT_EQ(gv[o + 5], 0.0f);
    }
    // Float32 GT gates the same pixels.
    auto gt_f = Tensor::from_vector([&] {
        std::vector<float> f;
        for (const auto code : gt_codes)
            f.push_back(static_cast<float>(code) / 255.0f);
        return f;
    }(), TensorShape({3, 2, 3}), Device::CUDA);
    const auto zf = stage.apply(input, 1, gt_f).to_vector();
    for (size_t i = 0; i < zf.size(); ++i)
        EXPECT_NEAR(zf[i], z[i], 1e-6f) << i;
}

// hdr_exposure = exif: the training forward equals the eval forward at the EXIF gain and nothing moves; learn: a
// step moves delta against its gradient and the prior, within the clamp.
TEST_F(HdrBracketStageTest, ExifModeIsDeltaFrozenAndLearnModeSteps) {
    const auto fx = load_fixture();
    HdrBracketStage::Config frozen;
    frozen.learn = false;
    HdrBracketStage exif(fx.crf, frozen);
    HdrBracketStage::Config learning;
    learning.lr = 0.05f;
    learning.max_delta = 0.1f;
    HdrBracketStage learn(fx.crf, learning);
    for (auto* s : {&exif, &learn}) {
        s->register_frame(3, -2.0f);
        s->register_frame(4, 0.5f);
        s->finalize();
    }
    std::vector<float> v(3 * 16);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = 0.5f + 4.0f * static_cast<float>(i) / static_cast<float>(v.size());
    auto input = Tensor::from_vector(v, TensorShape({3, 4, 4}), Device::CUDA);
    const auto gt = Tensor::full(TensorShape({3, 4, 4}), 0.5f, Device::CUDA);

    const auto z_train = exif.apply(input, 3, gt).to_vector();
    const auto z_eval = exif.apply_eval(input, -2.0f).to_vector();
    for (size_t i = 0; i < z_train.size(); ++i)
        EXPECT_FLOAT_EQ(z_train[i], z_eval[i]);
    auto g1 = Tensor::full(TensorShape({3, 4, 4}), 1.0f, Device::CUDA);
    exif.backward_in_place(g1, 3);
    exif.step(3);
    EXPECT_EQ(exif.delta(3), 0.0f);
    EXPECT_EQ(exif.grad_delta_for_test(3), 0.0f);
    EXPECT_EQ(exif.diagnostics().max_abs_delta, 0.0f);

    // Positive upstream gradient everywhere -> dL/ddelta > 0 -> delta decreases; the clamp holds at 0.1 stop.
    for (int it = 0; it < 30; ++it) {
        (void)learn.apply(input, 3, gt);
        auto g = Tensor::full(TensorShape({3, 4, 4}), 1.0f, Device::CUDA);
        learn.backward_in_place(g, 3);
        if (it == 0)
            EXPECT_GT(learn.grad_delta_for_test(3), 0.0f);
        learn.step(3);
    }
    EXPECT_LT(learn.delta(3), 0.0f);
    EXPECT_GE(learn.delta(3), -0.1f - 1e-6f);
    EXPECT_EQ(learn.delta(4), 0.0f) << "an unseen frame does not move";
    EXPECT_EQ(learn.grad_delta_for_test(3), 0.0f) << "step zeroes the accumulator";
    const auto d = learn.diagnostics();
    EXPECT_EQ(d.frames, 2);
    EXPECT_FLOAT_EQ(d.max_abs_delta, -learn.delta(3));

    // With a zero photometric gradient the prior alone pulls a displaced delta back toward 0.
    learn.set_delta_for_test(4, 0.08f);
    for (int it = 0; it < 20; ++it) {
        (void)learn.apply(input, 4, gt);
        auto g = Tensor::zeros(TensorShape({3, 4, 4}), Device::CUDA);
        learn.backward_in_place(g, 4);
        learn.step(4);
    }
    EXPECT_LT(std::abs(learn.delta(4)), 0.08f);
}

TEST_F(HdrBracketStageTest, CheckpointRoundTripRestoresDeltas) {
    const auto fx = load_fixture();
    HdrBracketStage stage(fx.crf, HdrBracketStage::Config{});
    stage.register_frame(10, -1.0f);
    stage.register_frame(11, 0.0f);
    stage.register_frame(12, 1.0f);
    stage.finalize();
    stage.set_delta_for_test(11, 0.25f);
    stage.set_delta_for_test(12, -0.5f);
    std::stringstream buffer;
    stage.serialize(buffer);

    HdrBracketStage fresh(fx.crf, HdrBracketStage::Config{});
    fresh.register_frame(12, 1.0f);
    fresh.register_frame(10, -1.0f);
    fresh.register_frame(11, 0.0f); // another registration order: matched by uid
    fresh.finalize();
    HdrBracketStage loaded;
    loaded.deserialize(buffer);
    EXPECT_EQ(loaded.num_frames(), 3);
    fresh.adopt_checkpoint_state(loaded);
    EXPECT_FLOAT_EQ(fresh.delta(10), 0.0f);
    EXPECT_FLOAT_EQ(fresh.delta(11), 0.25f);
    EXPECT_FLOAT_EQ(fresh.delta(12), -0.5f);
    EXPECT_FLOAT_EQ(fresh.b(12), 1.0f);

    std::stringstream again;
    stage.serialize(again);
    EXPECT_NO_THROW(HdrBracketStage::consume_checkpoint(again));
}

// B2: the hat weight is the channel mean of (min(z, 255 - z) / 127.5)^p: 1 at code 127.5, 0 at 0 and 255, 0.04 at
// the gate's 250 / 5, 1 everywhere at p = 0; UInt8 codes and Float32 [0,1] agree.
TEST_F(HdrBracketStageTest, HatWeightMatchesHostReference) {
    const auto fx = load_fixture();
    HdrBracketStage stage(fx.crf, HdrBracketStage::Config{});
    const std::vector<unsigned char> codes = {128, 0, 255, 250, 5, 64, 192, 100};
    std::vector<float> gt_f(3 * codes.size());
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < codes.size(); ++i)
            gt_f[static_cast<size_t>(c) * codes.size() + i] = static_cast<float>(codes[i]);
    const auto shape = TensorShape({3, 2, 4});
    auto gt_u8 = Tensor::from_vector(gt_f, shape, Device::CUDA).to(DataType::UInt8);
    std::vector<float> gt_unit(gt_f);
    for (auto& v : gt_unit)
        v /= 255.0f;
    auto gt_float = Tensor::from_vector(gt_unit, shape, Device::CUDA);
    for (const double p : {1.0, 2.0, 0.0}) {
        const auto w_u8 = stage.hat_weight(gt_u8, static_cast<float>(p)).to_vector();
        const auto w_f = stage.hat_weight(gt_float, static_cast<float>(p)).to_vector();
        ASSERT_EQ(w_u8.size(), codes.size());
        for (size_t i = 0; i < codes.size(); ++i) {
            const double expect = lfs::training::kernels::hdr_bracket_hat_host(codes[i], p);
            EXPECT_NEAR(w_u8[i], expect, 1e-5) << "code " << int(codes[i]) << " p " << p;
            EXPECT_NEAR(w_f[i], expect, 1e-4) << "code " << int(codes[i]) << " p " << p;
        }
    }
    EXPECT_NEAR(lfs::training::kernels::hdr_bracket_hat_host(128, 1.0), 127.0 / 127.5, 1e-12); // the peak is 127.5
    EXPECT_DOUBLE_EQ(lfs::training::kernels::hdr_bracket_hat_host(0, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(lfs::training::kernels::hdr_bracket_hat_host(255, 1.0), 0.0);
    EXPECT_NEAR(lfs::training::kernels::hdr_bracket_hat_host(250, 1.0), 5.0 / 127.5, 1e-12);
    EXPECT_DOUBLE_EQ(lfs::training::kernels::hdr_bracket_hat_host(7, 0.0), 1.0);
}

TEST_F(HdrBracketStageTest, CrfLoadRequiresTheExifAnchor) {
    const auto dir = std::filesystem::temp_directory_path() / "lfs_hdr_bracket_crf_test";
    std::filesystem::create_directories(dir);
    const auto fx = load_fixture();
    nlohmann::json j;
    j["log_irradiance"] = fx.crf.log_irradiance;
    {
        std::ofstream out(dir / "no_anchor.json");
        out << j.dump();
    }
    const auto missing = HdrBracketCrf::load(dir / "no_anchor.json");
    ASSERT_FALSE(missing.has_value());
    EXPECT_NE(missing.error().find("exif_ev_ref"), std::string::npos) << missing.error();
    j["exif_ev_ref"] = 2.644;
    {
        std::ofstream out(dir / "anchor.json");
        out << j.dump();
    }
    const auto ok = HdrBracketCrf::load(dir / "anchor.json");
    ASSERT_TRUE(ok.has_value()) << ok.error();
    EXPECT_FLOAT_EQ(ok->exif_ev_ref, 2.644f);
    for (int c = 0; c < 3; ++c)
        for (int z = 1; z < 256; ++z)
            EXPECT_GT(ok->log_irradiance[static_cast<size_t>(c)][static_cast<size_t>(z)],
                      ok->log_irradiance[static_cast<size_t>(c)][static_cast<size_t>(z - 1)]);
    EXPECT_FALSE(HdrBracketCrf::load(dir / "absent.json").has_value());
    std::filesystem::remove_all(dir);
}
