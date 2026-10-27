/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_hdr_color_chain.cpp
 * @brief HDR fork (PLAN.md M2): the mu-law loss transform against a LibTorch double reference, and the
 *        log-radiance SH colour chain (forward activation, cap, dL/dsh0) against central differences.
 */

#include "hdr_color_chain_cuda.hpp"

#include "core/tensor.hpp"
#include "kernels/hdr_mulaw.hpp"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>
#include <vector>

using lfs::core::Device;
using lfs::core::Tensor;
using lfs::core::TensorShape;

namespace {
    constexpr float kShC0 = 0.28209479177387814f;

    class HdrColorChainTest : public ::testing::Test {
    protected:
        void SetUp() override {
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            if (device_count == 0)
                GTEST_SKIP() << "No CUDA device available";
        }
    };

    double rel_error(const double got, const double want) {
        return std::abs(got - want) / std::max(1e-6, std::abs(want));
    }
} // namespace

// (a) y = log1p(mu*x)/log1p(mu) with x = max(v,0)/s, and its gradient, on v in [-1, 50].
TEST_F(HdrColorChainTest, MulawMatchesTorchDoubleReference) {
    constexpr float kScale = 2.5f;
    constexpr float kMu = 5000.0f;
    constexpr int kN = 4096;
    std::vector<float> v(kN);
    for (int i = 0; i < kN; ++i)
        v[static_cast<size_t>(i)] = -1.0f + 51.0f * static_cast<float>(i) / static_cast<float>(kN - 1);
    v[0] = -1.0f;
    v[1] = 0.0f;
    v[2] = 1e-6f;
    v[3] = 50.0f;
    std::vector<float> upstream(kN);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& g : upstream)
        g = dist(rng);

    auto input = Tensor::from_vector(v, TensorShape({3, 32, static_cast<size_t>(kN / 96)}), Device::CUDA);
    Tensor x_saved;
    const auto y = lfs::training::kernels::hdr_mulaw_forward(input, kScale, kMu, &x_saved);
    auto grad = Tensor::from_vector(upstream, input.shape(), Device::CUDA);
    lfs::training::kernels::hdr_mulaw_backward_(grad, x_saved, kScale, kMu);
    const auto y_host = y.to_vector();
    const auto grad_host = grad.to_vector();
    ASSERT_EQ(y_host.size(), static_cast<size_t>(kN));
    ASSERT_EQ(grad_host.size(), static_cast<size_t>(kN));

    // Reference in double through torch autograd: y = log1p(mu * clamp_min(v/s, 0)) / log1p(mu).
    auto tv = torch::tensor(std::vector<double>(v.begin(), v.end()), torch::dtype(torch::kFloat64)).requires_grad_(true);
    auto ty = torch::log1p(static_cast<double>(kMu) * torch::clamp_min(tv / static_cast<double>(kScale), 0.0)) /
              std::log1p(static_cast<double>(kMu));
    ty.backward(torch::tensor(std::vector<double>(upstream.begin(), upstream.end()), torch::dtype(torch::kFloat64)));
    const auto ty_acc = ty.detach().accessor<double, 1>();
    const auto tg_acc = tv.grad().accessor<double, 1>();

    double worst_y = 0.0, worst_g = 0.0;
    for (int i = 0; i < kN; ++i) {
        const size_t k = static_cast<size_t>(i);
        EXPECT_TRUE(std::isfinite(y_host[k])) << "y at v=" << v[k];
        EXPECT_GE(y_host[k], 0.0f);
        if (v[k] < 0.0f) {
            EXPECT_EQ(y_host[k], 0.0f) << "v=" << v[k];
            EXPECT_EQ(grad_host[k], 0.0f) << "zero gradient below black, v=" << v[k];
            continue;
        }
        worst_y = std::max(worst_y, rel_error(y_host[k], ty_acc[i]));
        worst_g = std::max(worst_g, rel_error(grad_host[k], tg_acc[i]));
    }
    EXPECT_LT(worst_y, 1e-5) << "mu-law forward vs torch double";
    EXPECT_LT(worst_g, 1e-5) << "mu-law backward vs torch double";
    // x saved is the signed v / s.
    const auto x_host = x_saved.to_vector();
    EXPECT_FLOAT_EQ(x_host[0], -1.0f / kScale);
    EXPECT_FLOAT_EQ(x_host[3], 50.0f / kScale);
}

// (b) 32 splats, SH degree 3: c = exp(min(b, ln cmax)); dL/dsh0 = SH_C0 * c * g, 0 at the cap; central differences.
TEST_F(HdrColorChainTest, LogRadianceShChainDegree3) {
    using hdr_color_chain_test::eval_sh_color_chain;
    using hdr_color_chain_test::ShChainInput;
    using hdr_color_chain_test::ShChainOutput;

    constexpr int kN = 32;
    ShChainInput in;
    in.n = kN;
    in.active_sh_bases = 16;
    in.color_max = 1e4f;
    in.cam[0] = 0.3f;
    in.cam[1] = -0.2f;
    in.cam[2] = 0.1f;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    in.sh0.resize(kN * 3);
    in.shN.assign(static_cast<size_t>(kN) * 15 * 3, 0.0f);
    in.means.resize(kN * 3);
    in.grad_c.resize(kN * 3);
    for (int p = 0; p < kN; ++p) {
        for (int ch = 0; ch < 3; ++ch) {
            // b spans dark shadows (ln 1e-3 ~ -6.9) to beyond the cap (ln 1e4 ~ 9.2): sh0 in [-28, +38].
            in.sh0[static_cast<size_t>(p * 3 + ch)] = (-7.5f + 18.0f * static_cast<float>(p) / (kN - 1) - 0.5f) / kShC0 +
                                                      0.3f * unit(rng);
            in.means[static_cast<size_t>(p * 3 + ch)] = 2.0f * unit(rng) + (ch == 2 ? 4.0f : 0.0f);
            in.grad_c[static_cast<size_t>(p * 3 + ch)] = unit(rng);
        }
        for (int k = 0; k < 15; ++k)
            for (int ch = 0; ch < 3; ++ch)
                in.shN[(static_cast<size_t>(p) * 15 + k) * 3 + ch] = 0.15f * unit(rng);
    }
    // Two splats pinned beyond the cap on every channel.
    for (int ch = 0; ch < 3; ++ch) {
        in.sh0[static_cast<size_t>(30 * 3 + ch)] = (12.0f - 0.5f) / kShC0;
        in.sh0[static_cast<size_t>(31 * 3 + ch)] = (9.5f - 0.5f) / kShC0;
    }

    ShChainOutput out;
    ASSERT_TRUE(eval_sh_color_chain(in, out));
    ASSERT_EQ(out.hdr_color.size(), static_cast<size_t>(kN * 3));

    const double log_cmax = std::log(static_cast<double>(in.color_max));
    int capped = 0;
    double worst_fwd = 0.0, worst_grad = 0.0;
    for (int p = 0; p < kN; ++p) {
        for (int ch = 0; ch < 3; ++ch) {
            const size_t k = static_cast<size_t>(p * 3 + ch);
            const double b = out.b[k];
            const double want_c = std::exp(std::min(b, log_cmax));
            const double got_c = out.hdr_color[k];
            EXPECT_TRUE(std::isfinite(got_c));
            EXPECT_GT(got_c, 0.0);
            EXPECT_LE(got_c, static_cast<double>(in.color_max));
            worst_fwd = std::max(worst_fwd, rel_error(got_c, want_c));
            const double g = in.grad_c[k];
            if (b >= log_cmax) {
                ++capped;
                EXPECT_EQ(out.hdr_dL_dsh0[k], 0.0f) << "no gradient above the cap, splat " << p;
                continue;
            }
            // Analytic: dL/dsh0 = SH_C0 * c * g.
            EXPECT_NEAR(out.hdr_dL_dsh0[k], kShC0 * got_c * g, 1e-4 * std::abs(kShC0 * got_c * g) + 1e-7);
            // Central differences on sh0 through c = exp(b): b depends linearly on sh0 with slope SH_C0.
            constexpr double h = 1e-3;
            const double c_plus = std::exp(b + kShC0 * h);
            const double c_minus = std::exp(b - kShC0 * h);
            const double numeric = (c_plus - c_minus) / (2.0 * h) * g;
            worst_grad = std::max(worst_grad, rel_error(out.hdr_dL_dsh0[k], numeric));
        }
    }
    EXPECT_GE(capped, 6) << "the two pinned splats must sit above the cap";
    EXPECT_LT(worst_fwd, 1e-4) << "exp(min(b, ln cmax)) forward";
    EXPECT_LT(worst_grad, 1e-4) << "dL/dsh0 vs central differences";
}

// (c) The LDR instantiation leaves the SH sum untouched: bit-equal to the old path.
TEST_F(HdrColorChainTest, LdrPathIsBitIdentical) {
    using hdr_color_chain_test::eval_sh_color_chain;
    using hdr_color_chain_test::ShChainInput;
    using hdr_color_chain_test::ShChainOutput;
    constexpr int kN = 32;
    ShChainInput in;
    in.n = kN;
    in.active_sh_bases = 16;
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    in.sh0.resize(kN * 3);
    in.shN.resize(static_cast<size_t>(kN) * 15 * 3);
    in.means.resize(kN * 3);
    in.grad_c.assign(kN * 3, 0.0f);
    for (auto& v : in.sh0)
        v = 3.0f * unit(rng);
    for (auto& v : in.shN)
        v = 0.5f * unit(rng);
    for (auto& v : in.means)
        v = 3.0f * unit(rng) + 1.0f;
    ShChainOutput out;
    ASSERT_TRUE(eval_sh_color_chain(in, out));
    ASSERT_EQ(out.b.size(), out.ldr_color.size());
    EXPECT_EQ(0, std::memcmp(out.b.data(), out.ldr_color.data(), out.b.size() * sizeof(float)));
    // And with a 4.0 cap the HDR activation of a sum below ln 4 is exp(b): no accidental clamp in the LDR range.
    for (size_t k = 0; k < out.b.size(); ++k)
        if (out.b[k] < std::log(1e4f))
            EXPECT_NEAR(out.hdr_color[k], std::exp(out.b[k]), 1e-4f * std::exp(out.b[k]) + 1e-7f);
}
