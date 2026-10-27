/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <vector>

// Device-side probe of the HDR colour chain (PLAN.md M2): the fastgs SH -> colour conversion with the
// log-radiance activation on, and the per-splat backward the blend + preprocess kernels apply
// (dL/db = c * dL/dc, masked at the cap, then dL/dsh0 = SH_C0 * dL/db). One thread per splat.

namespace hdr_color_chain_test {

    struct ShChainInput {
        int n = 0;
        int active_sh_bases = 1;   // 1, 4, 9 or 16
        std::vector<float> sh0;    // [n*3]
        std::vector<float> shN;    // canonical [n*15*3] rest coefficients (zeros beyond the active bases)
        std::vector<float> means;  // [n*3]
        float cam[3] = {0.f, 0.f, 0.f};
        std::vector<float> grad_c; // [n*3] dL/dc fed to the backward
        float color_max = 1e4f;
    };

    struct ShChainOutput {
        std::vector<float> b;            // [n*3] the raw SH sum (old LDR path, no activation)
        std::vector<float> ldr_color;    // [n*3] LDR instantiation of the activation step (expected == b)
        std::vector<float> hdr_color;    // [n*3] log-radiance activation
        std::vector<float> hdr_dL_dsh0;  // [n*3] through the log chain and the cap mask
    };

    // Returns false on a CUDA error.
    bool eval_sh_color_chain(const ShChainInput& in, ShChainOutput& out);

} // namespace hdr_color_chain_test
