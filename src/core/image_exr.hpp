/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "image_codecs.hpp"

namespace lfs::core::image_codecs {
    bool probe_exr(const std::filesystem::path& path, Probe& result, std::string& error);
    // Zero selects automatic scheduling; one keeps decoding sequential. The
    // internal override permits reproducible codec tests and measurements.
    bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error, unsigned max_workers = 0);
    // Scanline EXR, channels R/G/B as FLOAT, ZIP: rgb_chw is three planes of width*height floats (linear
    // radiance, unbounded). HDR eval renders (PLAN.md M2).
    bool encode_exr(const std::filesystem::path& path, const float* rgb_chw, int width, int height, std::string& error);
} // namespace lfs::core::image_codecs
