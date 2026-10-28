/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// HDR fork (PLAN.md M4): ONE place for everything that displays an HDR model outside the Vulkan viewer.
//   - per sample:   lfs::rendering::lfsDisplayHdr(radiance / s, tone_mode, std::exp2(EV))  (glm::vec3 -> sRGB [0,1])
//                   lfs::rendering::lfsSrgbOetf / lfsSrgbEotf / lfsHdrFalseColor / lfsDisplayTone
//   - whole tensors: lfs::rendering::srgb_oetf(linear), hdr_display_image(radiance, model.hdr(), EV),
//                   hdr_point_colors(model, EV), hdr_display_applies(model)
// Both are the SAME text as the shaders (display_color.inc), pinned by tests/test_hdr_display_chain.cpp.
// Tone IDs: 0 none, 1 linear, 2 filmic, 3 hejl, 4 aces, 5 aces2, 6 neutral.

#include "rendering/hdr_display.hpp"
#include "rendering/rasterizer/vulkan/src/display_color.h"
