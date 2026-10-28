/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// OpenColorIO colour management for the HDR fork (docs/hdr.md "Colour management").
//
// The invariant: inside LichtFeld an HDR model's radiance is ALWAYS in the working space `lin_srgb`
// (scene-linear, Rec.709/sRGB primaries, D65 — OCIO's "Linear Rec.709 (sRGB)", alias `lin_srgb` in the
// ACES configs). Files declare it (PLY `comment lichtfeld_colorspace lin_srgb`, EXR `chromaticities` +
// `lichtfeld_colorspace`); a foreign linear declaration is converted to it on load through this module.
//
// Display: an OCIO (display, view[, look]) of a config is baked ONCE into a 65^3 float LUT over a
// per-channel log2 shaper (lo..hi stops of normalised radiance); the viewer's compose pass, the CUDA bake
// and the C++ twin all sample it with the same text (display_color.inc lfsOcioLut), so what the viewer
// shows is what the exporters write. The CPU processor itself is the test oracle (apply_cpu).
//
// This header is OCIO-free: only src/core/ocio.cpp includes <OpenColorIO/OpenColorIO.h>.

#include "core/export.hpp"

#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core::ocio {

    // The working-space tag every LichtFeld HDR file carries.
    inline constexpr std::string_view kWorkingSpaceTag = "lin_srgb";

    // Which config and which display/view/look. Empty config_path = $OCIO, else the built-in
    // `studio-config-latest`; empty display/view = the config's defaults; empty look = none.
    struct LFS_CORE_API Selection {
        std::string config_path;
        std::string display;
        std::string view;
        std::string look;

        bool operator==(const Selection&) const = default;
    };

    // A baked display LUT. Node i on each axis is the normalised radiance x_i = 2^(lo + i/(size-1) * (hi-lo)),
    // node 0 is exactly x = 0 (black is exact). rgb holds 3 * size^3 floats, index (ir*size + ig)*size + ib,
    // i.e. blue fastest — the layout display_color.inc lfsOcioLut reads with LFS_OCIO_LUT_FETCH.
    struct LFS_CORE_API Lut3D {
        int size = 65;
        float log2_lo = -14.0f;
        float log2_hi = 8.0f;
        std::vector<float> rgb;
        std::string key;       // config|display|view|look|size|lo|hi — what the caches compare
        std::string config;    // the resolved config's name (for logs)
    };

    inline constexpr int kDefaultLutSize = 65;
    inline constexpr float kDefaultLog2Lo = -14.0f;
    inline constexpr float kDefaultLog2Hi = 8.0f;

    // OCIO is compiled in and the selection's config loads. `error` (optional) receives why not.
    LFS_CORE_API bool available(const Selection& selection = {}, std::string* error = nullptr);
    LFS_CORE_API std::string config_name(const Selection& selection = {});
    LFS_CORE_API std::string config_description(const Selection& selection = {});

    LFS_CORE_API std::vector<std::string> displays(const Selection& selection = {});
    LFS_CORE_API std::vector<std::string> views(const Selection& selection, std::string_view display);
    LFS_CORE_API std::vector<std::string> looks(const Selection& selection = {});
    LFS_CORE_API std::string default_display(const Selection& selection = {});
    LFS_CORE_API std::string default_view(const Selection& selection, std::string_view display);

    // The process-wide default selection: written by the CLI (--ocio, --ocio-display/-view/-look), the
    // viewer's render settings (tone "ocio") and Python lf.io.set_ocio(); read wherever a Selection is
    // left empty. Fields left empty in `s` keep the current values.
    LFS_CORE_API Selection current();
    LFS_CORE_API void set_current(const Selection& s);
    // `selection` with its empty fields filled from current() and the config's defaults.
    LFS_CORE_API Selection resolve(const Selection& selection);

    // Validates display / view / look against the config. Returns the resolved selection, or the message.
    LFS_CORE_API std::optional<Selection> validate(const Selection& selection, std::string* error);

    // The working space's name in this config: alias `lin_srgb`, else "Linear Rec.709 (sRGB)", else the
    // scene_linear role (warned once). Empty when none of them exists.
    LFS_CORE_API std::string working_space(const Selection& selection = {});

    // The baked LUT for a selection (cached by key; the bake runs the CPU processor on size^3 nodes).
    // nullptr when OCIO cannot build the processor (error says why).
    LFS_CORE_API std::shared_ptr<const Lut3D> lut_for(const Selection& selection, int size = kDefaultLutSize,
                                                      std::string* error = nullptr);

    // The reference path: working space -> [look] -> display/view through OCIO's own CPU processor on n RGB
    // triplets in place (the test oracle). Returns false (and the message) when the processor cannot be built.
    LFS_CORE_API bool apply_cpu(const Selection& selection, float* rgb, std::size_t n, std::string* error = nullptr);

    // The 3x3 matrix taking linear colour in colorspace `src` to colorspace `dst` of the config, when that
    // transform is linear (probed on the basis vectors at two scales); nullopt when it is not, or when a name
    // is unknown. `dst` / `src` may be the tag `lin_srgb` (mapped through working_space()).
    LFS_CORE_API std::optional<glm::mat3> linear_matrix(std::string_view src, std::string_view dst,
                                                        const Selection& selection = {},
                                                        std::string* error = nullptr);

    // Sample a LUT on the CPU (the C++ twin of the shader; display_color.inc lfsOcioLut on lut.rgb).
    LFS_CORE_API glm::vec3 sample(const Lut3D& lut, const glm::vec3& exposed_linear);

} // namespace lfs::core::ocio
