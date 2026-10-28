/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// HDR export options (PLAN.md M5). Deliberately tensor-free so parameters.hpp and the exporters can
// carry them without pulling the SH projection machinery (core/hdr_sh.hpp) into every include.
//
// An HDR model (SplatData::is_hdr()) stores LOG radiance in its SH. Nothing outside LichtFeld reads
// that, so a file export has three shapes:
//   LinearRadiance  the standard PLY convention, colour = 0.5 + SH_C0 * f_dc (+ f_rest * Y) in LINEAR
//                   radiance, unbounded above 1 (header: `comment lichtfeld_hdr linear`). The default
//                   for PLY. Other applications read it with no custom decoding; a viewer multiplies
//                   by `hdr_display_exposure` and tone-maps.
//   BakedLdr        display-referred sRGB colour in [0,1] at an exposure (EV relative to 1/s) through a
//                   tone operator, re-expressed as standard SH: opens in ANY viewer, HDR is gone.
//   LogInternal     the internal log SH verbatim (`comment lichtfeld_hdr log`): the lossless PLY twin
//                   of the project checkpoint; only LichtFeld reads it.

#include "core/export.hpp"
#include <optional>
#include <string>
#include <string_view>

namespace lfs::core::hdr {

    // The viewer's `color_tonemapping` list (display_color.inc), same indices.
    enum class ToneOperator : int {
        None = 0, // exposure only, no curve (same as Linear when baking)
        Linear = 1,
        Filmic = 2,
        Hejl = 3,
        Aces = 4,
        Aces2 = 5,
        Neutral = 6,
    };

    enum class ExportMode : int {
        Auto = 0,           // PLY: LinearRadiance; every other format: refuse an HDR model
        LinearRadiance = 1, // "linear"
        BakedLdr = 2,       // "baked"
        LogInternal = 3,    // "log" (PLY only)
    };

    struct ExportOptions {
        ExportMode mode = ExportMode::Auto;
        float exposure_ev = 0.0f;            // BakedLdr: EV relative to the scene's natural exposure 1/s
        ToneOperator tone = ToneOperator::Aces; // BakedLdr: tone operator
    };

    LFS_CORE_API std::optional<ToneOperator> parse_tone_operator(std::string_view name);
    LFS_CORE_API std::string_view tone_operator_name(ToneOperator op);
    LFS_CORE_API std::optional<ExportMode> parse_export_mode(std::string_view name);
    LFS_CORE_API std::string_view export_mode_name(ExportMode mode);

    inline constexpr std::string_view kToneOperatorNames = "none, linear, filmic, hejl, aces, aces2, neutral";
    inline constexpr std::string_view kExportModeNames = "auto, linear, baked, log";

    // One message for every writer that cannot hold log radiance.
    LFS_CORE_API std::string refusal_message(std::string_view format_name);

} // namespace lfs::core::hdr
