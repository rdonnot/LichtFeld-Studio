/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// The one gate every writer that cannot hold log radiance goes through (PLAN.md M5): an LDR model passes
// as is; an HDR model is baked to LDR when the options ask for it (ExportMode::BakedLdr) and refused with
// the message naming the two ways out otherwise.

#include "core/hdr_sh.hpp"
#include "io/error.hpp"
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfs::io {

    [[nodiscard]] inline Result<const core::SplatData*> hdr_source_for_ldr_writer(
        const core::SplatData& splat,
        const core::hdr::ExportOptions& hdr,
        std::optional<core::SplatData>& baked_storage,
        const std::string_view format_name,
        const std::filesystem::path& output_path) {
        if (!splat.is_hdr())
            return &splat;
        if (hdr.mode != core::hdr::ExportMode::BakedLdr)
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, core::hdr::refusal_message(format_name), output_path);
        try {
            baked_storage.emplace(core::hdr::bake_hdr_to_ldr(splat, hdr.exposure_ev, hdr.tone));
        } catch (const std::exception& e) {
            return make_error(ErrorCode::ENCODING_FAILED, std::string("HDR bake failed: ") + e.what(), output_path);
        }
        return &*baked_storage;
    }

} // namespace lfs::io
