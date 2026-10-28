/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    // Which images are held out for evaluation (PLAN.md M3 §1). One rule for every split site.
    //
    // by_position = false: image i (in the given order, name-sorted by the callers) is held out iff
    // i % test_every == 0 — the historical rule.
    //
    // by_position = true: an image's POSITION is its name up to the LAST '_' (pos_000_f3.jpg -> pos_000; a name
    // without '_' is its own position), positions are ranked by first appearance, and position j is held out iff
    // j % test_every == 0 — with ALL its frames. A bracket set (5 frames per pose) with --test-every 8 then holds out
    // pos_000/008/016/... exactly as the merged set's pos_NNN.png did; with one frame per position the two rules
    // agree. Never train on a held-out image.

    [[nodiscard]] inline std::string_view position_key(const std::string_view image_name) noexcept {
        const auto underscore = image_name.rfind('_');
        return underscore == std::string_view::npos ? image_name : image_name.substr(0, underscore);
    }

    [[nodiscard]] inline std::vector<bool> eval_split_mask(const std::span<const std::string> names,
                                                           const int test_every,
                                                           const bool by_position) {
        const size_t every = static_cast<size_t>(test_every > 0 ? test_every : 1);
        std::vector<bool> held(names.size(), false);
        if (!by_position) {
            for (size_t i = 0; i < names.size(); ++i)
                held[i] = (i % every) == 0;
            return held;
        }
        std::unordered_map<std::string_view, size_t> rank;
        rank.reserve(names.size());
        for (size_t i = 0; i < names.size(); ++i) {
            const auto key = position_key(names[i]);
            const auto [it, inserted] = rank.try_emplace(key, rank.size());
            held[i] = (it->second % every) == 0;
        }
        return held;
    }

    // The rule every split site applies: the explicit option, or — for a bracket set — by position whenever
    // anything is held out at all. Holding out single FRAMES of a bracket would leak the held-out pose through its
    // sibling frames (same camera, other exposures), so a bracket run never splits by image.
    [[nodiscard]] inline bool holds_out_by_position(const lfs::core::param::TrainingParameters& params) noexcept {
        return params.dataset.test_by_position || (params.optimization.hdr_brackets && params.dataset.test_every > 1);
    }

    [[nodiscard]] inline std::vector<bool> eval_split_mask(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        const int test_every,
        const bool by_position) {
        std::vector<std::string> names;
        names.reserve(cameras.size());
        for (const auto& camera : cameras)
            names.push_back(camera ? camera->image_name() : std::string{});
        return eval_split_mask(names, test_every, by_position);
    }

} // namespace lfs::training
