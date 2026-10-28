/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_eval_split.cpp
 * @brief HDR fork (PLAN.md M3 §1): the one hold-out rule behind every split site — by image (the historical
 *        i % test_every) and by position (every test_every-th position with ALL its frames).
 */

#include "eval_split.hpp"

#include <format>
#include <gtest/gtest.h>
#include <set>
#include <string>
#include <vector>

using lfs::training::eval_split_mask;
using lfs::training::position_key;

namespace {
    std::vector<std::string> bracket_names(const int positions, const int frames) {
        std::vector<std::string> names;
        for (int p = 0; p < positions; ++p)
            for (int f = 0; f < frames; ++f)
                names.push_back(std::format("pos_{:03d}_f{}.jpg", p, f));
        return names; // already name-sorted, position-major
    }
} // namespace

TEST(EvalSplitTest, PositionKeyIsTheNameUpToItsLastUnderscore) {
    EXPECT_EQ(position_key("pos_000_f3.jpg"), "pos_000");
    EXPECT_EQ(position_key("pos_012_f0"), "pos_012");
    EXPECT_EQ(position_key("IMG_0001.jpg"), "IMG");
    EXPECT_EQ(position_key("frame0001.png"), "frame0001.png"); // no '_' -> its own position
    EXPECT_EQ(position_key(""), "");
}

TEST(EvalSplitTest, ByImageIsEveryNthImage) {
    const auto names = bracket_names(40, 5);
    const auto held = eval_split_mask(names, 8, false);
    ASSERT_EQ(held.size(), names.size());
    size_t count = 0;
    for (size_t i = 0; i < held.size(); ++i) {
        EXPECT_EQ(held[i], i % 8 == 0) << names[i];
        count += held[i];
    }
    EXPECT_EQ(count, 25u);
    // The design's motivating case: --test-every 40 by image holds out ONE frame of pos_000/008/016/024/032.
    const auto by_image_40 = eval_split_mask(names, 40, false);
    std::vector<std::string> picked;
    for (size_t i = 0; i < names.size(); ++i)
        if (by_image_40[i])
            picked.push_back(names[i]);
    EXPECT_EQ(picked, (std::vector<std::string>{"pos_000_f0.jpg", "pos_008_f0.jpg", "pos_016_f0.jpg",
                                                "pos_024_f0.jpg", "pos_032_f0.jpg"}));
}

TEST(EvalSplitTest, ByPositionHoldsOutWholePositions) {
    // 40 positions x 5 frames, --test-every 8 --test-by-position = pos_000/008/016/024/032 with all frames: M2's set.
    const auto names = bracket_names(40, 5);
    const auto held = eval_split_mask(names, 8, true);
    ASSERT_EQ(held.size(), names.size());
    std::set<std::string> held_positions, train_positions;
    size_t count = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const std::string key(position_key(names[i]));
        (held[i] ? held_positions : train_positions).insert(key);
        count += held[i];
    }
    EXPECT_EQ(count, 25u);
    EXPECT_EQ(held_positions, (std::set<std::string>{"pos_000", "pos_008", "pos_016", "pos_024", "pos_032"}));
    for (const auto& key : held_positions)
        EXPECT_FALSE(train_positions.contains(key)) << key << " has frames on both sides";
    EXPECT_EQ(train_positions.size(), 35u);
}

TEST(EvalSplitTest, ByPositionRanksByFirstAppearanceAndMatchesByImageForSingleFrames) {
    // Ragged brackets: positions with 5, 3 and 1 frames still count once each.
    const std::vector<std::string> ragged = {"a_0", "a_1", "a_2", "a_3", "a_4", "b_0", "b_1", "b_2", "c_0", "d_0", "d_1"};
    const auto held = eval_split_mask(ragged, 2, true);
    const std::vector<bool> want = {true, true, true, true, true, false, false, false, true, false, false};
    EXPECT_EQ(held, want);

    // One frame per position with names carrying no '_' (frame0001.png): both rules give the same split.
    std::vector<std::string> singles;
    for (int p = 0; p < 40; ++p)
        singles.push_back(std::format("frame{:04d}.png", p));
    EXPECT_EQ(eval_split_mask(singles, 8, true), eval_split_mask(singles, 8, false));
    // The merged set's pos_NNN.png names are NOT per-frame names: by position they collapse to the one key "pos"
    // and everything is held out — which is why --test-by-position is for bracket sets only.
    std::vector<std::string> merged;
    for (int p = 0; p < 40; ++p)
        merged.push_back(std::format("pos_{:03d}.png", p));
    EXPECT_EQ(eval_split_mask(merged, 8, true), std::vector<bool>(40, true));

    // test_every <= 0 is treated as 1 (every image held out), never a division by zero.
    EXPECT_EQ(eval_split_mask(singles, 0, true), std::vector<bool>(40, true));
    EXPECT_TRUE(eval_split_mask(std::vector<std::string>{}, 8, true).empty());
}
