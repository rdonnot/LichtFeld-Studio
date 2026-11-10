/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// HDR fork: an RGBA EXR keeps its radiance past 1 when it is decoded for HDR training (its alpha is the image's
// transparency or a mask). The RGBA decode used to go through the u16/u8 decoders only, so every value above 1 was
// clipped and an HDR splat trained on RGBA renders came out 8-35x too dark.

#include "core/tensor.hpp"
#include "io/cache_image_loader.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

    constexpr int kWidth = 16;
    constexpr int kHeight = 8;
    constexpr float kRed = 50.0f, kGreen = 2.5f, kBlue = 0.25f, kAlpha = 0.5f;

    // A minimal scanline EXR, float A/B/G/R, no compression, written by hand: lichtfeld_tests does not link the
    // OpenEXR C++ library (only the app's OpenEXRCore decoder), and save_image_exr writes RGB only.
    class ExrWriter {
    public:
        void bytes(const void* data, size_t n) {
            const auto* p = static_cast<const char*>(data);
            out_.insert(out_.end(), p, p + n);
        }
        template <typename T>
        void put(T value) { bytes(&value, sizeof(T)); }
        void str(const std::string& s) { bytes(s.c_str(), s.size() + 1); }
        void attribute(const std::string& name, const std::string& type, const std::vector<char>& value) {
            str(name);
            str(type);
            put<int32_t>(static_cast<int32_t>(value.size()));
            bytes(value.data(), value.size());
        }
        std::vector<char>& data() { return out_; }

    private:
        std::vector<char> out_;
    };

    template <typename... T>
    std::vector<char> pack(T... values) {
        ExrWriter w;
        (w.put(values), ...);
        return w.data();
    }

    std::filesystem::path write_rgba_exr() {
        static std::atomic<int> sequence{0};
        const auto path = std::filesystem::temp_directory_path() /
                          ("lfs_hdr_rgba_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                           "_" + std::to_string(sequence.fetch_add(1)) + ".exr");
        ExrWriter w;
        w.put<int32_t>(20000630);  // magic
        w.put<int32_t>(2);         // version 2, scanline
        ExrWriter channels;
        for (const char* name : {"A", "B", "G", "R"}) {  // alphabetical, as EXR stores them
            channels.str(name);
            channels.put<int32_t>(2);  // FLOAT
            channels.put<uint8_t>(0);
            channels.put<uint8_t>(0);
            channels.put<uint8_t>(0);
            channels.put<uint8_t>(0);
            channels.put<int32_t>(1);
            channels.put<int32_t>(1);
        }
        channels.put<uint8_t>(0);
        w.attribute("channels", "chlist", channels.data());
        w.attribute("compression", "compression", {0});
        w.attribute("dataWindow", "box2i", pack<int32_t>(0, 0, kWidth - 1, kHeight - 1));
        w.attribute("displayWindow", "box2i", pack<int32_t>(0, 0, kWidth - 1, kHeight - 1));
        w.attribute("lineOrder", "lineOrder", {0});
        w.attribute("pixelAspectRatio", "float", pack<float>(1.0f));
        w.attribute("screenWindowCenter", "v2f", pack<float>(0.0f, 0.0f));
        w.attribute("screenWindowWidth", "float", pack<float>(1.0f));
        w.put<uint8_t>(0);  // end of header
        const int32_t line_bytes = kWidth * 4 * static_cast<int32_t>(sizeof(float));
        const uint64_t table_end = w.data().size() + sizeof(uint64_t) * kHeight;
        for (int y = 0; y < kHeight; ++y)
            w.put<uint64_t>(table_end + static_cast<uint64_t>(y) * (8 + line_bytes));
        for (int y = 0; y < kHeight; ++y) {
            w.put<int32_t>(y);
            w.put<int32_t>(line_bytes);
            for (const float v : {kAlpha, kBlue, kGreen, kRed})
                for (int x = 0; x < kWidth; ++x)
                    w.put<float>(v);
        }
        std::ofstream(path, std::ios::binary).write(w.data().data(), static_cast<std::streamsize>(w.data().size()));
        return path;
    }

    struct Channels {
        float r, g, b, a;
    };

    Channels mean_channels(const lfs::core::Tensor& chw) {
        const auto cpu = chw.to(lfs::core::Device::CPU).contiguous();
        const size_t plane = cpu.shape()[1] * cpu.shape()[2];
        const float* data = cpu.ptr<float>();
        float sums[4] = {0, 0, 0, 0};
        for (size_t c = 0; c < 4; ++c)
            for (size_t i = 0; i < plane; ++i)
                sums[c] += data[c * plane + i];
        return {sums[0] / plane, sums[1] / plane, sums[2] / plane, sums[3] / plane};
    }
} // namespace

TEST(HdrRgbaLoadTest, FloatDecodeKeepsRadiancePastOne) {
    const auto path = write_rgba_exr();
    const auto rgba = lfs::io::load_rgba_image_cpu_decoded(path, 1, 0, nullptr, true, false, nullptr, true);
    std::filesystem::remove(path);
    ASSERT_EQ(rgba.ndim(), 3u);
    ASSERT_EQ(rgba.shape()[0], 4u);
    ASSERT_EQ(rgba.dtype(), lfs::core::DataType::Float32);
    const auto m = mean_channels(rgba);
    EXPECT_NEAR(m.r, kRed, 1e-2f);
    EXPECT_NEAR(m.g, kGreen, 1e-3f);
    EXPECT_NEAR(m.b, kBlue, 1e-3f);
    EXPECT_NEAR(m.a, kAlpha, 1e-3f);
}

TEST(HdrRgbaLoadTest, FloatDecodeKeepsRadiancePastOneWhenResized) {
    const auto path = write_rgba_exr();
    const auto rgba = lfs::io::load_rgba_image_cpu_decoded(path, 1, kWidth / 2, nullptr, true, false, nullptr, true);
    std::filesystem::remove(path);
    ASSERT_EQ(rgba.shape()[2], static_cast<size_t>(kWidth / 2));
    const auto m = mean_channels(rgba);
    EXPECT_NEAR(m.r, kRed, 0.05f);   // a flat image resamples to itself
    EXPECT_NEAR(m.a, kAlpha, 1e-3f);
}

TEST(HdrRgbaLoadTest, DisplayDecodeStillQuantisesToUnitRange) {
    // The LDR path is unchanged: without decode_float the colour is on the [0, 1] grid.
    const auto path = write_rgba_exr();
    const auto rgba = lfs::io::load_rgba_image_cpu_decoded(path, 1, 0, nullptr, true, false, nullptr, false);
    std::filesystem::remove(path);
    const auto m = mean_channels(rgba);
    EXPECT_LE(m.r, 1.0f);
    EXPECT_NEAR(m.a, kAlpha, 1e-3f);
}
