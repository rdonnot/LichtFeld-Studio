/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include <array>
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::training {

    // The bracket set's camera response as crf.json stores it (PLAN.md M3, docs/hdr-m3-design.md §1, §4): Debevec
    // g_c[z] = ln(E * t) for code z of channel c, anchored g(128) = 0, plus the EXIF anchor of its x-axis:
    // exif_ev_ref = log2(iso_ref / fnumber_ref^2), so a frame with exif_ev = log2(t * ISO / N^2) exposes the
    // radiance by 2^(exif_ev - exif_ev_ref) = t at the shoot's ISO and f-number. Each g_c is made strictly
    // increasing exactly as tools/eval/hdr_metrics.apply_crf does (running maximum, then a 1e-12 * z ramp).
    struct LFS_CUDA_API HdrBracketCrf {
        std::array<std::array<float, 256>, 3> log_irradiance{};
        float exif_ev_ref = 0.0f;
        std::string source; // where it came from, for the logs

        static std::expected<HdrBracketCrf, std::string> load(const std::filesystem::path& crf_json);
        // From raw curves (1 or 3 of 256 values); repairs monotonicity like the Python tooling.
        static std::expected<HdrBracketCrf, std::string> from_curves(const std::vector<std::vector<double>>& curves,
                                                                     float exif_ev_ref);
        // Code z of channel c as linear exposure x = exp(g_c[z]) (the LUT's x-axis unit).
        [[nodiscard]] double exposure_of_code(int channel, int code) const;
    };

    // Exposure + fixed Debevec CRF + saturation gate between the linear render and the LDR-domain loss (§3-§5).
    // Per frame uid: x = v * 2^(b + delta); b from EXIF (constant), delta learnable (Adam, prior toward 0) or
    // frozen (`exif`). Not a PPISP mode: PPISP's CRF clamps its input to [0,1] and its exposure anchor pins the
    // MEAN; here the EXIF is the absolute anchor of the radiance units.
    class LFS_CUDA_API HdrBracketStage {
    public:
        struct Config {
            bool learn = true;        // false: delta frozen at 0 (hdr_exposure = exif)
            float lr = 2e-3f;         // Adam on delta
            float prior = 1.0f;       // loss += prior * delta^2 (per frame seen)
            float max_delta = 3.0f;   // |delta| clamp, stops
            float gate_high = 250.0f / 255.0f;
            float gate_low = 5.0f / 255.0f;
        };

        struct Diagnostics {
            int frames = 0;
            float mean_delta = 0.0f;
            float mean_abs_delta = 0.0f;
            float max_abs_delta = 0.0f;
        };

        HdrBracketStage(HdrBracketCrf crf, Config config);
        HdrBracketStage(); // state-only shell for deserialize()/adopt_checkpoint_state()
        ~HdrBracketStage();
        HdrBracketStage(const HdrBracketStage&) = delete;
        HdrBracketStage& operator=(const HdrBracketStage&) = delete;

        // b = exif_ev - exif_ev_ref (log2 gain). Register every frame (training and held-out) before finalize().
        void register_frame(int uid, float b);
        void finalize();
        [[nodiscard]] bool is_finalized() const noexcept { return finalized_; }
        [[nodiscard]] int num_frames() const noexcept { return static_cast<int>(uids_.size()); }
        [[nodiscard]] bool is_known_frame(int uid) const noexcept { return frame_of_uid_.contains(uid); }
        [[nodiscard]] float b(int uid) const;
        [[nodiscard]] float delta(int uid) const; // device -> host read
        [[nodiscard]] std::vector<float> deltas() const;
        [[nodiscard]] std::vector<int> uids() const { return uids_; }
        [[nodiscard]] const HdrBracketCrf& crf() const noexcept { return crf_; }
        [[nodiscard]] const Config& config() const noexcept { return config_; }
        [[nodiscard]] bool learns() const noexcept { return config_.learn; }

        // Training forward for frame uid: the gated LDR prediction z' [3,H,W] Float32 for the loss. gt is the
        // frame's JPEG (UInt8 codes or Float32 in [0,1]), same shape as v. Keeps what the backward needs.
        [[nodiscard]] lfs::core::Tensor apply(const lfs::core::Tensor& v, int uid, const lfs::core::Tensor& gt);
        // In place: dL/dz' -> dL/dv for the frame of the last apply(); accumulates dL/ddelta when learning.
        void backward_in_place(lfs::core::Tensor& grad, int uid);
        // Adam step on delta[uid] with the prior (no-op when frozen); call after backward_in_place.
        void step(int uid);

        // Evaluation / held-out frames: delta = 0, no gate; the LDR prediction at log2 gain b (unclamped).
        [[nodiscard]] lfs::core::Tensor apply_eval(const lfs::core::Tensor& v, float b) const;
        [[nodiscard]] lfs::core::Tensor apply_eval(const lfs::core::Tensor& v, int uid) const;

        // B2: Debevec's hat of the frame's JPEG as a per-pixel loss / error-map weight, [H,W] (kernels doc).
        [[nodiscard]] const lfs::core::Tensor& hat_weight(const lfs::core::Tensor& gt, float p);

        [[nodiscard]] Diagnostics diagnostics() const;
        [[nodiscard]] float grad_delta_for_test(int uid) const; // the accumulated dL/ddelta before step()
        void set_delta_for_test(int uid, float value);

        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);
        // Copies delta / Adam state / step counts of the frames whose uid this stage knows.
        void adopt_checkpoint_state(const HdrBracketStage& loaded);
        static void consume_checkpoint(std::istream& is);

    private:
        [[nodiscard]] int frame_index(int uid) const;
        void upload_lut();

        HdrBracketCrf crf_{};
        Config config_{};
        bool finalized_ = false;
        std::vector<int> uids_;
        std::vector<float> b_host_;
        std::unordered_map<int, int> frame_of_uid_;
        std::vector<int> step_count_;

        lfs::core::Tensor lut_;        // [3,256] Float32 CUDA
        lfs::core::Tensor b_;          // [N]
        lfs::core::Tensor delta_;      // [N]
        lfs::core::Tensor exp_avg_;    // [N]
        lfs::core::Tensor exp_avg_sq_; // [N]
        lfs::core::Tensor grad_delta_; // [N]

        // Saved by apply() for backward_in_place(): the render handle and the masked dz/du.
        lfs::core::Tensor hat_;        // [H,W], the last hat_weight()
        lfs::core::Tensor saved_v_;
        lfs::core::Tensor saved_dzdu_;
        int saved_uid_ = -1;
        cudaStream_t last_stream_ = nullptr;
    };

} // namespace lfs::training
