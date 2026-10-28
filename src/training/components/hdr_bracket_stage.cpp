/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "components/hdr_bracket_stage.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "kernels/hdr_bracket.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <istream>
#include <limits>
#include <nlohmann/json.hpp>
#include <ostream>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t kStageMagic = 0x48425253; // "HBRS"
        constexpr uint32_t kStageVersion = 1;
        constexpr int kCodes = 256;

        template <typename T>
        void write_pod(std::ostream& os, const T& value) {
            os.write(reinterpret_cast<const char*>(&value), sizeof(T));
        }

        template <typename T>
        T read_pod(std::istream& is, const char* what) {
            T value{};
            lfs::core::serialization_detail::read_exact(is, &value, sizeof(T), what);
            return value;
        }

        // Strictly increasing in float32: running maximum (the Python tooling's maximum.accumulate), then every
        // tie or inversion lifted to the next representable float (the tooling's 1e-12 ramp vanishes in float32,
        // so the lift is one ulp — ~1e-7 nats, below anything the loss can see).
        std::array<float, 256> strictly_increasing(const std::vector<double>& curve) {
            std::array<float, 256> g{};
            double running = -std::numeric_limits<double>::infinity();
            for (int z = 0; z < kCodes; ++z) {
                running = std::max(running, curve[static_cast<size_t>(z)]);
                g[static_cast<size_t>(z)] = static_cast<float>(running);
            }
            for (int z = 1; z < kCodes; ++z) {
                const float prev = g[static_cast<size_t>(z - 1)];
                if (!(g[static_cast<size_t>(z)] > prev))
                    g[static_cast<size_t>(z)] = std::nextafter(prev, std::numeric_limits<float>::infinity());
            }
            return g;
        }
    } // namespace

    // ── HdrBracketCrf ──────────────────────────────────────────────────────────────────────────────────────────

    std::expected<HdrBracketCrf, std::string> HdrBracketCrf::from_curves(
        const std::vector<std::vector<double>>& curves, const float exif_ev_ref) {
        if (curves.size() != 1 && curves.size() != 3)
            return std::unexpected(std::format("crf: log_irradiance must have 1 or 3 channels (got {})", curves.size()));
        for (const auto& curve : curves) {
            if (curve.size() != static_cast<size_t>(kCodes))
                return std::unexpected(std::format("crf: each log_irradiance curve needs 256 values (got {})", curve.size()));
            for (const double value : curve)
                if (!std::isfinite(value))
                    return std::unexpected("crf: log_irradiance contains a non-finite value");
            if (!(curve.back() > curve.front()))
                return std::unexpected("crf: log_irradiance must increase from code 0 to code 255");
        }
        if (!std::isfinite(exif_ev_ref))
            return std::unexpected("crf: exif_ev_ref must be finite");
        HdrBracketCrf crf;
        for (int c = 0; c < 3; ++c)
            crf.log_irradiance[static_cast<size_t>(c)] =
                strictly_increasing(curves[curves.size() == 1 ? 0 : static_cast<size_t>(c)]);
        crf.exif_ev_ref = exif_ev_ref;
        return crf;
    }

    std::expected<HdrBracketCrf, std::string> HdrBracketCrf::load(const std::filesystem::path& crf_json) {
        std::ifstream file(crf_json, std::ios::binary);
        if (!file)
            return std::unexpected(std::format("cannot read the bracket CRF '{}' (--hdr-crf)",
                                               lfs::core::path_to_utf8(crf_json)));
        nlohmann::json json;
        try {
            json = nlohmann::json::parse(file);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("'{}' is not JSON: {}", lfs::core::path_to_utf8(crf_json), e.what()));
        }
        if (!json.contains("log_irradiance"))
            return std::unexpected(std::format("'{}' has no log_irradiance (a Debevec crf.json has [3][256] or [256])",
                                               lfs::core::path_to_utf8(crf_json)));
        if (!json.contains("exif_ev_ref"))
            return std::unexpected(std::format(
                "'{}' has no exif_ev_ref: bracket training needs the EXIF anchor of the LUT's x-axis "
                "(log2(iso_ref / fnumber_ref^2)); the hdr_brackets plugin's M3 bracket set writes it to "
                "<out>/brackets/crf.json",
                lfs::core::path_to_utf8(crf_json)));
        std::vector<std::vector<double>> curves;
        try {
            const auto& li = json.at("log_irradiance");
            if (li.is_array() && !li.empty() && li.front().is_array())
                curves = li.get<std::vector<std::vector<double>>>();
            else
                curves = {li.get<std::vector<double>>()};
            auto crf = from_curves(curves, json.at("exif_ev_ref").get<float>());
            if (!crf)
                return std::unexpected(std::format("'{}': {}", lfs::core::path_to_utf8(crf_json), crf.error()));
            crf->source = lfs::core::path_to_utf8(crf_json);
            return crf;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("'{}': {}", lfs::core::path_to_utf8(crf_json), e.what()));
        }
    }

    double HdrBracketCrf::exposure_of_code(const int channel, const int code) const {
        return std::exp(static_cast<double>(
            log_irradiance[static_cast<size_t>(std::clamp(channel, 0, 2))][static_cast<size_t>(std::clamp(code, 0, 255))]));
    }

    // ── HdrBracketStage ────────────────────────────────────────────────────────────────────────────────────────

    HdrBracketStage::HdrBracketStage(HdrBracketCrf crf, Config config)
        : crf_(std::move(crf)),
          config_(config) {
        upload_lut();
    }

    HdrBracketStage::HdrBracketStage() = default;
    HdrBracketStage::~HdrBracketStage() = default;

    void HdrBracketStage::upload_lut() {
        std::vector<float> flat;
        flat.reserve(3 * kCodes);
        for (const auto& channel : crf_.log_irradiance)
            flat.insert(flat.end(), channel.begin(), channel.end());
        lut_ = lfs::core::Tensor::from_vector(flat, lfs::core::TensorShape({3, static_cast<size_t>(kCodes)}),
                                              lfs::core::Device::CUDA);
    }

    void HdrBracketStage::register_frame(const int uid, const float b) {
        if (finalized_)
            throw std::logic_error("HdrBracketStage: register_frame after finalize");
        if (!std::isfinite(b))
            throw std::invalid_argument(std::format("HdrBracketStage: frame {} has a non-finite log2 gain", uid));
        if (const auto it = frame_of_uid_.find(uid); it != frame_of_uid_.end()) {
            b_host_[static_cast<size_t>(it->second)] = b;
            return;
        }
        frame_of_uid_.emplace(uid, static_cast<int>(uids_.size()));
        uids_.push_back(uid);
        b_host_.push_back(b);
    }

    void HdrBracketStage::finalize() {
        if (uids_.empty())
            throw std::logic_error("HdrBracketStage: no frames registered");
        const auto n = uids_.size();
        b_ = lfs::core::Tensor::from_vector(b_host_, lfs::core::TensorShape({n}), lfs::core::Device::CUDA);
        delta_ = lfs::core::Tensor::zeros(lfs::core::TensorShape({n}), lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::zeros(lfs::core::TensorShape({n}), lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::zeros(lfs::core::TensorShape({n}), lfs::core::Device::CUDA);
        grad_delta_ = lfs::core::Tensor::zeros(lfs::core::TensorShape({n}), lfs::core::Device::CUDA);
        step_count_.assign(n, 0);
        finalized_ = true;
    }

    int HdrBracketStage::frame_index(const int uid) const {
        const auto it = frame_of_uid_.find(uid);
        if (it == frame_of_uid_.end())
            throw std::invalid_argument(std::format("HdrBracketStage: frame uid {} was not registered", uid));
        return it->second;
    }

    float HdrBracketStage::b(const int uid) const {
        return b_host_[static_cast<size_t>(frame_index(uid))];
    }

    float HdrBracketStage::delta(const int uid) const {
        if (!finalized_)
            return 0.0f;
        return delta_.to_vector()[static_cast<size_t>(frame_index(uid))];
    }

    std::vector<float> HdrBracketStage::deltas() const {
        if (!finalized_)
            return std::vector<float>(uids_.size(), 0.0f);
        return delta_.to_vector();
    }

    lfs::core::Tensor HdrBracketStage::apply(const lfs::core::Tensor& v, const int uid, const lfs::core::Tensor& gt) {
        if (!finalized_)
            throw std::logic_error("HdrBracketStage::apply before finalize");
        const int frame = frame_index(uid);
        const cudaStream_t stream = v.stream();
        lut_.sync_to_stream(stream);
        b_.sync_to_stream(stream);
        delta_.sync_to_stream(stream);
        kernels::HdrBracketForwardArgs args;
        args.log_irradiance = lut_.ptr<float>();
        args.b = b_.ptr<float>();
        args.delta = config_.learn ? delta_.ptr<float>() : nullptr;
        args.frame = frame;
        args.gate_high = config_.gate_high;
        args.gate_low = config_.gate_low;
        saved_v_ = v.is_contiguous() ? v : v.contiguous();
        auto z = kernels::hdr_bracket_forward(saved_v_, args, &gt, &saved_dzdu_);
        saved_uid_ = uid;
        return z;
    }

    void HdrBracketStage::backward_in_place(lfs::core::Tensor& grad, const int uid) {
        if (saved_uid_ != uid || !saved_v_.is_valid() || !saved_dzdu_.is_valid())
            throw std::logic_error(std::format("HdrBracketStage::backward_in_place: no forward saved for frame {}", uid));
        const int frame = frame_index(uid);
        float* grad_delta = nullptr;
        if (config_.learn) {
            grad_delta_.sync_to_stream(grad.stream());
            grad_delta = grad_delta_.ptr<float>() + frame;
        }
        kernels::hdr_bracket_backward_(grad, saved_v_, saved_dzdu_, grad_delta);
        last_stream_ = grad.stream();
        saved_v_ = {};
        saved_dzdu_ = {};
        saved_uid_ = -1;
    }

    void HdrBracketStage::step(const int uid) {
        if (!config_.learn || !finalized_)
            return;
        const int frame = frame_index(uid);
        auto& count = step_count_[static_cast<size_t>(frame)];
        ++count;
        kernels::HdrBracketAdamArgs args;
        args.lr = config_.lr;
        args.prior = config_.prior;
        args.max_delta = config_.max_delta;
        args.step = count;
        const cudaStream_t stream = last_stream_ != nullptr ? last_stream_ : grad_delta_.stream();
        delta_.sync_to_stream(stream);
        exp_avg_.sync_to_stream(stream);
        exp_avg_sq_.sync_to_stream(stream);
        grad_delta_.sync_to_stream(stream);
        kernels::hdr_bracket_adam_step(delta_.ptr<float>(), exp_avg_.ptr<float>(), exp_avg_sq_.ptr<float>(),
                                       grad_delta_.ptr<float>(), frame, args, stream);
    }

    lfs::core::Tensor HdrBracketStage::apply_eval(const lfs::core::Tensor& v, const float b) const {
        if (!lut_.is_valid())
            throw std::logic_error("HdrBracketStage::apply_eval without a CRF");
        const cudaStream_t stream = v.stream();
        lut_.sync_to_stream(stream);
        auto gain = lfs::core::Tensor::from_vector(std::vector<float>{b}, lfs::core::TensorShape({1}),
                                                   lfs::core::Device::CUDA);
        gain.sync_to_stream(stream);
        kernels::HdrBracketForwardArgs args;
        args.log_irradiance = lut_.ptr<float>();
        args.b = gain.ptr<float>();
        args.delta = nullptr;
        args.frame = 0;
        return kernels::hdr_bracket_forward(v, args, nullptr, nullptr);
    }

    lfs::core::Tensor HdrBracketStage::apply_eval(const lfs::core::Tensor& v, const int uid) const {
        return apply_eval(v, b(uid));
    }

    HdrBracketStage::Diagnostics HdrBracketStage::diagnostics() const {
        Diagnostics d;
        d.frames = num_frames();
        if (!finalized_ || !config_.learn || d.frames == 0)
            return d;
        const auto values = delta_.to_vector();
        double sum = 0.0, sum_abs = 0.0;
        float max_abs = 0.0f;
        for (const float value : values) {
            sum += value;
            sum_abs += std::abs(value);
            max_abs = std::max(max_abs, std::abs(value));
        }
        d.mean_delta = static_cast<float>(sum / static_cast<double>(values.size()));
        d.mean_abs_delta = static_cast<float>(sum_abs / static_cast<double>(values.size()));
        d.max_abs_delta = max_abs;
        return d;
    }

    float HdrBracketStage::grad_delta_for_test(const int uid) const {
        if (!finalized_)
            return 0.0f;
        return grad_delta_.to_vector()[static_cast<size_t>(frame_index(uid))];
    }

    void HdrBracketStage::set_delta_for_test(const int uid, const float value) {
        if (!finalized_)
            throw std::logic_error("HdrBracketStage::set_delta_for_test before finalize");
        auto values = delta_.to_vector();
        values[static_cast<size_t>(frame_index(uid))] = value;
        delta_ = lfs::core::Tensor::from_vector(values, lfs::core::TensorShape({values.size()}), lfs::core::Device::CUDA);
    }

    // ── checkpoint ─────────────────────────────────────────────────────────────────────────────────────────────

    void HdrBracketStage::serialize(std::ostream& os) const {
        write_pod(os, kStageMagic);
        write_pod(os, kStageVersion);
        write_pod(os, static_cast<uint8_t>(config_.learn ? 1 : 0));
        write_pod(os, config_.lr);
        write_pod(os, config_.prior);
        write_pod(os, config_.max_delta);
        write_pod(os, crf_.exif_ev_ref);
        const auto n = static_cast<uint32_t>(uids_.size());
        write_pod(os, n);
        const auto deltas_host = finalized_ ? delta_.to_vector() : std::vector<float>(n, 0.0f);
        const auto m_host = finalized_ ? exp_avg_.to_vector() : std::vector<float>(n, 0.0f);
        const auto s_host = finalized_ ? exp_avg_sq_.to_vector() : std::vector<float>(n, 0.0f);
        for (uint32_t i = 0; i < n; ++i) {
            write_pod(os, static_cast<int32_t>(uids_[i]));
            write_pod(os, b_host_[i]);
            write_pod(os, deltas_host[i]);
            write_pod(os, m_host[i]);
            write_pod(os, s_host[i]);
            write_pod(os, static_cast<int32_t>(finalized_ ? step_count_[i] : 0));
        }
    }

    void HdrBracketStage::deserialize(std::istream& is) {
        const auto magic = read_pod<uint32_t>(is, "HDR bracket stage magic");
        const auto version = read_pod<uint32_t>(is, "HDR bracket stage version");
        if (magic != kStageMagic)
            throw std::runtime_error("Invalid HDR bracket stage checkpoint");
        if (version != kStageVersion)
            throw std::runtime_error(std::format("Unsupported HDR bracket stage checkpoint version {}", version));
        config_.learn = read_pod<uint8_t>(is, "HDR bracket stage learn") != 0;
        config_.lr = read_pod<float>(is, "HDR bracket stage lr");
        config_.prior = read_pod<float>(is, "HDR bracket stage prior");
        config_.max_delta = read_pod<float>(is, "HDR bracket stage max delta");
        crf_.exif_ev_ref = read_pod<float>(is, "HDR bracket stage exif_ev_ref");
        const auto n = read_pod<uint32_t>(is, "HDR bracket stage frame count");
        if (n == 0 || n > 10'000'000)
            throw std::runtime_error("Invalid HDR bracket stage checkpoint: frame count");
        uids_.clear();
        b_host_.clear();
        frame_of_uid_.clear();
        std::vector<float> deltas_host(n), m_host(n), s_host(n);
        step_count_.assign(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            const auto uid = read_pod<int32_t>(is, "HDR bracket stage uid");
            frame_of_uid_.emplace(uid, static_cast<int>(uids_.size()));
            uids_.push_back(uid);
            b_host_.push_back(read_pod<float>(is, "HDR bracket stage b"));
            deltas_host[i] = read_pod<float>(is, "HDR bracket stage delta");
            m_host[i] = read_pod<float>(is, "HDR bracket stage exp_avg");
            s_host[i] = read_pod<float>(is, "HDR bracket stage exp_avg_sq");
            step_count_[i] = read_pod<int32_t>(is, "HDR bracket stage step");
        }
        const lfs::core::TensorShape shape({static_cast<size_t>(n)});
        b_ = lfs::core::Tensor::from_vector(b_host_, shape, lfs::core::Device::CUDA);
        delta_ = lfs::core::Tensor::from_vector(deltas_host, shape, lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::from_vector(m_host, shape, lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::from_vector(s_host, shape, lfs::core::Device::CUDA);
        grad_delta_ = lfs::core::Tensor::zeros(shape, lfs::core::Device::CUDA);
        finalized_ = true;
    }

    void HdrBracketStage::consume_checkpoint(std::istream& is) {
        HdrBracketStage shell;
        shell.deserialize(is);
    }

    void HdrBracketStage::adopt_checkpoint_state(const HdrBracketStage& loaded) {
        if (!finalized_)
            throw std::logic_error("HdrBracketStage::adopt_checkpoint_state before finalize");
        if (std::abs(loaded.crf_.exif_ev_ref - crf_.exif_ev_ref) > 1e-4f)
            LOG_WARN("HDR bracket stage: checkpoint exif_ev_ref {} differs from the CRF's {}; the deltas are kept as saved",
                     loaded.crf_.exif_ev_ref, crf_.exif_ev_ref);
        auto deltas_host = delta_.to_vector();
        auto m_host = exp_avg_.to_vector();
        auto s_host = exp_avg_sq_.to_vector();
        const auto loaded_delta = loaded.delta_.to_vector();
        const auto loaded_m = loaded.exp_avg_.to_vector();
        const auto loaded_s = loaded.exp_avg_sq_.to_vector();
        size_t matched = 0;
        for (size_t j = 0; j < loaded.uids_.size(); ++j) {
            const auto it = frame_of_uid_.find(loaded.uids_[j]);
            if (it == frame_of_uid_.end())
                continue;
            const auto i = static_cast<size_t>(it->second);
            deltas_host[i] = loaded_delta[j];
            m_host[i] = loaded_m[j];
            s_host[i] = loaded_s[j];
            step_count_[i] = loaded.step_count_[j];
            ++matched;
        }
        if (matched != loaded.uids_.size() || matched != uids_.size())
            LOG_WARN("HDR bracket stage: checkpoint frames {} vs {} registered, {} matched by uid",
                     loaded.uids_.size(), uids_.size(), matched);
        const lfs::core::TensorShape shape({uids_.size()});
        delta_ = lfs::core::Tensor::from_vector(deltas_host, shape, lfs::core::Device::CUDA);
        exp_avg_ = lfs::core::Tensor::from_vector(m_host, shape, lfs::core::Device::CUDA);
        exp_avg_sq_ = lfs::core::Tensor::from_vector(s_host, shape, lfs::core::Device::CUDA);
        grad_delta_ = lfs::core::Tensor::zeros(shape, lfs::core::Device::CUDA);
    }

} // namespace lfs::training
