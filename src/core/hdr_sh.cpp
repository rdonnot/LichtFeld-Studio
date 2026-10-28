/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/hdr_sh.hpp"
#include "core/cuda/hdr_sh_kernels.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <format>
#include <map>
#include <mutex>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace lfs::core::hdr {

    // ── Options ──────────────────────────────────────────────────────────────────────────────────

    namespace {
        constexpr std::array<std::pair<std::string_view, ToneOperator>, 8> kToneNames{{
            {"none", ToneOperator::None},
            {"linear", ToneOperator::Linear},
            {"filmic", ToneOperator::Filmic},
            {"hejl", ToneOperator::Hejl},
            {"aces", ToneOperator::Aces},
            {"aces2", ToneOperator::Aces2},
            {"neutral", ToneOperator::Neutral},
            {"ocio", ToneOperator::Ocio},
        }};
        constexpr std::array<std::pair<std::string_view, ExportMode>, 4> kModeNames{{
            {"auto", ExportMode::Auto},
            {"linear", ExportMode::LinearRadiance},
            {"baked", ExportMode::BakedLdr},
            {"log", ExportMode::LogInternal},
        }};

        std::string lowercase(std::string_view s) {
            std::string out(s);
            for (auto& c : out)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return out;
        }
    } // namespace

    std::optional<ToneOperator> parse_tone_operator(const std::string_view name) {
        const auto key = lowercase(name);
        for (const auto& [n, op] : kToneNames)
            if (n == key)
                return op;
        return std::nullopt;
    }

    std::string_view tone_operator_name(const ToneOperator op) {
        for (const auto& [n, o] : kToneNames)
            if (o == op)
                return n;
        return "aces";
    }

    std::optional<ExportMode> parse_export_mode(const std::string_view name) {
        const auto key = lowercase(name);
        for (const auto& [n, m] : kModeNames)
            if (n == key)
                return m;
        return std::nullopt;
    }

    std::string_view export_mode_name(const ExportMode mode) {
        for (const auto& [n, m] : kModeNames)
            if (m == mode)
                return n;
        return "auto";
    }

    std::string refusal_message(const std::string_view format_name) {
        return std::format(
            "{} cannot hold an HDR model (its SH are log radiance and the format quantises colour to [0,1]). "
            "Two ways out: export a PLY (linear radiance, `lichtfeld_hdr linear`, read by any splat tool), or "
            "bake to LDR at an exposure (--hdr-export baked --hdr-export-ev <EV> --hdr-tonemap <op>; "
            "hdr=\"baked\" in Python; the HDR row of the export panel).",
            format_name);
    }

    // ── SH basis and the projection ──────────────────────────────────────────────────────────────

    int projection_direction_count(const int sh_degree) {
        switch (sh_degree) {
        case 0: return 1;
        case 1: return 32;
        case 2: return 64;
        default: return 128;
        }
    }

    // fastgs convert_sh_to_color ordering and signs (kernel_utils.cuh), evaluated in double.
    void sh_basis(const float xf, const float yf, const float zf, const int K, double* out) {
        const double x = xf, y = yf, z = zf;
        out[0] = 1.0;
        if (K < 3)
            return;
        out[1] = -0.48860251190291987 * y;
        out[2] = 0.48860251190291987 * z;
        out[3] = -0.48860251190291987 * x;
        if (K < 8)
            return;
        const double xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z;
        out[4] = 1.0925484305920792 * xy;
        out[5] = -1.0925484305920792 * yz;
        out[6] = 0.94617469575755997 * zz - 0.31539156525251999;
        out[7] = -1.0925484305920792 * xz;
        out[8] = 0.54627421529603959 * xx - 0.54627421529603959 * yy;
        if (K < 15)
            return;
        out[9] = 0.59004358992664352 * y * (-3.0 * xx + yy);
        out[10] = 2.8906114426405538 * xy * z;
        out[11] = 0.45704579946446572 * y * (1.0 - 5.0 * zz);
        out[12] = 0.3731763325901154 * z * (5.0 * zz - 3.0);
        out[13] = 0.45704579946446572 * x * (1.0 - 5.0 * zz);
        out[14] = 1.4453057213202769 * z * (xx - yy);
        out[15] = 0.59004358992664352 * x * (-xx + 3.0 * yy);
    }

    namespace {

        int degree_for_rest(const size_t K) {
            switch (K) {
            case 0: return 0;
            case 3: return 1;
            case 8: return 2;
            case 15: return 3;
            default:
                throw std::invalid_argument(std::format("HDR SH projection: {} rest coefficients is not an SH degree", K));
            }
        }

        // Fibonacci sphere: D quasi-uniform unit directions.
        std::vector<std::array<float, 3>> fibonacci_directions(const int D) {
            std::vector<std::array<float, 3>> dirs(static_cast<size_t>(D));
            const double golden = std::numbers::pi * (3.0 - std::sqrt(5.0));
            for (int i = 0; i < D; ++i) {
                const double z = D == 1 ? 0.0 : 1.0 - 2.0 * (i + 0.5) / D;
                const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
                const double phi = golden * i;
                dirs[static_cast<size_t>(i)] = {static_cast<float>(r * std::cos(phi)),
                                                static_cast<float>(r * std::sin(phi)), static_cast<float>(z)};
            }
            return dirs;
        }

        // Gauss-Jordan with partial pivoting on a small dense matrix (row-major n x n), in place.
        void invert_in_place(std::vector<double>& m, const int n) {
            std::vector<double> inv(static_cast<size_t>(n) * n, 0.0);
            for (int i = 0; i < n; ++i)
                inv[static_cast<size_t>(i) * n + i] = 1.0;
            for (int col = 0; col < n; ++col) {
                int pivot = col;
                for (int r = col + 1; r < n; ++r)
                    if (std::abs(m[static_cast<size_t>(r) * n + col]) > std::abs(m[static_cast<size_t>(pivot) * n + col]))
                        pivot = r;
                if (std::abs(m[static_cast<size_t>(pivot) * n + col]) < 1e-12)
                    throw std::runtime_error("HDR SH projection: singular normal matrix");
                if (pivot != col) {
                    for (int c = 0; c < n; ++c) {
                        std::swap(m[static_cast<size_t>(col) * n + c], m[static_cast<size_t>(pivot) * n + c]);
                        std::swap(inv[static_cast<size_t>(col) * n + c], inv[static_cast<size_t>(pivot) * n + c]);
                    }
                }
                const double p = m[static_cast<size_t>(col) * n + col];
                for (int c = 0; c < n; ++c) {
                    m[static_cast<size_t>(col) * n + c] /= p;
                    inv[static_cast<size_t>(col) * n + c] /= p;
                }
                for (int r = 0; r < n; ++r) {
                    if (r == col)
                        continue;
                    const double f = m[static_cast<size_t>(r) * n + col];
                    if (f == 0.0)
                        continue;
                    for (int c = 0; c < n; ++c) {
                        m[static_cast<size_t>(r) * n + c] -= f * m[static_cast<size_t>(col) * n + c];
                        inv[static_cast<size_t>(r) * n + c] -= f * inv[static_cast<size_t>(col) * n + c];
                    }
                }
            }
            m.swap(inv);
        }

        struct Projection {
            int K = 0;        // rest coefficients
            int D = 1;        // directions
            Tensor basis;     // [K+1, D] CUDA: row 0 = 1 (DC carrier), rows 1..K = Y_i(w_d)
            Tensor pinv_t;    // [D, K+1] CUDA: least-squares projection, coefficients = values · pinv_t
        };

        Projection make_projection(const int K) {
            Projection p;
            p.K = K;
            p.D = projection_direction_count(degree_for_rest(static_cast<size_t>(K)));
            const int B = K + 1;
            const auto dirs = fibonacci_directions(p.D);
            std::vector<double> Bmat(static_cast<size_t>(p.D) * B); // [D, B]
            std::array<double, 16> row{};
            for (int d = 0; d < p.D; ++d) {
                sh_basis(dirs[static_cast<size_t>(d)][0], dirs[static_cast<size_t>(d)][1], dirs[static_cast<size_t>(d)][2], K, row.data());
                for (int k = 0; k < B; ++k)
                    Bmat[static_cast<size_t>(d) * B + k] = row[static_cast<size_t>(k)];
            }
            // Normal matrix M = Bᵀ B, then pinv_t = B M⁻¹  ([D, B]).
            std::vector<double> M(static_cast<size_t>(B) * B, 0.0);
            for (int d = 0; d < p.D; ++d)
                for (int i = 0; i < B; ++i)
                    for (int j = 0; j < B; ++j)
                        M[static_cast<size_t>(i) * B + j] += Bmat[static_cast<size_t>(d) * B + i] * Bmat[static_cast<size_t>(d) * B + j];
            invert_in_place(M, B);
            std::vector<float> basis_f(static_cast<size_t>(B) * p.D);
            std::vector<float> pinv_f(static_cast<size_t>(p.D) * B);
            for (int d = 0; d < p.D; ++d) {
                for (int k = 0; k < B; ++k) {
                    basis_f[static_cast<size_t>(k) * p.D + d] = static_cast<float>(Bmat[static_cast<size_t>(d) * B + k]);
                    double acc = 0.0;
                    for (int j = 0; j < B; ++j)
                        acc += Bmat[static_cast<size_t>(d) * B + j] * M[static_cast<size_t>(j) * B + k];
                    pinv_f[static_cast<size_t>(d) * B + k] = static_cast<float>(acc);
                }
            }
            p.basis = Tensor::from_vector(basis_f, TensorShape{static_cast<size_t>(B), static_cast<size_t>(p.D)}, Device::CUDA);
            p.pinv_t = Tensor::from_vector(pinv_f, TensorShape{static_cast<size_t>(p.D), static_cast<size_t>(B)}, Device::CUDA);
            return p;
        }

        constexpr size_t kChunk = 65536;

        // The OCIO display LUT on the device (core/ocio.hpp), cached by its key. Throws when OCIO cannot build it:
        // the CLI / Python validate the selection before any bake, so this is the rare path.
        struct DeviceLut {
            Tensor data; // [3 * size^3] float32 CUDA
            cuda::OcioLutRef ref;
        };
        DeviceLut device_lut(const ocio::Selection& selection) {
            static std::mutex mutex;
            static std::map<std::string, DeviceLut> cache;
            std::string error;
            const auto lut = ocio::lut_for(selection, ocio::kDefaultLutSize, &error);
            if (!lut)
                throw std::runtime_error(std::format("HDR bake through OCIO is not possible: {}", error));
            std::lock_guard lock(mutex);
            if (auto it = cache.find(lut->key); it != cache.end())
                return it->second;
            DeviceLut d;
            d.data = Tensor::from_vector(lut->rgb, TensorShape{lut->rgb.size()}, Device::CUDA);
            d.ref = cuda::OcioLutRef{d.data.ptr<float>(), lut->size, lut->log2_lo, lut->log2_hi};
            cache[lut->key] = d;
            LOG_INFO("HDR: OCIO display LUT on the device ({}^3, config '{}')", lut->size, lut->config);
            return d;
        }

        // The shared pipeline: evaluate b on the sphere, apply the per-direction function, project back.
        ShPair project(const Tensor& sh0_in, const Tensor& shN_in, const int mode, const float color_max,
                       const float exposure, const ToneOperator tone, const ocio::Selection* ocio_selection = nullptr) {
            DeviceLut lut;
            if (tone == ToneOperator::Ocio && mode == 2)
                lut = device_lut(ocio_selection ? *ocio_selection : ocio::Selection{});
            if (!sh0_in.is_valid() || sh0_in.numel() == 0)
                return {sh0_in, shN_in};
            if (sh0_in.ndim() != 3 || sh0_in.size(2) != 3 || sh0_in.size(1) != 1)
                throw std::invalid_argument(std::format("HDR SH projection: sh0 must be [N, 1, 3], got {}", sh0_in.shape().str()));
            const bool has_rest = shN_in.is_valid() && shN_in.numel() > 0;
            if (has_rest && (shN_in.ndim() != 3 || shN_in.size(2) != 3 || shN_in.size(0) != sh0_in.size(0)))
                throw std::invalid_argument(std::format("HDR SH projection: shN must be canonical [N, K, 3] matching sh0, got {}",
                                                        shN_in.shape().str()));
            const Tensor sh0 = sh0_in.to(Device::CUDA).to(DataType::Float32).contiguous();
            const Tensor shN = has_rest ? shN_in.to(Device::CUDA).to(DataType::Float32).contiguous() : Tensor{};
            const size_t N = static_cast<size_t>(sh0.size(0));
            const size_t K = has_rest ? static_cast<size_t>(shN.size(1)) : 0;
            const Projection proj = make_projection(static_cast<int>(K));
            const size_t B = K + 1;

            std::vector<Tensor> sh0_chunks, shN_chunks;
            for (size_t start = 0; start < N; start += kChunk) {
                const size_t end = std::min(N, start + kChunk);
                const size_t n = end - start;
                // X [n*3, B]: column 0 = 0.5 + C0 * sh0 (the DC value), columns 1..K the rest coefficients.
                Tensor dc = (sh0.slice(0, start, end) * kShC0 + 0.5f).contiguous().reshape(TensorShape{n * 3, 1});
                Tensor X = dc;
                if (K > 0) {
                    Tensor rest = shN.slice(0, start, end).permute({0, 2, 1}).contiguous().reshape(TensorShape{n * 3, K});
                    X = Tensor::cat({dc, rest}, 1);
                }
                Tensor values = X.mm(proj.basis).contiguous(); // [n*3, D]
                // mode 1 (linear -> log) floors each direction at 1 % of the splat's DC: X's column 0, row stride B.
                cuda::apply_per_direction(values.ptr<float>(), n, static_cast<size_t>(proj.D), mode, color_max, exposure,
                                          static_cast<int>(tone), values.stream(), X.ptr<float>(), B, lut.ref);
                LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(values.stream()), "cudaStreamSynchronize(hdr_sh per-direction)");
                Tensor A = values.mm(proj.pinv_t); // [n*3, B]
                Tensor A3 = A.reshape(TensorShape{n, 3, B}).permute({0, 2, 1}).contiguous(); // [n, B, 3]
                sh0_chunks.push_back(((A3.slice(1, 0, 1) - 0.5f) / kShC0).contiguous());
                if (K > 0)
                    shN_chunks.push_back(A3.slice(1, 1, B).contiguous());
            }
            ShPair out;
            out.sh0 = sh0_chunks.size() == 1 ? sh0_chunks.front() : Tensor::cat(sh0_chunks, 0);
            if (K > 0)
                out.shN = shN_chunks.size() == 1 ? shN_chunks.front() : Tensor::cat(shN_chunks, 0);
            return out;
        }

    } // namespace

    ShPair log_sh_to_linear(const Tensor& sh0, const Tensor& shN_canonical, const float color_max) {
        return project(sh0, shN_canonical, 0, color_max, 1.0f, ToneOperator::None);
    }

    ShPair linear_sh_to_log(const Tensor& sh0, const Tensor& shN_canonical) {
        return project(sh0, shN_canonical, 1, 0.0f, 1.0f, ToneOperator::None);
    }

    ShPair bake_log_sh_to_display(const Tensor& sh0, const Tensor& shN_canonical, const float color_max,
                                  const float exposure, const ToneOperator tone, const ocio::Selection& ocio) {
        return project(sh0, shN_canonical, 2, color_max, exposure, tone, &ocio);
    }

    float display_exposure(const SplatData::HdrState& state, const float exposure_ev) {
        const float s = state.scale > 0.0f ? state.scale : 1.0f;
        return std::exp2(exposure_ev) / s;
    }

    Tensor display_image(const Tensor& linear_chw, const SplatData::HdrState& state, const float exposure_ev,
                         const ToneOperator tone, const ocio::Selection& ocio) {
        if (!linear_chw.is_valid() || linear_chw.ndim() != 3 || linear_chw.size(0) != 3)
            throw std::invalid_argument(std::format("display_image: expected [3, H, W], got {}", linear_chw.shape().str()));
        const size_t h = static_cast<size_t>(linear_chw.size(1)), w = static_cast<size_t>(linear_chw.size(2));
        DeviceLut lut;
        if (tone == ToneOperator::Ocio)
            lut = device_lut(ocio);
        Tensor hwc = linear_chw.to(Device::CUDA).to(DataType::Float32).permute({1, 2, 0}).contiguous(); // [H, W, 3]
        cuda::apply_per_direction(hwc.ptr<float>(), h * w, 1, 3, 0.0f, display_exposure(state, exposure_ev),
                                  static_cast<int>(tone), hwc.stream(), nullptr, 1, lut.ref);
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(hwc.stream()), "cudaStreamSynchronize(hdr_sh image bake)");
        return hwc.permute({2, 0, 1}).contiguous();
    }

    SplatData with_sh(const SplatData& source, const ShPair& pair, const SplatData::HdrState& state) {
        SplatData copy = source.clone();
        if (pair.sh0.is_valid()) {
            Tensor sh0 = pair.sh0.to(source.sh0().device()).contiguous();
            sh0.set_name("splat.sh0");
            copy.sh0() = std::move(sh0);
        }
        if (copy.max_sh_coeffs_rest() > 0 && pair.shN.is_valid() && pair.shN.numel() > 0)
            copy.shN_set_from_canonical(pair.shN);
        copy.set_hdr(state);
        return copy;
    }

    SplatData bake_hdr_to_ldr(const SplatData& source, const float exposure_ev, const ToneOperator tone,
                              const ocio::Selection& ocio) {
        if (!source.is_hdr())
            return source.clone();
        const auto& state = source.hdr();
        const float exposure = display_exposure(state, exposure_ev);
        std::string tone_desc(tone_operator_name(tone));
        if (tone == ToneOperator::Ocio) {
            const auto sel = ocio::resolve(ocio);
            tone_desc = std::format("ocio '{}' / '{}'{} (config '{}')", sel.display, sel.view,
                                    sel.look.empty() ? "" : std::format(" + look '{}'", sel.look), ocio::config_name(sel));
        }
        LOG_INFO("HDR: baking {} splats to LDR at {:+.2f} EV (exposure {:.4g} = 2^EV / s, s = {}), tone {}",
                 source.size(), exposure_ev, exposure, state.scale, tone_desc);
        const ShPair baked = bake_log_sh_to_display(source.sh0(), source.shN_canonical(), state.color_max, exposure, tone, ocio);
        return with_sh(source, baked, SplatData::HdrState{});
    }

    SplatData to_linear_radiance(const SplatData& source) {
        if (!source.is_hdr())
            return source.clone();
        const ShPair linear = log_sh_to_linear(source.sh0(), source.shN_canonical(), source.hdr().color_max);
        return with_sh(source, linear, SplatData::HdrState{});
    }

} // namespace lfs::core::hdr
