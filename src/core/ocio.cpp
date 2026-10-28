/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// OpenColorIO glue (core/ocio.hpp). The only translation unit that includes OCIO.

#include "core/ocio.hpp"
#include "core/logger.hpp"

#include <OpenColorIO/OpenColorIO.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <map>
#include <mutex>

namespace OCIO = OCIO_NAMESPACE;

// The C++ twin of the LUT sampler: the same text the compose shader and the CUDA bake compile.
namespace {
    namespace lut_twin {
#define LFS_COLOR_INLINE inline
#define LFS_COLOR_VEC3 glm::vec3
#define LFS_COLOR_UINT std::uint32_t
#define LFS_COLOR_MIX glm::mix
#define LFS_OCIO_LUT_PARAMS const float* lut,
#define LFS_OCIO_LUT_FETCH(i) glm::vec3(lut[3 * (i)], lut[3 * (i) + 1], lut[3 * (i) + 2])
        using glm::clamp;
        using glm::dot;
        using glm::max;
        using glm::min;
        using glm::pow;
        using std::exp2;
        using std::log;
        using std::log2;
#include "display_color.inc"
#undef LFS_COLOR_INLINE
#undef LFS_COLOR_VEC3
#undef LFS_COLOR_UINT
#undef LFS_COLOR_MIX
#undef LFS_OCIO_LUT_PARAMS
#undef LFS_OCIO_LUT_FETCH
    } // namespace lut_twin
} // namespace

namespace lfs::core::ocio {

    namespace {

        std::mutex g_mutex;
        Selection g_current;
        std::map<std::string, OCIO::ConstConfigRcPtr> g_configs;           // by config_path ("" = env / built-in)
        std::map<std::string, std::shared_ptr<const Lut3D>> g_luts;         // by Lut3D::key
        bool g_warned_scene_linear = false;

        constexpr const char* kBuiltinConfig = "studio-config-latest";

        // Loads (and caches) the config a selection names. Throws OCIO::Exception on failure.
        OCIO::ConstConfigRcPtr config_for_locked(const std::string& config_path) {
            if (auto it = g_configs.find(config_path); it != g_configs.end())
                return it->second;
            OCIO::ConstConfigRcPtr config;
            std::string how;
            if (!config_path.empty()) {
                config = OCIO::Config::CreateFromFile(config_path.c_str());
                how = config_path;
            } else if (const char* env = std::getenv("OCIO"); env != nullptr && env[0] != '\0') {
                config = OCIO::Config::CreateFromEnv();
                how = std::format("$OCIO = {}", env);
            } else {
                config = OCIO::Config::CreateFromBuiltinConfig(kBuiltinConfig);
                how = std::format("built-in ocio://{}", kBuiltinConfig);
            }
            LOG_INFO("OCIO {}: config '{}' ({}; {} displays)", OCIO::GetVersion(), config->getName(), how,
                     config->getNumDisplays());
            g_configs[config_path] = config;
            return config;
        }

        struct Resolved {
            OCIO::ConstConfigRcPtr config;
            Selection selection;
            std::string working;
        };

        std::string working_space_locked(const OCIO::ConstConfigRcPtr& config) {
            for (const char* name : {"lin_srgb", "Linear Rec.709 (sRGB)", "lin_rec709_srgb", "lin_rec709", "linear"}) {
                if (config->getColorSpace(name))
                    return config->getCanonicalName(name);
            }
            const char* role = config->getRoleColorSpace(OCIO::ROLE_SCENE_LINEAR);
            if (role != nullptr && role[0] != '\0' && config->getColorSpace(role)) {
                if (!g_warned_scene_linear) {
                    LOG_WARN("OCIO: config '{}' has no 'lin_srgb' / 'Linear Rec.709 (sRGB)' colorspace; LichtFeld's working "
                             "space is taken as its scene_linear role '{}' — only right if that role is Rec.709-linear",
                             config->getName(), role);
                    g_warned_scene_linear = true;
                }
                return role;
            }
            return {};
        }

        // Fills the empty fields of `s` from the process default and the config's own defaults, validating
        // display / view / look. Throws std::runtime_error with the message on an invalid name.
        Resolved resolve_locked(const Selection& in) {
            Selection s = in;
            if (s.config_path.empty())
                s.config_path = g_current.config_path;
            if (s.display.empty())
                s.display = g_current.display;
            if (s.view.empty() && s.display == g_current.display)
                s.view = g_current.view;
            if (s.look.empty())
                s.look = g_current.look;
            else if (s.look == "none")
                s.look.clear();
            auto config = config_for_locked(s.config_path);
            if (s.display.empty())
                s.display = config->getDefaultDisplay();
            bool display_ok = false;
            for (int i = 0; i < config->getNumDisplays(); ++i)
                display_ok |= s.display == config->getDisplay(i);
            if (!display_ok) {
                std::string names;
                for (int i = 0; i < config->getNumDisplays(); ++i)
                    names += std::format("{}'{}'", i ? ", " : "", config->getDisplay(i));
                throw std::runtime_error(std::format("OCIO: no display '{}' in config '{}'. Displays: {}", s.display,
                                                     config->getName(), names));
            }
            if (s.view.empty())
                s.view = config->getDefaultView(s.display.c_str());
            bool view_ok = false;
            for (int i = 0; i < config->getNumViews(s.display.c_str()); ++i)
                view_ok |= s.view == config->getView(s.display.c_str(), i);
            if (!view_ok) {
                std::string names;
                for (int i = 0; i < config->getNumViews(s.display.c_str()); ++i)
                    names += std::format("{}'{}'", i ? ", " : "", config->getView(s.display.c_str(), i));
                throw std::runtime_error(std::format("OCIO: no view '{}' for display '{}' in config '{}'. Views: {}",
                                                     s.view, s.display, config->getName(), names));
            }
            if (!s.look.empty()) {
                bool look_ok = false;
                for (int i = 0; i < config->getNumLooks(); ++i)
                    look_ok |= s.look == config->getLookNameByIndex(i);
                if (!look_ok) {
                    std::string names;
                    for (int i = 0; i < config->getNumLooks(); ++i)
                        names += std::format("{}'{}'", i ? ", " : "", config->getLookNameByIndex(i));
                    throw std::runtime_error(std::format("OCIO: no look '{}' in config '{}'. Looks: {}", s.look,
                                                         config->getName(), names.empty() ? "(none)" : names));
                }
            }
            const std::string working = working_space_locked(config);
            if (working.empty())
                throw std::runtime_error(std::format(
                    "OCIO: config '{}' has no colorspace LichtFeld can use as its working space (lin_srgb / "
                    "Linear Rec.709 (sRGB) / a scene_linear role)",
                    config->getName()));
            return {config, s, working};
        }

        // working space -> [look] -> display/view. The LookTransform converts to the look's own process space
        // and back; the DisplayViewTransform keeps the view's own looks (no bypass).
        OCIO::ConstCPUProcessorRcPtr display_processor_locked(const Resolved& r) {
            auto group = OCIO::GroupTransform::Create();
            if (!r.selection.look.empty()) {
                auto look = OCIO::LookTransform::Create();
                look->setSrc(r.working.c_str());
                look->setDst(r.working.c_str());
                look->setLooks(r.selection.look.c_str());
                group->appendTransform(look);
            }
            auto dv = OCIO::DisplayViewTransform::Create();
            dv->setSrc(r.working.c_str());
            dv->setDisplay(r.selection.display.c_str());
            dv->setView(r.selection.view.c_str());
            group->appendTransform(dv);
            return r.config->getProcessor(group)->getDefaultCPUProcessor();
        }

        std::string lut_key(const Resolved& r, const int size, const float lo, const float hi) {
            return std::format("{}|{}|{}|{}|{}|{}|{}", r.selection.config_path.empty() ? r.config->getName() : r.selection.config_path,
                               r.selection.display, r.selection.view, r.selection.look, size, lo, hi);
        }

        std::string canonical_or_tag(const OCIO::ConstConfigRcPtr& config, const std::string_view name) {
            if (name == kWorkingSpaceTag)
                return working_space_locked(config);
            const std::string n(name);
            if (config->getColorSpace(n.c_str()))
                return config->getCanonicalName(n.c_str());
            return {};
        }

    } // namespace

    Selection current() {
        std::lock_guard lock(g_mutex);
        return g_current;
    }

    void set_current(const Selection& s) {
        std::lock_guard lock(g_mutex);
        if (!s.config_path.empty() && s.config_path != g_current.config_path) {
            g_current.config_path = s.config_path;
            // A new config invalidates the display/view/look chosen for the old one.
            g_current.display.clear();
            g_current.view.clear();
            g_current.look.clear();
        }
        if (!s.display.empty() && s.display != g_current.display) {
            g_current.display = s.display;
            g_current.view.clear();
        }
        if (!s.view.empty())
            g_current.view = s.view;
        if (!s.look.empty())
            g_current.look = s.look == "none" ? std::string{} : s.look;
    }

    bool available(const Selection& selection, std::string* error) {
        std::lock_guard lock(g_mutex);
        try {
            (void)resolve_locked(selection);
            return true;
        } catch (const std::exception& e) {
            if (error)
                *error = e.what();
            return false;
        }
    }

    std::string config_name(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        try {
            return config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path)->getName();
        } catch (const std::exception&) {
            return {};
        }
    }

    std::string config_description(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        try {
            return config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path)->getDescription();
        } catch (const std::exception&) {
            return {};
        }
    }

    std::vector<std::string> displays(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        std::vector<std::string> out;
        try {
            auto config = config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path);
            for (int i = 0; i < config->getNumDisplays(); ++i)
                out.emplace_back(config->getDisplay(i));
        } catch (const std::exception& e) {
            LOG_WARN("OCIO: {}", e.what());
        }
        return out;
    }

    std::vector<std::string> views(const Selection& selection, const std::string_view display) {
        std::lock_guard lock(g_mutex);
        std::vector<std::string> out;
        try {
            auto config = config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path);
            const std::string d = display.empty() ? std::string(config->getDefaultDisplay()) : std::string(display);
            for (int i = 0; i < config->getNumViews(d.c_str()); ++i)
                out.emplace_back(config->getView(d.c_str(), i));
        } catch (const std::exception& e) {
            LOG_WARN("OCIO: {}", e.what());
        }
        return out;
    }

    std::vector<std::string> looks(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        std::vector<std::string> out;
        try {
            auto config = config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path);
            for (int i = 0; i < config->getNumLooks(); ++i)
                out.emplace_back(config->getLookNameByIndex(i));
        } catch (const std::exception& e) {
            LOG_WARN("OCIO: {}", e.what());
        }
        return out;
    }

    std::string default_display(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        try {
            return config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path)->getDefaultDisplay();
        } catch (const std::exception&) {
            return {};
        }
    }

    std::string default_view(const Selection& selection, const std::string_view display) {
        std::lock_guard lock(g_mutex);
        try {
            auto config = config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path);
            const std::string d = display.empty() ? std::string(config->getDefaultDisplay()) : std::string(display);
            return config->getDefaultView(d.c_str());
        } catch (const std::exception&) {
            return {};
        }
    }

    Selection resolve(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        try {
            return resolve_locked(selection).selection;
        } catch (const std::exception&) {
            return selection;
        }
    }

    std::optional<Selection> validate(const Selection& selection, std::string* error) {
        std::lock_guard lock(g_mutex);
        try {
            return resolve_locked(selection).selection;
        } catch (const std::exception& e) {
            if (error)
                *error = e.what();
            return std::nullopt;
        }
    }

    std::string working_space(const Selection& selection) {
        std::lock_guard lock(g_mutex);
        try {
            return working_space_locked(config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path));
        } catch (const std::exception&) {
            return {};
        }
    }

    std::shared_ptr<const Lut3D> lut_for(const Selection& selection, const int size_in, std::string* error) {
        std::lock_guard lock(g_mutex);
        const int size = std::clamp(size_in, 2, 129);
        try {
            const Resolved r = resolve_locked(selection);
            const float lo = kDefaultLog2Lo, hi = kDefaultLog2Hi;
            const std::string key = lut_key(r, size, lo, hi);
            if (auto it = g_luts.find(key); it != g_luts.end())
                return it->second;
            auto cpu = display_processor_locked(r);
            auto lut = std::make_shared<Lut3D>();
            lut->size = size;
            lut->log2_lo = lo;
            lut->log2_hi = hi;
            lut->key = key;
            lut->config = r.config->getName();
            const std::size_t n = static_cast<std::size_t>(size) * size * size;
            lut->rgb.resize(n * 3);
            std::vector<float> axis(static_cast<std::size_t>(size));
            for (int i = 0; i < size; ++i)
                axis[i] = i == 0 ? 0.0f : std::exp2(lo + (hi - lo) * static_cast<float>(i) / static_cast<float>(size - 1));
            for (int ir = 0; ir < size; ++ir)
                for (int ig = 0; ig < size; ++ig)
                    for (int ib = 0; ib < size; ++ib) {
                        float* p = lut->rgb.data() + 3 * ((static_cast<std::size_t>(ir) * size + ig) * size + ib);
                        p[0] = axis[ir];
                        p[1] = axis[ig];
                        p[2] = axis[ib];
                    }
            OCIO::PackedImageDesc desc(lut->rgb.data(), static_cast<long>(n), 1, 3);
            cpu->apply(desc);
            for (float& v : lut->rgb)
                v = std::isfinite(v) ? std::clamp(v, 0.0f, 1.0f) : 0.0f;
            LOG_INFO("OCIO: baked {}^3 LUT for '{}' / '{}'{} of config '{}' (log2 shaper {}..{} stops)", size,
                     r.selection.display, r.selection.view,
                     r.selection.look.empty() ? "" : std::format(" + look '{}'", r.selection.look), lut->config, lo, hi);
            std::shared_ptr<const Lut3D> out = lut;
            g_luts[key] = out;
            return out;
        } catch (const std::exception& e) {
            if (error)
                *error = e.what();
            LOG_WARN("OCIO: cannot bake the display LUT: {}", e.what());
            return nullptr;
        }
    }

    bool apply_cpu(const Selection& selection, float* rgb, const std::size_t n, std::string* error) {
        std::lock_guard lock(g_mutex);
        try {
            const Resolved r = resolve_locked(selection);
            auto cpu = display_processor_locked(r);
            if (n == 0)
                return true;
            OCIO::PackedImageDesc desc(rgb, static_cast<long>(n), 1, 3);
            cpu->apply(desc);
            return true;
        } catch (const std::exception& e) {
            if (error)
                *error = e.what();
            return false;
        }
    }

    std::optional<glm::mat3> linear_matrix(const std::string_view src, const std::string_view dst,
                                           const Selection& selection, std::string* error) {
        std::lock_guard lock(g_mutex);
        try {
            auto config = config_for_locked(selection.config_path.empty() ? g_current.config_path : selection.config_path);
            const std::string s = canonical_or_tag(config, src), d = canonical_or_tag(config, dst);
            if (s.empty() || d.empty()) {
                if (error)
                    *error = std::format("OCIO: colorspace '{}' is not in config '{}'", s.empty() ? src : dst, config->getName());
                return std::nullopt;
            }
            if (s == d)
                return glm::mat3(1.0f);
            auto cpu = config->getProcessor(s.c_str(), d.c_str())->getDefaultCPUProcessor();
            // Probe: 0, e1, e2, e3, 2e1, 2e2, 2e3, and a mixed vector (linearity = additivity + homogeneity).
            float probe[8 * 3] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0.3f, 0.5f, 0.7f};
            OCIO::PackedImageDesc desc(probe, 8, 1, 3);
            cpu->apply(desc);
            const auto at = [&](const int i) { return glm::vec3(probe[3 * i], probe[3 * i + 1], probe[3 * i + 2]); };
            const glm::vec3 zero = at(0), e1 = at(1), e2 = at(2), e3 = at(3);
            const glm::mat3 m(e1, e2, e3); // columns = f(e_i)
            const auto close = [](const glm::vec3& a, const glm::vec3& b) {
                const float scale = std::max(1.0f, std::max(glm::length(a), glm::length(b)));
                return glm::length(a - b) <= 1e-4f * scale;
            };
            const bool linear = close(zero, glm::vec3(0.0f)) && close(at(4), 2.0f * e1) && close(at(5), 2.0f * e2) &&
                                close(at(6), 2.0f * e3) && close(at(7), m * glm::vec3(0.3f, 0.5f, 0.7f));
            if (!linear) {
                if (error)
                    *error = std::format("OCIO: '{}' -> '{}' is not a linear (matrix) transform in config '{}'", s, d,
                                         config->getName());
                return std::nullopt;
            }
            return m;
        } catch (const std::exception& e) {
            if (error)
                *error = e.what();
            return std::nullopt;
        }
    }

    glm::vec3 sample(const Lut3D& lut, const glm::vec3& exposed_linear) {
        if (lut.rgb.empty() || lut.size < 2)
            return glm::clamp(exposed_linear, 0.0f, 1.0f);
        return lut_twin::lfsOcioLut(lut.rgb.data(), exposed_linear, lut.size, lut.log2_lo, lut.log2_hi);
    }

} // namespace lfs::core::ocio
