#pragma once
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

namespace lfs::rendering {
#define LFS_COLOR_INLINE inline
#define LFS_COLOR_VEC3   glm::vec3
#define LFS_COLOR_UINT   std::uint32_t
#define LFS_COLOR_MIX    glm::mix
// OCIO LUT sampler (docs/hdr.md): the C++ twin reads a flat float array.
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
#include "../shader/src/slang/display_color.inc"
#undef LFS_COLOR_INLINE
#undef LFS_COLOR_VEC3
#undef LFS_COLOR_UINT
#undef LFS_COLOR_MIX
#undef LFS_OCIO_LUT_PARAMS
#undef LFS_OCIO_LUT_FETCH
} // namespace lfs::rendering
