/* Tone curves adapted from PlayCanvas.
MIT License

Copyright (c) 2011-2026 PlayCanvas Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Tone operators match the pinned PlayCanvas 2.22.0 curves. Inputs are
// display RGB for Studio composition; standard splats decode/encode below.
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsFilmicCurve(LFS_COLOR_VEC3 x) {
    return ((x*(0.15f*x+0.05f)+0.004f)/(x*(0.15f*x+0.50f)+0.06f))-0.02f/0.30f;
}
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsDisplayTone(LFS_COLOR_VEC3 color, LFS_COLOR_UINT mode, float exposure) {
    if (mode == 0u) return color;
    color *= exposure;
    if (mode == 1u) return color;
    if (mode == 2u) return lfsFilmicCurve(color) / lfsFilmicCurve(LFS_COLOR_VEC3(11.2f));
    if (mode == 3u) {
        LFS_COLOR_VEC3 h = max(LFS_COLOR_VEC3(0.0f), color - LFS_COLOR_VEC3(0.004f));
        return (h*(0.275f*h+LFS_COLOR_VEC3(0.0375f))+LFS_COLOR_VEC3(0.0025f)) /
               (h*(0.22f*h+LFS_COLOR_VEC3(0.3f))+LFS_COLOR_VEC3(0.06f)) - LFS_COLOR_VEC3(1.25f*0.01f/0.3f);
    }
    if (mode == 4u) return (color*(2.51f*color+0.03f))/(color*(2.43f*color+0.59f)+0.14f);
    if (mode == 5u) {
        color /= 0.6f;
        color = LFS_COLOR_VEC3(dot(color,LFS_COLOR_VEC3(0.59719f,0.35458f,0.04823f)),
                              dot(color,LFS_COLOR_VEC3(0.07600f,0.90834f,0.01566f)),
                              dot(color,LFS_COLOR_VEC3(0.02840f,0.13383f,0.83777f)));
        color = (color*(color+0.0245786f)-0.000090537f)/(color*(0.983729f*color+0.4329510f)+0.238081f);
        color = LFS_COLOR_VEC3(dot(color,LFS_COLOR_VEC3(1.60475f,-0.53108f,-0.07367f)),
                              dot(color,LFS_COLOR_VEC3(-0.10208f,1.10813f,-0.00605f)),
                              dot(color,LFS_COLOR_VEC3(-0.00327f,-0.07276f,1.07602f)));
        return clamp(color,LFS_COLOR_VEC3(0.0f),LFS_COLOR_VEC3(1.0f));
    }
    if (mode == 6u) {
        float x = min(color.x,min(color.y,color.z));
        float offset = x < 0.08f ? x-6.25f*x*x : 0.04f;
        color -= offset;
        float peak = max(color.x,max(color.y,color.z));
        if (peak < 0.76f) return color;
        float newPeak = 1.0f-0.24f*0.24f/(peak+0.24f-0.76f);
        color *= newPeak/peak;
        float g = 1.0f-1.0f/(0.15f*(peak-newPeak)+1.0f);
        return LFS_COLOR_MIX(color,LFS_COLOR_VEC3(newPeak),g);
    }
    return color;
}
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsDisplaySplat(LFS_COLOR_VEC3 color, LFS_COLOR_UINT mode, float exposure) {
    if (mode == 0u) return color;
    LFS_COLOR_VEC3 linear = pow(max(color,LFS_COLOR_VEC3(0.0f)),LFS_COLOR_VEC3(2.2f));
    return pow(lfsDisplayTone(linear,mode,exposure)+LFS_COLOR_VEC3(0.0000001f),LFS_COLOR_VEC3(1.0f/2.2f));
}

// ---------------------------------------------------------------------------------------------------
// HDR fork (PLAN.md M4). An HDR model's blended colour is LINEAR radiance normalised by the scene's
// P99 luminance s (white = 1 at EV 0), never display-referred, so its chain is
//     radiance * exposure (= 2^EV)  ->  tone operator  ->  sRGB OETF  ->  RGBA8,
// with lfsDisplayTone's curves (mode 0 "none" and 1 "linear" are both a plain clamp here: exposure
// always applies). Nothing in this file clamps before the tone operator. The same text compiles as
// Slang, GLSL (the compose pass) and C++ (display_color.h), which is what the CPU test pins.
// ---------------------------------------------------------------------------------------------------
LFS_COLOR_INLINE float lfsSrgbOetfChannel(float c) {
    c = clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
}
LFS_COLOR_INLINE float lfsSrgbEotfChannel(float c) {
    c = clamp(c, 0.0f, 1.0f);
    return c <= 0.04045f ? c / 12.92f : pow((c + 0.055f) / 1.055f, 2.4f);
}
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsSrgbOetf(LFS_COLOR_VEC3 c) {
    return LFS_COLOR_VEC3(lfsSrgbOetfChannel(c.x), lfsSrgbOetfChannel(c.y), lfsSrgbOetfChannel(c.z));
}
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsSrgbEotf(LFS_COLOR_VEC3 c) {
    return LFS_COLOR_VEC3(lfsSrgbEotfChannel(c.x), lfsSrgbEotfChannel(c.y), lfsSrgbEotfChannel(c.z));
}
// The names the baked LDR export (M5, core/hdr_sh.cu) uses: the SAME OETF, one definition, so the
// exporter and the viewer cannot drift.
LFS_COLOR_INLINE float lfsSrgbEncodeChannel(float x) { return lfsSrgbOetfChannel(x); }
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsSrgbEncode(LFS_COLOR_VEC3 c) { return lfsSrgbOetf(c); }
// Linear normalised radiance -> display RGB in [0, 1] (sRGB-encoded). mode: the tone IDs above.
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsDisplayHdr(LFS_COLOR_VEC3 radiance, LFS_COLOR_UINT mode, float exposure) {
    LFS_COLOR_VEC3 toned = lfsDisplayTone(max(radiance, LFS_COLOR_VEC3(0.0f)), mode == 0u ? 1u : mode, exposure);
    return lfsSrgbOetf(toned);
}
// Luminance in stops relative to the scene's white (radiance / s = 1 -> 0 EV) on a viridis-like ramp:
// -8 EV dark violet ... +8 EV yellow; at or above the model's cap -> magenta; no light -> black.
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsHdrFalseColor(float luminance, float ln_cap_norm) {
    if (!(luminance > 0.0f)) return LFS_COLOR_VEC3(0.0f);
    if (log(luminance) >= ln_cap_norm - 1.0e-3f) return LFS_COLOR_VEC3(1.0f, 0.0f, 1.0f);
    float t = clamp((log2(luminance) + 8.0f) / 16.0f, 0.0f, 1.0f);
    LFS_COLOR_VEC3 c0 = LFS_COLOR_VEC3(0.267f, 0.005f, 0.329f);
    LFS_COLOR_VEC3 c1 = LFS_COLOR_VEC3(0.282f, 0.141f, 0.458f);
    LFS_COLOR_VEC3 c2 = LFS_COLOR_VEC3(0.254f, 0.265f, 0.530f);
    LFS_COLOR_VEC3 c3 = LFS_COLOR_VEC3(0.207f, 0.372f, 0.553f);
    LFS_COLOR_VEC3 c4 = LFS_COLOR_VEC3(0.164f, 0.471f, 0.558f);
    LFS_COLOR_VEC3 c5 = LFS_COLOR_VEC3(0.128f, 0.567f, 0.551f);
    LFS_COLOR_VEC3 c6 = LFS_COLOR_VEC3(0.135f, 0.659f, 0.518f);
    LFS_COLOR_VEC3 c7 = LFS_COLOR_VEC3(0.267f, 0.749f, 0.441f);
    LFS_COLOR_VEC3 c8 = LFS_COLOR_VEC3(0.478f, 0.821f, 0.318f);
    LFS_COLOR_VEC3 c9 = LFS_COLOR_VEC3(0.741f, 0.873f, 0.150f);
    LFS_COLOR_VEC3 c10 = LFS_COLOR_VEC3(0.993f, 0.906f, 0.144f);
    float f = t * 10.0f;
    LFS_COLOR_UINT i = LFS_COLOR_UINT(min(f, 9.0f));
    float w = f - float(i);
    LFS_COLOR_VEC3 a = i == 0u ? c0 : i == 1u ? c1 : i == 2u ? c2 : i == 3u ? c3 : i == 4u ? c4
                     : i == 5u ? c5 : i == 6u ? c6 : i == 7u ? c7 : i == 8u ? c8 : c9;
    LFS_COLOR_VEC3 b = i == 0u ? c1 : i == 1u ? c2 : i == 2u ? c3 : i == 3u ? c4 : i == 4u ? c5
                     : i == 5u ? c6 : i == 6u ? c7 : i == 7u ? c8 : i == 8u ? c9 : c10;
    return LFS_COLOR_MIX(a, b, w);
}

// ---------------------------------------------------------------------------------------------------
// OpenColorIO display (docs/hdr.md "Colour management"). The (display, view, look) of an OCIO config
// is baked on the CPU into a size^3 float LUT over a per-channel log2 shaper: node i of an axis holds the
// output for normalised radiance 2^(lo + i/(size-1) * (hi-lo)), node 0 for exactly 0. Input is EXPOSED
// scene-linear light (radiance / s * 2^EV, the exposure applied before the view transform, as OCIO
// expects); output is the display encoding in [0, 1]. The same text samples the LUT in the compose
// shader (GLSL, an SSBO of floats), the CUDA bake (a device pointer) and the C++ twin, so the three
// cannot drift; core/ocio.cpp bakes it and tests/test_hdr_display_chain.cpp pins it against OCIO's own
// CPU processor. The block is compiled only where a LUT can be bound (LFS_OCIO_LUT_FETCH defined): the
// Slang projection shaders never define it and compile exactly as before.
//   LFS_OCIO_LUT_PARAMS  the leading parameter(s) that carry the LUT storage (empty in GLSL)
//   LFS_OCIO_LUT_FETCH(i) the vec3 at node index i = (ir*size + ig)*size + ib of a FLAT float array
//                        (3 floats per node; never a vec3[] — std430 would pad it to 16 bytes)
// ---------------------------------------------------------------------------------------------------
#ifdef LFS_OCIO_LUT_FETCH
LFS_COLOR_INLINE LFS_COLOR_VEC3 lfsOcioLut(LFS_OCIO_LUT_PARAMS LFS_COLOR_VEC3 x, int size, float lo, float hi) {
    const float span = max(hi - lo, 1.0e-6f);
    LFS_COLOR_VEC3 t = (log2(max(x, LFS_COLOR_VEC3(exp2(lo)))) - LFS_COLOR_VEC3(lo)) / span;
    t = clamp(t, LFS_COLOR_VEC3(0.0f), LFS_COLOR_VEC3(1.0f)) * float(size - 1);
    int ir = min(int(t.x), size - 2);
    int ig = min(int(t.y), size - 2);
    int ib = min(int(t.z), size - 2);
    float fr = t.x - float(ir);
    float fg = t.y - float(ig);
    float fb = t.z - float(ib);
    int base = (ir * size + ig) * size + ib;
    int sr = size * size;
    LFS_COLOR_VEC3 c000 = LFS_OCIO_LUT_FETCH(base);
    LFS_COLOR_VEC3 c001 = LFS_OCIO_LUT_FETCH(base + 1);
    LFS_COLOR_VEC3 c010 = LFS_OCIO_LUT_FETCH(base + size);
    LFS_COLOR_VEC3 c011 = LFS_OCIO_LUT_FETCH(base + size + 1);
    LFS_COLOR_VEC3 c100 = LFS_OCIO_LUT_FETCH(base + sr);
    LFS_COLOR_VEC3 c101 = LFS_OCIO_LUT_FETCH(base + sr + 1);
    LFS_COLOR_VEC3 c110 = LFS_OCIO_LUT_FETCH(base + sr + size);
    LFS_COLOR_VEC3 c111 = LFS_OCIO_LUT_FETCH(base + sr + size + 1);
    LFS_COLOR_VEC3 c00 = LFS_COLOR_MIX(c000, c001, fb);
    LFS_COLOR_VEC3 c01 = LFS_COLOR_MIX(c010, c011, fb);
    LFS_COLOR_VEC3 c10 = LFS_COLOR_MIX(c100, c101, fb);
    LFS_COLOR_VEC3 c11 = LFS_COLOR_MIX(c110, c111, fb);
    LFS_COLOR_VEC3 c0 = LFS_COLOR_MIX(c00, c01, fg);
    LFS_COLOR_VEC3 c1 = LFS_COLOR_MIX(c10, c11, fg);
    return clamp(LFS_COLOR_MIX(c0, c1, fr), LFS_COLOR_VEC3(0.0f), LFS_COLOR_VEC3(1.0f));
}
#endif
