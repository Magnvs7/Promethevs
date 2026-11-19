// test_hv41c_bitwise.cpp -- Standalone bitwise verification & benchmark harness
// Links directly against PROMETHEVS_0.2/source/frozen/obj/deepseek41.cpp.obj
// to verify 0 uint32_t bitwise mismatches between hv41f::linear2 (CPU ref) and
// hv41cuda.dll (AMD RX 6700 XT gfx1031 HIP core).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace hv41f {
struct View {
    int N = 0;
    int K = 0;
    std::vector<uint8_t> codes;
    std::vector<int8_t> scales;
};
bool build(const uint16_t* bf16, int N, int K, View& view);
void linear2(const float* x, int rows, const View& view, float* out);
} // namespace hv41f

namespace hv41 {
void act_quant32(const float* x, int K, float* q, float* s);
} // namespace hv41

extern "C" int hilo_v41_vram_report(char* buf, int len);

// MSVC 14.51 STL compatibility helpers required when linking VS 2026 objects/libs under VS 2022 msvcprt
extern "C" {

void __cdecl __std_rotate(void* first, void* mid, void* last) noexcept {
    uint8_t* f = static_cast<uint8_t*>(first);
    uint8_t* m = static_cast<uint8_t*>(mid);
    uint8_t* l = static_cast<uint8_t*>(last);
    auto rev = [](uint8_t* a, uint8_t* b) {
        while (a < b) {
            uint8_t tmp = *a;
            *a++ = *--b;
            *b = tmp;
        }
    };
    rev(f, m);
    rev(m, l);
    rev(f, l);
}

void* __cdecl __std_unique_4(void* first, void* last) noexcept {
    uint32_t* f = static_cast<uint32_t*>(first);
    uint32_t* l = static_cast<uint32_t*>(last);
    if (f == l) return l;
    uint32_t* dest = f;
    while (++f != l) {
        if (*dest != *f && ++dest != f) *dest = *f;
    }
    return ++dest;
}

void* __cdecl __std_unique_8(void* first, void* last) noexcept {
    uint64_t* f = static_cast<uint64_t*>(first);
    uint64_t* l = static_cast<uint64_t*>(last);
    if (f == l) return l;
    uint64_t* dest = f;
    while (++f != l) {
        if (*dest != *f && ++dest != f) *dest = *f;
    }
    return ++dest;
}

const void* __cdecl __std_find_not_ch_1(const void* first, const void* last, uint8_t val) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(first);
    const uint8_t* e = static_cast<const uint8_t*>(last);
    for (; p != e; ++p) {
        if (*p != val) return p;
    }
    return e;
}

size_t __cdecl __std_regex_transform_primary_char(
    char* first1, char* last1, const char* first2, const char* last2, const void* ploc) noexcept {
    (void)first1; (void)last1; (void)first2; (void)last2; (void)ploc;
    return ~size_t{ 0 };
}

size_t __cdecl __std_regex_transform_primary_wchar_t(
    wchar_t* first1, wchar_t* last1, const wchar_t* first2, const wchar_t* last2, const void* ploc) noexcept {
    (void)first1; (void)last1; (void)first2; (void)last2; (void)ploc;
    return ~size_t{ 0 };
}

} // extern "C"

using fn_init_t      = int (*)(char*, int);
using fn_upload_t    = void* (*)(const uint8_t*, const uint8_t*, int, int);
using fn_linear_t    = int (*)(void*, const int32_t*, const float*, int, float*, float*);
using fn_resident_t  = size_t (*)();
using fn_vram_free_t = size_t (*)();
using fn_error_t     = const char* (*)();

static uint16_t float_to_bf16_rne(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static uint32_t splitmix32(uint32_t& state) {
    uint32_t z = (state += 0x9E3779B9u);
    z = (z ^ (z >> 16)) * 0x85EBCA6Bu;
    z = (z ^ (z >> 13)) * 0xC2B2AE35u;
    return z ^ (z >> 16);
}

static float rand_sym(uint32_t& state, float scale) {
    uint32_t r = splitmix32(state);
    float u = (float)(int32_t)(r >> 8) / 8388608.0f; // [-1, 1)
    return u * scale;
}

static uint32_t f32_bits(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return u;
}

struct ShapeCase {
    const char* name;
    int N;
    int K;
};

extern "C" __declspec(dllexport) int run_bitwise_suite(const char* gpu_dll_path) {
    char dll_path[MAX_PATH] = {};
    if (gpu_dll_path && *gpu_dll_path) {
        snprintf(dll_path, sizeof(dll_path), "%s", gpu_dll_path);
    } else {
        GetFullPathNameA("hv41cuda.dll", MAX_PATH, dll_path, nullptr);
    }

    printf("Testing GPU DLL: %s\n", dll_path);
    _putenv_s("HILO_V41_GPU_DLL", dll_path);

    // 1. Verify deepseek41.cpp.obj's own internal loader accepts our hv41cuda.dll
    char engine_vram_buf[512] = {};
    hilo_v41_vram_report(engine_vram_buf, (int)sizeof(engine_vram_buf));
    printf("Engine hilo_v41_vram_report(): %s\n", engine_vram_buf);
    if (std::strstr(engine_vram_buf, "-1 MiB")) {
        fprintf(stderr, "FAIL: deepseek41.cpp.obj failed to load/init hv41cuda.dll via gpu()\n");
        return 1;
    }

    // 2. Also resolve the 6 exports directly for kernel-level bitwise & speed testing
    HMODULE mod = LoadLibraryExA(dll_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!mod) {
        fprintf(stderr, "FAIL: LoadLibraryExA(%s) error %lu\n", dll_path, GetLastError());
        return 1;
    }

    auto p_init      = (fn_init_t)GetProcAddress(mod, "hv41c_init");
    auto p_upload    = (fn_upload_t)GetProcAddress(mod, "hv41c_upload");
    auto p_linear    = (fn_linear_t)GetProcAddress(mod, "hv41c_linear");
    auto p_resident  = (fn_resident_t)GetProcAddress(mod, "hv41c_resident");
    auto p_vram_free = (fn_vram_free_t)GetProcAddress(mod, "hv41c_vram_free");
    auto p_error     = (fn_error_t)GetProcAddress(mod, "hv41c_error");

    if (!p_init || !p_upload || !p_linear || !p_resident || !p_vram_free || !p_error) {
        fprintf(stderr, "FAIL: missing one or more required hv41c_* exports\n");
        return 1;
    }

    char info[512] = {};
    if (p_init(info, (int)sizeof(info)) != 0) {
        fprintf(stderr, "FAIL: hv41c_init returned error: %s\n", p_error());
        return 1;
    }
    printf("hv41c_init: %s\n", info);
    printf("Initial hv41c_vram_free: %.2f MiB, hv41c_resident: %zu bytes\n\n",
        (double)p_vram_free() / (1024.0 * 1024.0), p_resident());

    // DeepSeek-V4.1-Flash non-expert shapes + edge test shapes
    const ShapeCase shapes[] = {
        { "tiny_edge",           128,   256 },
        { "proj_512x5120",       512,  5120 },
        { "shared_gate_up",     1280,  5120 },
        { "attn_q_wkv_a",       2304,  5120 },
        { "attn_output",        5120,  2304 },
        { "shared_down",        5120,  1280 },
        { "attn_q_b",           8192,  5120 },
        { "lm_head_chunk",     32768,  1280 },
    };
    const int batch_rows[] = { 1, 2, 5, 8, 16, 64 };

    uint64_t total_values_checked = 0;
    uint64_t total_mismatches = 0;

    printf("%-16s %6s %6s %4s | %10s %10s | %9s %9s | %s\n",
        "Shape", "N", "K", "Rows", "CPU ms", "GPU ms", "GPU GB/s", "Mismatches", "Status");
    printf("------------------------------------------------------------------------------------------------\n");

    for (const auto& sc : shapes) {
        const int N = sc.N;
        const int K = sc.K;
        const int nkb = K / 32;

        // Deterministic FP8-E4M3-representable BF16 weights per 32x32 tile (required by hv41f::build lossless check)
        // Pinning element (0,0) of each non-zero 32x32 tile to +/-448.0*2^e_tile lets us exercise ALL 254 finite E4M3 codes!
        std::vector<uint16_t> w_bf16((size_t)N * (size_t)K);
        uint32_t rng = 0x12345678u ^ (uint32_t)(N * 31 + K);
        auto decode_e4m3 = [](uint8_t c) -> float {
            int sign = (c >> 7) & 1;
            int exp  = (c >> 3) & 0xF;
            int mant = c & 7;
            int mag  = (exp == 0) ? mant : ((8 + mant) << (exp - 1));
            float v  = (float)mag * (1.0f / 512.0f);
            return sign ? -v : v;
        };
        const int nnb = N / 32;
        for (int nb = 0; nb < nnb; ++nb) {
            for (int kb = 0; kb < nkb; ++kb) {
                bool zero_tile = ((nb % 11 == 3) && (kb % 7 == 2));
                // Varied E8M0 exponents in [-18, 2]
                int e_tile = -18 + (int)((nb * 7 + kb * 13) % 21);
                for (int r_t = 0; r_t < 32; ++r_t) {
                    int n = nb * 32 + r_t;
                    for (int j = 0; j < 32; ++j) {
                        uint8_t code = 0;
                        if (!zero_tile) {
                            if (r_t == 0 && j == 0) {
                                code = ((nb + kb) & 1) ? 0xFEu : 0x7Eu; // +/-448.0 pins e_w = e_tile
                            } else {
                                uint32_t r = splitmix32(rng) & 0xFFu;
                                // Exclude E4M3 NaN encodings 0x7F and 0xFF, map 0x80 (-0.0) -> 0x00
                                if ((r & 0x7Fu) == 0x7Fu) r ^= 0x01u;
                                if (r == 0x80u) r = 0x00u;
                                code = (uint8_t)r;
                            }
                        }
                        float val = zero_tile ? 0.0f : std::ldexp(decode_e4m3(code), e_tile);
                        w_bf16[(size_t)n * K + kb * 32 + j] = float_to_bf16_rne(val);
                    }
                }
            }
        }

        hv41f::View view;
        if (!hv41f::build(w_bf16.data(), N, K, view)) {
            fprintf(stderr, "FAIL: hv41f::build failed for %dx%d\n", N, K);
            return 1;
        }

        void* gpu_h = p_upload(view.codes.data(), (const uint8_t*)view.scales.data(), N, K);
        if (!gpu_h) {
            fprintf(stderr, "FAIL: hv41c_upload failed for %dx%d: %s\n", N, K, p_error());
            return 1;
        }

        for (int rows : batch_rows) {
            std::vector<float> x((size_t)rows * (size_t)K);
            for (int r = 0; r < rows; ++r) {
                for (int kb = 0; kb < nkb; ++kb) {
                    float xscale = (r % 3 == 0 && kb % 9 == 0) ? 1e-3f : ((kb % 13 == 0) ? 8.0f : 1.5f);
                    for (int j = 0; j < 32; ++j) {
                        x[(size_t)r * K + kb * 32 + j] = rand_sym(rng, xscale);
                    }
                }
            }

            std::vector<float> cpu_out((size_t)rows * (size_t)N, 0.0f);
            std::vector<float> gpu_out((size_t)rows * (size_t)N, 0.0f);

            auto t0 = std::chrono::steady_clock::now();
            hv41f::linear2(x.data(), rows, view, cpu_out.data());
            auto t1 = std::chrono::steady_clock::now();
            double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Exact same activation quantization as deepseek41.cpp::gpu_linear
            std::vector<float> q((size_t)rows * (size_t)K);
            std::vector<float> s((size_t)rows * (size_t)nkb);
            std::vector<int32_t> a((size_t)rows * (size_t)K);
            for (int r = 0; r < rows; ++r) {
                hv41::act_quant32(x.data() + (size_t)r * K, K,
                                  q.data() + (size_t)r * K,
                                  s.data() + (size_t)r * nkb);
            }
            for (size_t i = 0; i < a.size(); ++i) {
                a[i] = (int32_t)(q[i] * 512.0f);
            }

            // Warmup call + timed benchmark call (keep output checked!)
            float warmup_ms = 0.0f;
            int rc = p_linear(gpu_h, a.data(), s.data(), rows, gpu_out.data(), &warmup_ms);
            if (rc != 0) {
                fprintf(stderr, "FAIL: hv41c_linear warmup returned %d (%s)\n", rc, p_error());
                return 1;
            }
            float gpu_ms = 0.0f;
            rc = p_linear(gpu_h, a.data(), s.data(), rows, gpu_out.data(), &gpu_ms);
            if (rc != 0) {
                fprintf(stderr, "FAIL: hv41c_linear returned %d (%s)\n", rc, p_error());
                return 1;
            }

            // Compare every element bitwise as uint32_t
            uint64_t mismatches = 0;
            size_t total = (size_t)rows * (size_t)N;
            for (size_t i = 0; i < total; ++i) {
                uint32_t cb = f32_bits(cpu_out[i]);
                uint32_t gb = f32_bits(gpu_out[i]);
                if (cb != gb) {
                    if (mismatches < 5) {
                        fprintf(stderr, "  MISMATCH [%s rows=%d idx=%zu (r=%zu,n=%zu)]: cpu=0x%08X (%g) gpu=0x%08X (%g)\n",
                            sc.name, rows, i, i / (size_t)N, i % (size_t)N, cb, cpu_out[i], gb, gpu_out[i]);
                    }
                    ++mismatches;
                }
            }
            total_values_checked += total;
            total_mismatches += mismatches;

            // Effective weight bytes + activation bytes per call for memory-bandwidth metric
            double bytes = (double)N * (double)K + (double)(N / 32) * (double)nkb + (double)rows * (double)K * 4.0 + (double)rows * (double)N * 4.0;
            double gbps = (gpu_ms > 0.0f) ? (bytes / (gpu_ms * 1e6)) : 0.0;

            printf("%-16s %6d %6d %4d | %10.3f %10.3f | %9.2f %9llu | %s\n",
                sc.name, N, K, rows, cpu_ms, (double)gpu_ms, gbps,
                (unsigned long long)mismatches, mismatches == 0 ? "PASS" : "FAIL");
        }
    }

    printf("------------------------------------------------------------------------------------------------\n");
    printf("Final hv41c_resident: %.2f MiB, hv41c_vram_free: %.2f MiB\n",
        (double)p_resident() / (1024.0 * 1024.0),
        (double)p_vram_free() / (1024.0 * 1024.0));
    printf("Total values checked: %llu | Total bitwise mismatches: %llu\n",
        (unsigned long long)total_values_checked, (unsigned long long)total_mismatches);
    fflush(stdout);

    return (total_mismatches == 0) ? 0 : 2;
}

int main(int argc, char** argv) {
    return run_bitwise_suite(argc > 1 ? argv[1] : nullptr);
}
