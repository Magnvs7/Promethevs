// hv41cuda_hip.cpp -- AMD RDNA2 (gfx1031) drop-in replacement for hv41cuda.dll
// Implements the 6-function hv41c_* core API verified by deepseek41.cpp.obj:
//   hv41c_init, hv41c_upload, hv41c_linear, hv41c_resident, hv41c_vram_free, hv41c_error
// Uses Windows inbox WHQL-signed amdhip64_6.dll + amd_comgr_2.dll (zero ROCm SDK install required).

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// AMD Comgr 2.9 minimal C ABI (dynamically loaded from amd_comgr_2.dll)
// ---------------------------------------------------------------------------
using amd_comgr_status_t = int;
using amd_comgr_language_t = int;
using amd_comgr_data_kind_t = int;
using amd_comgr_action_kind_t = int;

struct amd_comgr_data_s { uint64_t handle; };
struct amd_comgr_data_set_s { uint64_t handle; };
struct amd_comgr_action_info_s { uint64_t handle; };

constexpr amd_comgr_status_t AMD_COMGR_STATUS_SUCCESS = 0;
constexpr amd_comgr_language_t AMD_COMGR_LANGUAGE_HIP = 3;

constexpr amd_comgr_data_kind_t AMD_COMGR_DATA_KIND_SOURCE = 1;
constexpr amd_comgr_data_kind_t AMD_COMGR_DATA_KIND_DIAGNOSTIC = 4;
constexpr amd_comgr_data_kind_t AMD_COMGR_DATA_KIND_LOG = 5;
constexpr amd_comgr_data_kind_t AMD_COMGR_DATA_KIND_EXECUTABLE = 8;

constexpr amd_comgr_action_kind_t AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC = 2;
constexpr amd_comgr_action_kind_t AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE = 4;
constexpr amd_comgr_action_kind_t AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE = 7;

struct ComgrApi {
    HMODULE mod = nullptr;
    amd_comgr_status_t (*status_string)(amd_comgr_status_t, const char**) = nullptr;
    void (*get_version)(size_t*, size_t*) = nullptr;
    amd_comgr_status_t (*create_data)(amd_comgr_data_kind_t, amd_comgr_data_s*) = nullptr;
    amd_comgr_status_t (*release_data)(amd_comgr_data_s) = nullptr;
    amd_comgr_status_t (*set_data)(amd_comgr_data_s, size_t, const char*) = nullptr;
    amd_comgr_status_t (*set_data_name)(amd_comgr_data_s, const char*) = nullptr;
    amd_comgr_status_t (*get_data)(amd_comgr_data_s, size_t*, char*) = nullptr;
    amd_comgr_status_t (*create_data_set)(amd_comgr_data_set_s*) = nullptr;
    amd_comgr_status_t (*destroy_data_set)(amd_comgr_data_set_s) = nullptr;
    amd_comgr_status_t (*data_set_add)(amd_comgr_data_set_s, amd_comgr_data_s) = nullptr;
    amd_comgr_status_t (*action_data_count)(amd_comgr_data_set_s, amd_comgr_data_kind_t, size_t*) = nullptr;
    amd_comgr_status_t (*action_data_get_data)(amd_comgr_data_set_s, amd_comgr_data_kind_t, size_t, amd_comgr_data_s*) = nullptr;
    amd_comgr_status_t (*create_action_info)(amd_comgr_action_info_s*) = nullptr;
    amd_comgr_status_t (*destroy_action_info)(amd_comgr_action_info_s) = nullptr;
    amd_comgr_status_t (*action_info_set_isa_name)(amd_comgr_action_info_s, const char*) = nullptr;
    amd_comgr_status_t (*action_info_set_language)(amd_comgr_action_info_s, amd_comgr_language_t) = nullptr;
    amd_comgr_status_t (*action_info_set_logging)(amd_comgr_action_info_s, bool) = nullptr;
    amd_comgr_status_t (*action_info_set_option_list)(amd_comgr_action_info_s, const char*[], size_t) = nullptr;
    amd_comgr_status_t (*do_action)(amd_comgr_action_kind_t, amd_comgr_action_info_s, amd_comgr_data_set_s, amd_comgr_data_set_s) = nullptr;

    bool load() {
        if (mod) return true;
        mod = LoadLibraryA("amd_comgr_2.dll");
        if (!mod) return false;
#define LOAD_C(field, sym) \
    field = reinterpret_cast<decltype(field)>(GetProcAddress(mod, sym)); \
    if (!field) return false
        LOAD_C(status_string, "amd_comgr_status_string");
        LOAD_C(get_version, "amd_comgr_get_version");
        LOAD_C(create_data, "amd_comgr_create_data");
        LOAD_C(release_data, "amd_comgr_release_data");
        LOAD_C(set_data, "amd_comgr_set_data");
        LOAD_C(set_data_name, "amd_comgr_set_data_name");
        LOAD_C(get_data, "amd_comgr_get_data");
        LOAD_C(create_data_set, "amd_comgr_create_data_set");
        LOAD_C(destroy_data_set, "amd_comgr_destroy_data_set");
        LOAD_C(data_set_add, "amd_comgr_data_set_add");
        LOAD_C(action_data_count, "amd_comgr_action_data_count");
        LOAD_C(action_data_get_data, "amd_comgr_action_data_get_data");
        LOAD_C(create_action_info, "amd_comgr_create_action_info");
        LOAD_C(destroy_action_info, "amd_comgr_destroy_action_info");
        LOAD_C(action_info_set_isa_name, "amd_comgr_action_info_set_isa_name");
        LOAD_C(action_info_set_language, "amd_comgr_action_info_set_language");
        LOAD_C(action_info_set_logging, "amd_comgr_action_info_set_logging");
        LOAD_C(action_info_set_option_list, "amd_comgr_action_info_set_option_list");
        LOAD_C(do_action, "amd_comgr_do_action");
#undef LOAD_C
        return true;
    }
};

// ---------------------------------------------------------------------------
// AMD HIP 6 minimal C ABI (dynamically loaded from amdhip64_6.dll)
// ---------------------------------------------------------------------------
constexpr int HIP_SUCCESS = 0;
constexpr int HIP_MEMCPY_HOST_TO_DEVICE = 1;
constexpr int HIP_MEMCPY_DEVICE_TO_HOST = 2;
constexpr int HIP_MEMCPY_DEVICE_TO_DEVICE = 3;
constexpr int HIP_DEVICE_ATTR_MP_COUNT = 63;

struct HipApi {
    HMODULE mod = nullptr;
    const char* (*getErrorName)(int) = nullptr;
    const char* (*getErrorString)(int) = nullptr;
    int (*init)(unsigned int) = nullptr;
    int (*setDevice)(int) = nullptr;
    int (*deviceGetName)(char*, int, int) = nullptr;
    int (*deviceComputeCapability)(int*, int*, int) = nullptr;
    int (*deviceGetAttribute)(int*, int, int) = nullptr;
    int (*memGetInfo)(size_t*, size_t*) = nullptr;
    int (*moduleLoadData)(void**, const void*) = nullptr;
    int (*moduleGetFunction)(void**, void*, const char*) = nullptr;
    int (*malloc)(void**, size_t) = nullptr;
    int (*free)(void*) = nullptr;
    int (*hostMalloc)(void**, size_t, unsigned int) = nullptr;
    int (*hostFree)(void*) = nullptr;
    int (*memcpy)(void*, const void*, size_t, int) = nullptr;
    int (*memcpyAsync)(void*, const void*, size_t, int, void*) = nullptr;
    int (*streamCreate)(void**) = nullptr;
    int (*streamSynchronize)(void*) = nullptr;
    int (*streamWaitEvent)(void*, void*, unsigned int) = nullptr;
    int (*eventCreate)(void**) = nullptr;
    int (*eventRecord)(void*, void*) = nullptr;
    int (*eventSynchronize)(void*) = nullptr;
    int (*eventElapsedTime)(float*, void*, void*) = nullptr;
    int (*moduleLaunchKernel)(
        void*, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int,
        unsigned int, void*, void**, void**) = nullptr;
    int (*deviceSynchronize)(void) = nullptr;

    bool load() {
        if (mod) return true;
        mod = LoadLibraryA("amdhip64_6.dll");
        if (!mod) return false;
#define LOAD_H(field, sym) \
    field = reinterpret_cast<decltype(field)>(GetProcAddress(mod, sym)); \
    if (!field) return false
        LOAD_H(getErrorName, "hipGetErrorName");
        LOAD_H(getErrorString, "hipGetErrorString");
        LOAD_H(init, "hipInit");
        LOAD_H(setDevice, "hipSetDevice");
        LOAD_H(deviceGetName, "hipDeviceGetName");
        LOAD_H(deviceComputeCapability, "hipDeviceComputeCapability");
        LOAD_H(deviceGetAttribute, "hipDeviceGetAttribute");
        LOAD_H(memGetInfo, "hipMemGetInfo");
        LOAD_H(moduleLoadData, "hipModuleLoadData");
        LOAD_H(moduleGetFunction, "hipModuleGetFunction");
        LOAD_H(malloc, "hipMalloc");
        LOAD_H(free, "hipFree");
        LOAD_H(hostMalloc, "hipHostMalloc");
        LOAD_H(hostFree, "hipHostFree");
        LOAD_H(memcpy, "hipMemcpy");
        LOAD_H(memcpyAsync, "hipMemcpyAsync");
        LOAD_H(streamCreate, "hipStreamCreate");
        LOAD_H(streamSynchronize, "hipStreamSynchronize");
        LOAD_H(streamWaitEvent, "hipStreamWaitEvent");
        LOAD_H(eventCreate, "hipEventCreate");
        LOAD_H(eventRecord, "hipEventRecord");
        LOAD_H(eventSynchronize, "hipEventSynchronize");
        LOAD_H(eventElapsedTime, "hipEventElapsedTime");
        LOAD_H(moduleLaunchKernel, "hipModuleLaunchKernel");
        LOAD_H(deviceSynchronize, "hipDeviceSynchronize");
#undef LOAD_H
        return true;
    }
};

// ---------------------------------------------------------------------------
// Embedded gfx1031 wave32 HIP kernel matching hv41f::linear2 bit-for-bit
// Coalesced [N/32][K/32][8][32] uint32 weight layout (zero-shuffle 32-cols/wave32) + integer-RNE scaling
// ---------------------------------------------------------------------------
const char* kHipKernelSource = R"HIP(
#pragma STDC FP_CONTRACT OFF
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef struct __attribute__((aligned(16))) {
    int x, y, z, w;
} i4_t;

typedef struct __attribute__((aligned(16))) {
    unsigned int x, y, z, w;
} u4_t;

__device__ static __inline__ int decode_e4m3_x512(unsigned int b) {
    int sign = (int)((b >> 7) & 1u);
    int exp  = (int)((b >> 3) & 0xFu);
    int mant = (int)(b & 7u);
    int mag  = (exp == 0) ? mant : ((8 + mant) << (exp - 1));
    return sign ? -mag : mag;
}

// Bit-exact equivalent of (float)(((double)dot * 2^-18) * ldexp(1.0, e_w)) using integer ops
__device__ static __inline__ float dot_to_scaled_f32(long long dot, int e_w) {
    if (dot == 0ll) return 0.0f;
    unsigned int sign = (dot < 0ll) ? 0x80000000u : 0u;
    unsigned long long mag = (dot < 0ll) ? (unsigned long long)(-dot) : (unsigned long long)dot;
    int lz = __builtin_clzll(mag);
    int msb = 63 - lz;
    int f32_exp = msb + e_w + 109; // msb + (e_w - 18) + 127
    union { unsigned int u; float f; } out;
    if (f32_exp >= 255) {
        out.u = sign | 0x7F800000u;
        return out.f;
    }
    unsigned long long norm = mag << lz;
    if (f32_exp >= 1) {
        unsigned int mant23    = (unsigned int)((norm >> 40) & 0x7FFFFFull);
        unsigned int round_bit = (unsigned int)((norm >> 39) & 1ull);
        unsigned int sticky    = ((norm & 0x7FFFFFFFFFull) != 0ull) ? 1u : 0u;
        unsigned int round_up  = round_bit & (sticky | (mant23 & 1u));
        out.u = sign | (((unsigned int)f32_exp << 23) + mant23 + round_up);
        return out.f;
    }
    if (f32_exp <= -24) {
        out.u = sign;
        return out.f;
    }
    if (f32_exp == -23) {
        // Value is in [2^-150, 2^-149): exact 2^-150 ties to even (0), strictly greater rounds to 1
        out.u = sign | (((norm << 1) != 0ull) ? 1u : 0u);
        return out.f;
    }
    int shift = 41 - f32_exp; // 41..63 (strictly < 64)
    unsigned int mant       = (unsigned int)(norm >> shift);
    unsigned long long rem  = norm & ((1ull << shift) - 1ull);
    unsigned long long half = 1ull << (shift - 1);
    unsigned int round_bit  = ((rem & half) != 0ull) ? 1u : 0u;
    unsigned int sticky     = ((rem & (half - 1ull)) != 0ull) ? 1u : 0u;
    unsigned int round_up   = round_bit & (sticky | (mant & 1u));
    out.u = sign | (mant + round_up);
    return out.f;
}

__device__ static __inline__ float round_bf16_in_f32(float v) {
    union { unsigned int u; float f; } u;
    u.f = v;
    u.u = (u.u + 0x7FFFu + ((u.u >> 16) & 1u)) & 0xFFFF0000u;
    return u.f;
}

// Fast path for single-row decode (rows == 1): 1 wave32 per workgroup (blockDim.x = 32),
// each lane handles 1 output column n = nb * 32 + lane; 128-byte coalesced weight loads, 0 shuffles.
extern "C" __global__ void hv41_fp8_linear_m1(
    const unsigned int* __restrict__ packed_codes,
    const signed char*  __restrict__ scales,
    const int*          __restrict__ a,
    const float*        __restrict__ s,
    float*              __restrict__ out,
    int N,
    int K)
{
    int nb   = (int)__builtin_amdgcn_workgroup_id_x();
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int nnb  = N >> 5;
    if (nb >= nnb) return;

    int n = (nb << 5) + lane;
    int nkb = K >> 5;
    const unsigned int* w_tile = packed_codes + ((unsigned long long)nb * (unsigned long long)nkb << 8);
    const signed char*  w_sc   = scales + (unsigned long long)nb * (unsigned long long)nkb;
    const i4_t*         a_vec  = (const i4_t*)a;

    float acc = 0.0f;
    for (int kb = 0; kb < nkb; ++kb) {
        const unsigned int* blk = w_tile + ((unsigned long long)kb << 8);
        const i4_t*         ak  = a_vec + (kb << 3);
        long long dot = 0ll;

        #pragma unroll 8
        for (int grp = 0; grp < 8; ++grp) {
            unsigned int w4 = blk[(grp << 5) + lane];
            i4_t av = ak[grp];
            int w0 = decode_e4m3_x512(w4 & 0xFFu);
            int w1 = decode_e4m3_x512((w4 >> 8) & 0xFFu);
            int w2 = decode_e4m3_x512((w4 >> 16) & 0xFFu);
            int w3 = decode_e4m3_x512((w4 >> 24) & 0xFFu);
            dot += (long long)w0 * (long long)av.x
                 + (long long)w1 * (long long)av.y
                 + (long long)w2 * (long long)av.z
                 + (long long)w3 * (long long)av.w;
        }

        float bval = dot_to_scaled_f32(dot, (int)w_sc[kb]);
        float term = bval * s[kb];
        acc = term + acc;
    }
    out[n] = round_bf16_in_f32(acc);
}

// Multi-wave M=1 kernel: WAVES wave32s per 32-col workgroup share LDS activation/scale-free
// 256-entry E4M3 LUT and 32x32 float term buffer, summing exact K-blocks sequentially on wave 0.
template<int WAVES>
__device__ static __inline__ void fp8_m1_mw_lut_impl(
    const unsigned int* __restrict__ packed_codes,
    const signed char*  __restrict__ scales,
    const int*          __restrict__ a,
    const float*        __restrict__ s,
    float*              __restrict__ out,
    int N,
    int K)
{
    int nb   = (int)__builtin_amdgcn_workgroup_id_x();
    int tid  = (int)__builtin_amdgcn_workitem_id_x();
    int lane = tid & 31;
    int wy   = tid >> 5;
    int nnb  = N >> 5;
    if (nb >= nnb) return;

    __attribute__((shared)) static float s_term[32][32];
    __attribute__((shared)) static int   s_lut[256];

    for (int i = tid; i < 256; i += (WAVES * 32)) {
        s_lut[i] = decode_e4m3_x512((unsigned int)i);
    }
    __builtin_amdgcn_s_barrier();

    int n = (nb << 5) + lane;
    int nkb = K >> 5;
    const unsigned int* w_tile = packed_codes + ((unsigned long long)nb * (unsigned long long)nkb << 8);
    const signed char*  w_sc   = scales + (unsigned long long)nb * (unsigned long long)nkb;
    const i4_t*         a_vec  = (const i4_t*)a;

    float acc = 0.0f;
    for (int kb0 = 0; kb0 < nkb; kb0 += 32) {
        int rem = nkb - kb0;
        int cnt = (rem < 32) ? rem : 32;
        for (int t = wy; t < cnt; t += WAVES) {
            int kb = kb0 + t;
            const unsigned int* blk = w_tile + ((unsigned long long)kb << 8);
            const i4_t*         ak  = a_vec + (kb << 3);
            long long dot = 0ll;

            #pragma unroll 8
            for (int grp = 0; grp < 8; ++grp) {
                unsigned int w4 = blk[(grp << 5) + lane];
                i4_t av = ak[grp];
                int w0 = s_lut[w4 & 0xFFu];
                int w1 = s_lut[(w4 >> 8) & 0xFFu];
                int w2 = s_lut[(w4 >> 16) & 0xFFu];
                int w3 = s_lut[(w4 >> 24) & 0xFFu];
                dot += (long long)w0 * (long long)av.x
                     + (long long)w1 * (long long)av.y
                     + (long long)w2 * (long long)av.z
                     + (long long)w3 * (long long)av.w;
            }

            float bval = dot_to_scaled_f32(dot, (int)w_sc[kb]);
            s_term[t][lane] = bval * s[kb];
        }
        __builtin_amdgcn_s_barrier();
        if (wy == 0) {
            for (int t = 0; t < cnt; ++t) {
                acc = s_term[t][lane] + acc;
            }
        }
        __builtin_amdgcn_s_barrier();
    }
    if (wy == 0) {
        out[n] = round_bf16_in_f32(acc);
    }
}

extern "C" __global__ void hv41_fp8_linear_m1_w8(
    const unsigned int* __restrict__ packed_codes,
    const signed char*  __restrict__ scales,
    const int*          __restrict__ a,
    const float*        __restrict__ s,
    float*              __restrict__ out,
    int N,
    int K)
{
    fp8_m1_mw_lut_impl<8>(packed_codes, scales, a, s, out, N, K);
}

extern "C" __global__ void hv41_fp8_linear_m1_w16(
    const unsigned int* __restrict__ packed_codes,
    const signed char*  __restrict__ scales,
    const int*          __restrict__ a,
    const float*        __restrict__ s,
    float*              __restrict__ out,
    int N,
    int K)
{
    fp8_m1_mw_lut_impl<16>(packed_codes, scales, a, s, out, N, K);
}

// Multi-row kernel: processes up to 4 rows per workgroup.y slice (1 wave32/workgroup) with zero shuffles.
extern "C" __global__ void hv41_fp8_linear_tile4(
    const unsigned int* __restrict__ packed_codes,
    const signed char*  __restrict__ scales,
    const int*          __restrict__ a,
    const float*        __restrict__ s,
    float*              __restrict__ out,
    int N,
    int K,
    int rows)
{
    int nb   = (int)__builtin_amdgcn_workgroup_id_x();
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int nnb  = N >> 5;
    if (nb >= nnb) return;

    __attribute__((shared)) static int s_lut[256];
    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        s_lut[lane + i * 32] = decode_e4m3_x512((unsigned int)(lane + i * 32));
    }
    __builtin_amdgcn_s_barrier();

    int r0 = (int)(__builtin_amdgcn_workgroup_id_y() * 4u);
    int r_cnt = rows - r0;
    if (r_cnt <= 0) return;
    if (r_cnt > 4) r_cnt = 4;

    int o1 = (r_cnt > 1) ? 1 : 0;
    int o2 = (r_cnt > 2) ? 2 : 0;
    int o3 = (r_cnt > 3) ? 3 : 0;

    int n = (nb << 5) + lane;
    int nkb = K >> 5;
    const unsigned int* w_tile = packed_codes + ((unsigned long long)nb * (unsigned long long)nkb << 8);
    const signed char*  w_sc   = scales + (unsigned long long)nb * (unsigned long long)nkb;

    const i4_t* a0 = (const i4_t*)(a + (unsigned long long)r0 * (unsigned long long)K);
    const i4_t* a1 = (const i4_t*)(a + (unsigned long long)(r0 + o1) * (unsigned long long)K);
    const i4_t* a2 = (const i4_t*)(a + (unsigned long long)(r0 + o2) * (unsigned long long)K);
    const i4_t* a3 = (const i4_t*)(a + (unsigned long long)(r0 + o3) * (unsigned long long)K);

    const float* s0 = s + (unsigned long long)r0 * (unsigned long long)nkb;
    const float* s1 = s + (unsigned long long)(r0 + o1) * (unsigned long long)nkb;
    const float* s2 = s + (unsigned long long)(r0 + o2) * (unsigned long long)nkb;
    const float* s3 = s + (unsigned long long)(r0 + o3) * (unsigned long long)nkb;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int kb = 0; kb < nkb; ++kb) {
        const unsigned int* blk = w_tile + ((unsigned long long)kb << 8);
        int k_vec = kb << 3;
        const i4_t* ak0 = a0 + k_vec;
        const i4_t* ak1 = a1 + k_vec;
        const i4_t* ak2 = a2 + k_vec;
        const i4_t* ak3 = a3 + k_vec;

        long long dot0 = 0ll;
        long long dot1 = 0ll;
        long long dot2 = 0ll;
        long long dot3 = 0ll;

        #pragma unroll 8
        for (int grp = 0; grp < 8; ++grp) {
            unsigned int w4 = blk[(grp << 5) + lane];
            long long w0 = (long long)s_lut[w4 & 0xFFu];
            long long w1 = (long long)s_lut[(w4 >> 8) & 0xFFu];
            long long w2 = (long long)s_lut[(w4 >> 16) & 0xFFu];
            long long w3 = (long long)s_lut[(w4 >> 24) & 0xFFu];

            i4_t v0 = ak0[grp];
            dot0 += w0 * (long long)v0.x + w1 * (long long)v0.y + w2 * (long long)v0.z + w3 * (long long)v0.w;
            i4_t v1 = ak1[grp];
            dot1 += w0 * (long long)v1.x + w1 * (long long)v1.y + w2 * (long long)v1.z + w3 * (long long)v1.w;
            i4_t v2 = ak2[grp];
            dot2 += w0 * (long long)v2.x + w1 * (long long)v2.y + w2 * (long long)v2.z + w3 * (long long)v2.w;
            i4_t v3 = ak3[grp];
            dot3 += w0 * (long long)v3.x + w1 * (long long)v3.y + w2 * (long long)v3.z + w3 * (long long)v3.w;
        }

        int e_w = (int)w_sc[kb];
        acc0 = dot_to_scaled_f32(dot0, e_w) * s0[kb] + acc0;
        acc1 = dot_to_scaled_f32(dot1, e_w) * s1[kb] + acc1;
        acc2 = dot_to_scaled_f32(dot2, e_w) * s2[kb] + acc2;
        acc3 = dot_to_scaled_f32(dot3, e_w) * s3[kb] + acc3;
    }

    out[(unsigned long long)r0 * (unsigned long long)N + (unsigned long long)n] = round_bf16_in_f32(acc0);
    if (r_cnt > 1) out[(unsigned long long)(r0 + 1) * (unsigned long long)N + (unsigned long long)n] = round_bf16_in_f32(acc1);
    if (r_cnt > 2) out[(unsigned long long)(r0 + 2) * (unsigned long long)N + (unsigned long long)n] = round_bf16_in_f32(acc2);
    if (r_cnt > 3) out[(unsigned long long)(r0 + 3) * (unsigned long long)N + (unsigned long long)n] = round_bf16_in_f32(acc3);
}

__device__ static __inline__ float bf16_to_f32(unsigned short b) {
    union { unsigned int u; float f; } u;
    u.u = ((unsigned int)b) << 16;
    return u.f;
}

__device__ static __inline__ float uint_as_f32(unsigned int bits) {
    union { unsigned int u; float f; } u;
    u.u = bits;
    return u.f;
}

__device__ static __inline__ unsigned int f32_as_uint(float val) {
    union { unsigned int u; float f; } u;
    u.f = val;
    return u.u;
}

extern "C" __global__ void hv41_bf16_seq_kernel(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,
    float*                __restrict__ y,
    int M,
    int N,
    int K,
    int ng,
    int K_x,
    int bf16_round)
{
    unsigned long long idx = (unsigned long long)__builtin_amdgcn_workgroup_id_x() * 64ull
                           + (unsigned long long)__builtin_amdgcn_workitem_id_x();
    unsigned long long total = (unsigned long long)M * (unsigned long long)N;
    if (idx >= total) return;

    int m = (int)(idx / (unsigned long long)N);
    int n = (int)(idx % (unsigned long long)N);
    int g = n / ng;

    const float* x_ptr = x + (unsigned long long)m * (unsigned long long)K_x + (unsigned long long)g * (unsigned long long)K;
    const unsigned short* w_ptr = W + (unsigned long long)n * (unsigned long long)K;

    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        acc += (double)x_ptr[k] * (double)bf16_to_f32(w_ptr[k]);
    }
    float res = (float)acc;
    if (bf16_round) res = round_bf16_in_f32(res);
    y[idx] = res;
}

extern "C" __global__ void hv41_bf16_w32_kernel(
    const unsigned short* __restrict__ W,
    const float*          __restrict__ x,
    float*                __restrict__ y,
    int M,
    int N,
    int K,
    int ng,
    int K_x,
    int bf16_round)
{
    int n = (int)__builtin_amdgcn_workgroup_id_x();
    int m = (int)__builtin_amdgcn_workgroup_id_y();
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    if (n >= N || m >= M) return;

    int g = n / ng;
    const float* x_ptr = x + (unsigned long long)m * (unsigned long long)K_x + (unsigned long long)g * (unsigned long long)K;
    const unsigned short* w_ptr = W + (unsigned long long)n * (unsigned long long)K;

    int nblocks = K >> 3;
    float acc = 0.0f;
    for (int b = lane; b < nblocks; b += 32) {
        int base = b << 3;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc = __builtin_fmaf(x_ptr[base + t], bf16_to_f32(w_ptr[base + t]), acc);
        }
    }

    __attribute__((shared)) static float smem[32];
    smem[lane] = acc;
    __builtin_amdgcn_s_barrier();

    for (int stride = 16; stride >= 1; stride >>= 1) {
        if (lane < stride) {
            smem[lane] = smem[lane + stride] + smem[lane];
        }
        __builtin_amdgcn_s_barrier();
    }

    if (lane == 0) {
        float res = smem[0];
        if (bf16_round) res = round_bf16_in_f32(res);
        y[(unsigned long long)m * (unsigned long long)N + (unsigned long long)n] = res;
    }
}

__device__ static __inline__ float sattn_expf(float x, const unsigned int* __restrict__ exptable, int table_count) {
    if (exptable && table_count > 0) {
        unsigned int u = f32_as_uint(x);
        int lo = 0;
        int hi = table_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            unsigned int key = exptable[mid * 2];
            if (key == u) return uint_as_f32(exptable[mid * 2 + 1]);
            if (key < u) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    return __builtin_expf(x);
}
)HIP";

static const char* kHipKernelSource2 = R"HIP(
// 1 workgroup (64 threads) per (m, h) head; d <= 512 (8 elements/thread).
extern "C" __global__ void hv41_sattn_kernel(
    const float*        __restrict__ Q,
    const float*        __restrict__ K,
    const int*          __restrict__ idx,
    const float*        __restrict__ V,
    const float*        __restrict__ sink,
    float*              __restrict__ O,
    const unsigned int* __restrict__ exptable,
    int table_count,
    int M,
    int H,
    int d,
    int kv_len,
    int topk,
    float scale)
{
    int h = (int)__builtin_amdgcn_workgroup_id_x();
    int m = (int)__builtin_amdgcn_workgroup_id_y();
    int tid = (int)__builtin_amdgcn_workitem_id_x();
    if (h >= H || m >= M) return;

    const float* q_ptr = Q + ((unsigned long long)m * (unsigned long long)H + (unsigned long long)h) * (unsigned long long)d;
    const int* idx_ptr = idx + (unsigned long long)m * (unsigned long long)topk;
    float* o_ptr = O + ((unsigned long long)m * (unsigned long long)H + (unsigned long long)h) * (unsigned long long)d;

    __attribute__((shared)) static float s_q[512];
    __attribute__((shared)) static int   s_tok[64];
    __attribute__((shared)) static float s_scores[64];
    __attribute__((shared)) static float s_p[64];
    __attribute__((shared)) static float s_red[64];
    __attribute__((shared)) static float s_m;
    __attribute__((shared)) static float s_l;
    __attribute__((shared)) static float s_alpha;

    float o_acc[8];
    for (int i = 0; i < 8; ++i) o_acc[i] = 0.0f;

    for (int i = 0; i < 8; ++i) {
        int j = tid + i * 64;
        if (j < d) {
            s_q[j] = q_ptr[j];
        }
    }
    if (tid == 0) {
        s_m = -1e30f;
        s_l = 0.0f;
    }
    __builtin_amdgcn_s_barrier();

    for (int c0 = 0; c0 < topk; c0 += 64) {
        int c_len = topk - c0;
        if (c_len > 64) c_len = 64;

        // 1. Compute score[tid] for t = c0 + tid
        if (tid < c_len) {
            int tok = idx_ptr[c0 + tid];
            s_tok[tid] = tok;
            if (tok < 0 || tok >= kv_len) {
                s_scores[tid] = uint_as_f32(0xFF800000u); // -inf
            } else {
                const float* k_row = K + (unsigned long long)tok * (unsigned long long)d;
                float dot = 0.0f;
                for (int k = 0; k < d; ++k) {
                    dot += s_q[k] * k_row[k];
                }
                s_scores[tid] = dot * scale;
            }
        }
        __builtin_amdgcn_s_barrier();

        // 2. Chunk max and online softmax update
        if (tid == 0) {
            float m_old = s_m;
            float m_new = m_old;
            for (int t = 0; t < c_len; ++t) {
                m_new = __builtin_fmaxf(m_new, s_scores[t]);
            }
            s_m = m_new;
            s_alpha = sattn_expf(m_old - m_new, exptable, table_count);
        }
        __builtin_amdgcn_s_barrier();

        if (tid < c_len) {
            float p = sattn_expf(s_scores[tid] - s_m, exptable, table_count);
            s_red[tid] = p;
            s_p[tid] = round_bf16_in_f32(p);
        } else {
            s_red[tid] = 0.0f;
            s_p[tid] = 0.0f;
        }
        __builtin_amdgcn_s_barrier();

        if (tid == 0) {
            float psum = 0.0f;
            for (int t = 0; t < c_len; ++t) psum += s_red[t];
            s_l = s_alpha * s_l + psum;
        }
        __builtin_amdgcn_s_barrier();

        float alpha = s_alpha;
        for (int i = 0; i < 8; ++i) {
            int j = tid + i * 64;
            if (j < d) {
                float acc = alpha * o_acc[i];
                for (int t = 0; t < c_len; ++t) {
                    int tok = s_tok[t];
                    if (tok >= 0 && tok < kv_len) {
                        acc += s_p[t] * V[(unsigned long long)tok * (unsigned long long)d + (unsigned long long)j];
                    }
                }
                o_acc[i] = acc;
            }
        }
        __builtin_amdgcn_s_barrier();
    }

    if (tid == 0) {
        float sink_val = sink ? sink[h] : uint_as_f32(0xFF800000u);
        s_l = s_l + sattn_expf(sink_val - s_m, exptable, table_count);
    }
    __builtin_amdgcn_s_barrier();

    float l_final = s_l;
    for (int i = 0; i < 8; ++i) {
        int j = tid + i * 64;
        if (j < d) {
            o_ptr[j] = round_bf16_in_f32(o_acc[i] / l_final);
        }
    }
}

// 1 workgroup (32 threads) per 32-element block of x[K], matching act_quant32_scalar + (int)(qv * 512.0f).
extern "C" __global__ void hv41_act_quant32_kernel(
    const float* __restrict__ x,
    int*         __restrict__ av,
    float*       __restrict__ sv,
    int K)
{
    int g = (int)__builtin_amdgcn_workgroup_id_x();
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int idx = (g << 5) + lane;
    if (idx >= K) return;

    float v = x[idx];
    float av_abs = uint_as_f32(f32_as_uint(v) & 0x7FFFFFFFu);

    __attribute__((shared)) static float smax[32];
    smax[lane] = av_abs;
    __builtin_amdgcn_s_barrier();
    if (lane == 0) {
        float m = 0.0f;
        for (int i = 0; i < 32; ++i) {
            m = __builtin_fmaxf(m, smax[i]);
        }
        m = __builtin_fmaxf(m, uint_as_f32(0x38D1B717u)); // 1e-4f
        float s_raw = m * uint_as_f32(0x3B124925u);       // m / 448.0f
        unsigned int u = f32_as_uint(s_raw);
        unsigned int e = ((u >> 23) & 0xFFu) + ((u & 0x7FFFFFu) != 0u ? 1u : 0u);
        float sc = uint_as_f32(e << 23);
        smax[0] = sc;
        sv[g] = sc;
    }
    __builtin_amdgcn_s_barrier();

    float scale = smax[0];
    float ratio = v / scale;
    float clamped = ratio;
    if (!(ratio != ratio)) { // not NaN
        clamped = __builtin_fmaxf(-448.0f, __builtin_fminf(448.0f, ratio));
        if (clamped != 0.0f) {
            float abs_c = uint_as_f32(f32_as_uint(clamped) & 0x7FFFFFFFu);
            int exp_unbiased = (int)((f32_as_uint(abs_c) >> 23) & 0xFFu) - 127;
            int e_q = (exp_unbiased < -6) ? -6 : exp_unbiased;
            float step = uint_as_f32((unsigned int)(e_q + 124) << 23);
            float rounded = __builtin_rintf(abs_c / step) * step;
            float mag = __builtin_fminf(448.0f, rounded);
            unsigned int sign = f32_as_uint(clamped) & 0x80000000u;
            clamped = uint_as_f32(sign | f32_as_uint(mag));
        }
    }
    av[idx] = (int)(clamped * 512.0f);
}

extern "C" __global__ void hv41_shx_silu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float*       __restrict__ out,
    int N)
{
    int i = (int)(__builtin_amdgcn_workgroup_id_x() * 64u + __builtin_amdgcn_workitem_id_x());
    if (i >= N) return;
    float g = __builtin_fminf(10.0f, gate[i]);
    float u = up[i];
    u = (u < -10.0f) ? -10.0f : __builtin_fminf(10.0f, u);
    float silu = g / (1.0f + __builtin_expf(-g));
    out[i] = round_bf16_in_f32(silu * u);
}

extern "C" __global__ void hv41_e1_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ out,
    int K,
    float eps)
{
    int tid = (int)__builtin_amdgcn_workitem_id_x();
    __attribute__((shared)) static float s_sum[64];
    __attribute__((shared)) static float s_inv;

    float local_sq = 0.0f;
    for (int i = tid; i < K; i += 64) {
        float v = x[i];
        local_sq += v * v;
    }
    s_sum[tid] = local_sq;
    __builtin_amdgcn_s_barrier();

    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 64; ++i) tot += s_sum[i];
        float rms = __builtin_sqrtf(tot / (float)K + eps);
        s_inv = 1.0f / rms;
    }
    __builtin_amdgcn_s_barrier();

    float inv = s_inv;
    for (int i = tid; i < K; i += 64) {
        out[i] = round_bf16_in_f32(inv * x[i] * w[i]);
    }
}
)HIP";

const char* kHipKernelSource3 = R"HIP(
__device__ static __inline__ unsigned int corr_read_bits(
    const unsigned char* __restrict__ bytes,
    unsigned long long bit_pos,
    int nbits)
{
    unsigned long long byte_idx = bit_pos >> 3;
    unsigned int sh = (unsigned int)(bit_pos & 7ull);
    unsigned int w = (unsigned int)bytes[byte_idx]
                   | ((unsigned int)bytes[byte_idx + 1ull] << 8);
    unsigned int mask = (nbits == 4) ? 0xFu : 0x1u;
    return (w >> sh) & mask;
}

extern "C" __global__ void hv41_plane_restore_kernel(
    const unsigned char* __restrict__ field,
    const unsigned char* __restrict__ plane1,
    const unsigned int* __restrict__ patch_idx,
    const unsigned char* __restrict__ patch_bytes,
    const unsigned char* __restrict__ lut32kb,
    int N,
    int K,
    int zs,
    unsigned char* __restrict__ out_codes,
    unsigned char* __restrict__ out_scales)
{
    int g   = (int)__builtin_amdgcn_workgroup_id_x();
    int tid = (int)__builtin_amdgcn_workitem_id_x(); // 0..255
    int G_n = N >> 3;
    if (g >= G_n) return;
    int K_blk = K >> 5;
    int total_rows = K_blk << 3;
    long long G_tot = (long long)G_n * (long long)K_blk;

    const unsigned char* p0  = field  + (long long)g * K_blk * 64ll;
    const unsigned char* p1  = plane1 + (long long)g * K_blk * 64ll;
    const unsigned char* tbl = field  + G_tot * 64ll + (long long)g * K_blk * 19ll;

    __attribute__((shared)) static unsigned int s_scan[256];
    __attribute__((shared)) static unsigned long long s_base_bit;

    if (tid == 0) {
        s_base_bit = (unsigned long long)patch_idx[g];
    }
    __builtin_amdgcn_s_barrier();

    for (int base = 0; base < total_rows; base += 256) {
        int row_idx = base + tid;
        unsigned int row_u32[4] = { 0u, 0u, 0u, 0u };
        unsigned int mask4 = 0u, mask1 = 0u;
        unsigned int my_bits = 0u;
        int kb = row_idx >> 3;
        int r  = row_idx & 7;

        if (row_idx < total_rows) {
            const unsigned char* t = tbl + (long long)kb * 19ll;
            out_scales[(long long)(g * 8 + r) * K_blk + kb] = t[11 + r];

            int bit_off = r * 11;
            int b_i = bit_off >> 3;
            int b_s = bit_off & 7;
            unsigned int idx_w = (unsigned int)t[b_i]
                               | ((unsigned int)t[b_i + 1] << 8)
                               | ((unsigned int)t[b_i + 2] << 16);
            int lut_idx = (int)((idx_w >> b_s) & 0x7FFu);
            const unsigned char* lut = lut32kb + (long long)lut_idx * 16ll;

            int h = r >> 2;
            int q = r & 3;
            const unsigned char* p0_8 = p0 + (long long)kb * 64ll + h * 32 + q * 8;
            const unsigned char* p1_8 = p1 + (long long)kb * 64ll + h * 32 + q * 8;

            #pragma unroll 8
            for (int i = 0; i < 8; ++i) {
                unsigned int a0 = (unsigned int)p0_8[i];
                unsigned int a1 = (unsigned int)p1_8[i];
                unsigned int b_lo = ((a1 & 0x33u) << 2) | (a0 & 0x33u);
                unsigned int b_hi = ((a0 >> 2) & 0x33u) | (a1 & 0xCCu);

                unsigned int lo0 = b_lo & 0xFu;
                unsigned int hi0 = (b_lo >> 4) & 0xFu;
                unsigned int o_lo0 = lut[lo0];
                unsigned int o_hi0 = lut[hi0];
                row_u32[i >> 2] |= ((o_hi0 << 4) | o_lo0) << ((i & 3) * 8);
                if (lo0 > 11u) mask4 |= (1u << i);
                if (hi0 > 11u) mask4 |= (1u << (i + 16));
                if (o_lo0 == 0u) mask1 |= (1u << i);
                if (o_hi0 == 0u) mask1 |= (1u << (i + 16));

                unsigned int lo1 = b_hi & 0xFu;
                unsigned int hi1 = (b_hi >> 4) & 0xFu;
                unsigned int o_lo1 = lut[lo1];
                unsigned int o_hi1 = lut[hi1];
                row_u32[(8 + i) >> 2] |= ((o_hi1 << 4) | o_lo1) << ((i & 3) * 8);
                if (lo1 > 11u) mask4 |= (1u << (8 + i));
                if (hi1 > 11u) mask4 |= (1u << (24 + i));
                if (o_lo1 == 0u) mask1 |= (1u << (8 + i));
                if (o_hi1 == 0u) mask1 |= (1u << (24 + i));
            }
            my_bits = (unsigned int)__builtin_popcount(mask4) * 4u
                    + (zs ? (unsigned int)__builtin_popcount(mask1) : 0u);
        }

        s_scan[tid] = my_bits;
        __builtin_amdgcn_s_barrier();
        #pragma unroll
        for (int off = 1; off < 256; off <<= 1) {
            unsigned int v = (tid >= off) ? s_scan[tid - off] : 0u;
            __builtin_amdgcn_s_barrier();
            s_scan[tid] += v;
            __builtin_amdgcn_s_barrier();
        }
        unsigned int excl = s_scan[tid] - my_bits;
        unsigned long long bit_pos = s_base_bit + (unsigned long long)excl;

        if (row_idx < total_rows) {
            while (mask4) {
                int pos = __builtin_ctz(mask4);
                mask4 &= (mask4 - 1u);
                unsigned int nib = corr_read_bits(patch_bytes, bit_pos, 4);
                bit_pos += 4ull;
                int bi = pos & 15;
                int wi = bi >> 2;
                int sh = (bi & 3) * 8 + ((pos < 16) ? 0 : 4);
                row_u32[wi] = (row_u32[wi] & ~(0xFu << sh)) | (nib << sh);
            }
            if (zs != 0) {
                while (mask1) {
                    int pos = __builtin_ctz(mask1);
                    mask1 &= (mask1 - 1u);
                    unsigned int bit = corr_read_bits(patch_bytes, bit_pos, 1);
                    bit_pos += 1ull;
                    if (bit) {
                        int bi = pos & 15;
                        int wi = bi >> 2;
                        int sh = (bi & 3) * 8 + ((pos < 16) ? 3 : 7);
                        row_u32[wi] |= (1u << sh);
                    }
                }
            }
            u4_t* dst = (u4_t*)(out_codes + (long long)(g * 8 + r) * (K >> 1) + (long long)kb * 16ll);
            u4_t val;
            val.x = row_u32[0];
            val.y = row_u32[1];
            val.z = row_u32[2];
            val.w = row_u32[3];
            *dst = val;
        }
        if (tid == 255) {
            s_base_bit += (unsigned long long)s_scan[255];
        }
        __builtin_amdgcn_s_barrier();
    }
}

extern "C" __global__ void hv41_fp4_linear_kernel(
    const unsigned char* __restrict__ codes,
    const unsigned char* __restrict__ scales,
    const int* __restrict__ av,
    const float* __restrict__ sv,
    const unsigned char* __restrict__ bad,
    float* __restrict__ y,
    int M,
    int N,
    int K)
{
    int r = (int)__builtin_amdgcn_workgroup_id_x() * 64 + (int)__builtin_amdgcn_workitem_id_x();
    int m = (int)__builtin_amdgcn_workgroup_id_y();
    if (r >= N || m >= M) return;
    if (bad && bad[m] != 0) {
        union { unsigned int u; float f; } nan_u;
        nan_u.u = 0x7FC00000u;
        y[(long long)m * N + r] = nan_u.f;
        return;
    }

    static const int KV[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    int K_blk = K >> 5;
    const unsigned char* row_c = codes + (long long)r * (K >> 1);
    const unsigned char* row_s = scales + (long long)r * K_blk;
    const int* av_m = av + (long long)m * K;
    const float* sv_m = sv + (long long)m * K_blk;

    float acc = 0.0f;
    for (int g = 0; g < K_blk; ++g) {
        const unsigned char* cg = row_c + (long long)g * 16ll;
        const int* ag = av_m + g * 32;
        int dot = 0;
        for (int j = 0; j < 16; ++j) {
            unsigned char b = cg[j];
            dot += KV[b & 0xFu] * ag[j] + KV[(b >> 4) & 0xFu] * ag[16 + j];
        }
        union { unsigned long long u; double d; } sc_u;
        sc_u.u = (unsigned long long)((unsigned int)row_s[g] + 0x376u) << 52;
        acc += (float)((double)dot * sc_u.d) * sv_m[g];
    }
    y[(long long)m * N + r] = round_bf16_in_f32(acc);
}
)HIP";

struct WeightHandle {
    int N = 0;
    int K = 0;
    void* d_codes = nullptr;
    void* d_scales = nullptr;
};

struct Bf16WeightHandle {
    int N = 0;
    int K = 0;
    void* d_W = nullptr;
};

struct State {
    std::mutex mu;
    bool initialized = false;
    std::string init_info;
    char last_error[1024] = {};

    ComgrApi comgr;
    HipApi hip;
    void* hip_module = nullptr;
    void* fn_m1 = nullptr;
    void* fn_m1_w8 = nullptr;
    void* fn_m1_w16 = nullptr;
    void* fn_tile4 = nullptr;
    void* fn_bf16_seq = nullptr;
    void* fn_bf16_w32 = nullptr;
    void* fn_sattn = nullptr;
    void* fn_act_quant32 = nullptr;
    void* fn_shx_silu = nullptr;
    void* fn_e1_rmsnorm = nullptr;
    void* fn_plane_restore = nullptr;
    void* fn_fp4_linear = nullptr;
    void* stream = nullptr;
    void* ev_start = nullptr;
    void* ev_stop = nullptr;

    void* d_a = nullptr;
    size_t d_a_bytes = 0;
    void* d_s = nullptr;
    size_t d_s_bytes = 0;
    void* d_out = nullptr;
    size_t d_out_bytes = 0;

    void* d_bf16_x = nullptr;
    size_t d_bf16_x_bytes = 0;
    void* d_bf16_y = nullptr;
    size_t d_bf16_y_bytes = 0;

    void* d_exptable = nullptr;
    int exptable_count = 0;

    void* d_sa_q = nullptr; size_t d_sa_q_bytes = 0;
    void* d_sa_k = nullptr; size_t d_sa_k_bytes = 0;
    void* d_sa_idx = nullptr; size_t d_sa_idx_bytes = 0;
    void* d_sa_v = nullptr; size_t d_sa_v_bytes = 0;
    void* d_sa_sink = nullptr; size_t d_sa_sink_bytes = 0;
    void* d_sa_o = nullptr; size_t d_sa_o_bytes = 0;

    void* d_sq_tmp0 = nullptr; size_t d_sq_tmp0_bytes = 0;
    void* d_sq_tmp1 = nullptr; size_t d_sq_tmp1_bytes = 0;
    void* d_sq_tmp2 = nullptr; size_t d_sq_tmp2_bytes = 0;
    void* d_sq_av = nullptr;   size_t d_sq_av_bytes = 0;
    void* d_sq_sv = nullptr;   size_t d_sq_sv_bytes = 0;
    void* d_sq_out0 = nullptr; size_t d_sq_out0_bytes = 0;
    void* d_sq_out1 = nullptr; size_t d_sq_out1_bytes = 0;

    static constexpr size_t kMaxPinnedStage = 2ull * 1024ull * 1024ull;
    void* h_in0 = nullptr;  size_t h_in0_bytes = 0;
    void* h_in1 = nullptr;  size_t h_in1_bytes = 0;
    void* h_out0 = nullptr; size_t h_out0_bytes = 0;
    void* h_out1 = nullptr; size_t h_out1_bytes = 0;
    void* h_out2 = nullptr; size_t h_out2_bytes = 0;
    void* h_out3 = nullptr; size_t h_out3_bytes = 0;

    std::atomic<size_t> resident_bytes{ 0 };
};

State& state() {
    static State* s = new State();
    return *s;
}

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(state().last_error, sizeof(state().last_error), fmt, ap);
    va_end(ap);
}

static bool extract_comgr_logs(ComgrApi& c, amd_comgr_data_set_s ds, std::string& out_log) {
    out_log.clear();
    for (amd_comgr_data_kind_t kind : { AMD_COMGR_DATA_KIND_LOG, AMD_COMGR_DATA_KIND_DIAGNOSTIC }) {
        size_t count = 0;
        if (c.action_data_count(ds, kind, &count) != AMD_COMGR_STATUS_SUCCESS) continue;
        for (size_t i = 0; i < count; ++i) {
            amd_comgr_data_s d{ 0 };
            if (c.action_data_get_data(ds, kind, i, &d) != AMD_COMGR_STATUS_SUCCESS) continue;
            size_t sz = 0;
            if (c.get_data(d, &sz, nullptr) == AMD_COMGR_STATUS_SUCCESS && sz > 0) {
                std::string chunk(sz, '\0');
                if (c.get_data(d, &sz, chunk.data()) == AMD_COMGR_STATUS_SUCCESS) {
                    if (!out_log.empty()) out_log += "\n";
                    out_log += chunk;
                }
            }
            c.release_data(d);
        }
    }
    return !out_log.empty();
}

static bool compile_hsaco_gfx1031(ComgrApi& c, const char* isa_name, std::vector<char>& hsaco_out) {
    amd_comgr_data_s src_data{ 0 };
    amd_comgr_data_set_s src_set{ 0 }, bc_set{ 0 }, reloc_set{ 0 }, exec_set{ 0 };
    amd_comgr_action_info_s info{ 0 };
    bool ok = false;

    const char* err_ctx = "init";
    amd_comgr_status_t rc = AMD_COMGR_STATUS_SUCCESS;
    std::string build_log;

    auto fail = [&](const char* step, amd_comgr_status_t status, amd_comgr_data_set_s ds) -> bool {
        const char* status_str = nullptr;
        c.status_string(status, &status_str);
        if (ds.handle) extract_comgr_logs(c, ds, build_log);
        set_error("comgr %s failed (%d: %s)%s%s",
            step, status, status_str ? status_str : "?",
            build_log.empty() ? "" : " | ", build_log.c_str());
        return false;
    };

    rc = c.create_data(AMD_COMGR_DATA_KIND_SOURCE, &src_data);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "create_data"; goto cleanup; }
    {
        std::string full_hip_src = std::string(kHipKernelSource) + kHipKernelSource2 + kHipKernelSource3;
        rc = c.set_data(src_data, full_hip_src.size(), full_hip_src.c_str());
    }
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_data"; goto cleanup; }
    rc = c.set_data_name(src_data, "hv41_fp8_linear.hip");
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_data_name"; goto cleanup; }

    rc = c.create_data_set(&src_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "create_data_set(src)"; goto cleanup; }
    rc = c.data_set_add(src_set, src_data);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "data_set_add"; goto cleanup; }

    rc = c.create_action_info(&info);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "create_action_info"; goto cleanup; }
    rc = c.action_info_set_isa_name(info, isa_name);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_isa_name"; goto cleanup; }
    rc = c.action_info_set_language(info, AMD_COMGR_LANGUAGE_HIP);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_language"; goto cleanup; }
    c.action_info_set_logging(info, true);

    {
        const char* opts[] = { "-O3", "-mcumode", "-ffp-contract=off" };
        rc = c.action_info_set_option_list(info, opts, 3);
        if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_option_list"; goto cleanup; }
    }

    rc = c.create_data_set(&bc_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "create_data_set(bc)"; goto cleanup; }
    rc = c.do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, src_set, bc_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { fail("COMPILE_SOURCE_TO_BC", rc, bc_set); goto cleanup_err_already_set; }

    c.destroy_action_info(info);
    info.handle = 0;
    rc = c.create_action_info(&info);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "reinit_action_info"; goto cleanup; }
    rc = c.action_info_set_isa_name(info, isa_name);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_isa_name(codegen)"; goto cleanup; }
    c.action_info_set_logging(info, true);
    {
        const char* cg_opts[] = { "-O3", "-mcumode" };
        rc = c.action_info_set_option_list(info, cg_opts, 2);
        if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "set_option_list(codegen)"; goto cleanup; }
    }

    rc = c.create_data_set(&reloc_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "create_data_set(reloc)"; goto cleanup; }
    rc = c.do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { fail("CODEGEN_BC_TO_RELOCATABLE", rc, reloc_set); goto cleanup_err_already_set; }

    rc = c.create_data_set(&exec_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "create_data_set(exec)"; goto cleanup; }
    rc = c.do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc_set, exec_set);
    if (rc != AMD_COMGR_STATUS_SUCCESS) { fail("LINK_RELOCATABLE_TO_EXECUTABLE", rc, exec_set); goto cleanup_err_already_set; }

    {
        size_t exe_cnt = 0;
        rc = c.action_data_count(exec_set, AMD_COMGR_DATA_KIND_EXECUTABLE, &exe_cnt);
        if (rc != AMD_COMGR_STATUS_SUCCESS || exe_cnt == 0) {
            set_error("comgr produced 0 executables (rc=%d)", rc);
            goto cleanup_err_already_set;
        }
        amd_comgr_data_s exe_obj{ 0 };
        rc = c.action_data_get_data(exec_set, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &exe_obj);
        if (rc != AMD_COMGR_STATUS_SUCCESS) { err_ctx = "get_exe"; goto cleanup; }
        size_t exe_sz = 0;
        rc = c.get_data(exe_obj, &exe_sz, nullptr);
        if (rc == AMD_COMGR_STATUS_SUCCESS && exe_sz > 0) {
            hsaco_out.resize(exe_sz);
            rc = c.get_data(exe_obj, &exe_sz, hsaco_out.data());
        }
        c.release_data(exe_obj);
        if (rc != AMD_COMGR_STATUS_SUCCESS || hsaco_out.empty()) {
            set_error("comgr get_data executable failed (%d)", rc);
            goto cleanup_err_already_set;
        }
    }

    ok = true;
    goto cleanup_err_already_set;

cleanup:
    fail(err_ctx, rc, amd_comgr_data_set_s{ 0 });

cleanup_err_already_set:
    if (exec_set.handle) c.destroy_data_set(exec_set);
    if (reloc_set.handle) c.destroy_data_set(reloc_set);
    if (bc_set.handle) c.destroy_data_set(bc_set);
    if (src_set.handle) c.destroy_data_set(src_set);
    if (info.handle) c.destroy_action_info(info);
    if (src_data.handle) c.release_data(src_data);
    return ok;
}

static bool ensure_scratch(State& st, size_t need_a, size_t need_s, size_t need_out) {
    auto alloc_grow = [&](void** ptr, size_t* cap, size_t need, const char* label) -> bool {
        if (*cap >= need) return true;
        if (*ptr) {
            st.hip.free(*ptr);
            *ptr = nullptr;
            *cap = 0;
        }
        void* p = nullptr;
        int rc = st.hip.malloc(&p, need);
        if (rc != HIP_SUCCESS || !p) {
            set_error("hipMalloc scratch %s (%zu bytes) failed (%d: %s)",
                label, need, rc, st.hip.getErrorString(rc));
            return false;
        }
        *ptr = p;
        *cap = need;
        return true;
    };
    return alloc_grow(&st.d_a, &st.d_a_bytes, need_a, "d_a") &&
           alloc_grow(&st.d_s, &st.d_s_bytes, need_s, "d_s") &&
           alloc_grow(&st.d_out, &st.d_out_bytes, need_out, "d_out");
}

static bool ensure_pinned_buf(State& st, void*& ptr, size_t& cap, size_t need_bytes) {
    if (need_bytes == 0 || need_bytes > State::kMaxPinnedStage) return false;
    if (cap >= need_bytes && ptr != nullptr) return true;
    if (ptr) {
        st.hip.hostFree(ptr);
        ptr = nullptr;
        cap = 0;
    }
    int rc = st.hip.hostMalloc(&ptr, need_bytes, 0u);
    if (rc != HIP_SUCCESS || !ptr) {
        ptr = nullptr;
        cap = 0;
        return false;
    }
    cap = need_bytes;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Exported hv41c_* Core API (MSVC C ABI)
// ---------------------------------------------------------------------------

extern "C" __declspec(dllexport) const char* hv41c_error(void) {
    return state().last_error;
}

static int load_exptable_locked(State& st, const char* path);

extern "C" __declspec(dllexport) int hv41c_init(char* info_buf, int info_buf_size) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);

    if (st.initialized) {
        if (info_buf && info_buf_size > 0) {
            snprintf(info_buf, (size_t)info_buf_size, "%s", st.init_info.c_str());
        }
        return 0;
    }

    if (!st.comgr.load()) {
        set_error("failed to load amd_comgr_2.dll (win32=%lu)", GetLastError());
        return -1;
    }
    if (!st.hip.load()) {
        set_error("failed to load amdhip64_6.dll (win32=%lu)", GetLastError());
        return -1;
    }

    int rc = st.hip.init(0);
    if (rc != HIP_SUCCESS) {
        set_error("hipInit failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.setDevice(0);
    if (rc != HIP_SUCCESS) {
        set_error("hipSetDevice(0) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }

    char dev_name[256] = {};
    if (st.hip.deviceGetName(dev_name, (int)sizeof(dev_name), 0) != HIP_SUCCESS) {
        snprintf(dev_name, sizeof(dev_name), "AMD Radeon GPU");
    }
    int cc_maj = 10, cc_min = 3;
    st.hip.deviceComputeCapability(&cc_maj, &cc_min, 0);
    int wgps = 0;
    st.hip.deviceGetAttribute(&wgps, HIP_DEVICE_ATTR_MP_COUNT, 0);

    // Format gfxISA string from compute capability (e.g. 10.3 -> gfx1030/1031; override or target gfx1031 on 10.3)
    char isa_buf[64];
    const char* env_arch = std::getenv("HV41C_HIP_ARCH");
    if (env_arch && *env_arch) {
        snprintf(isa_buf, sizeof(isa_buf), "amdgcn-amd-amdhsa--%s", env_arch);
    } else if (cc_maj == 10 && cc_min == 3) {
        snprintf(isa_buf, sizeof(isa_buf), "amdgcn-amd-amdhsa--gfx1031");
    } else {
        snprintf(isa_buf, sizeof(isa_buf), "amdgcn-amd-amdhsa--gfx%d%d0", cc_maj, cc_min);
    }

    std::vector<char> hsaco;
    if (!compile_hsaco_gfx1031(st.comgr, isa_buf, hsaco)) {
        return -1;
    }

    rc = st.hip.moduleLoadData(&st.hip_module, hsaco.data());
    if (rc != HIP_SUCCESS || !st.hip_module) {
        set_error("hipModuleLoadData failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_m1, st.hip_module, "hv41_fp8_linear_m1");
    if (rc != HIP_SUCCESS || !st.fn_m1) {
        set_error("hipModuleGetFunction(hv41_fp8_linear_m1) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_m1_w8, st.hip_module, "hv41_fp8_linear_m1_w8");
    if (rc != HIP_SUCCESS || !st.fn_m1_w8) {
        set_error("hipModuleGetFunction(hv41_fp8_linear_m1_w8) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_m1_w16, st.hip_module, "hv41_fp8_linear_m1_w16");
    if (rc != HIP_SUCCESS || !st.fn_m1_w16) {
        set_error("hipModuleGetFunction(hv41_fp8_linear_m1_w16) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_tile4, st.hip_module, "hv41_fp8_linear_tile4");
    if (rc != HIP_SUCCESS || !st.fn_tile4) {
        set_error("hipModuleGetFunction(hv41_fp8_linear_tile4) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_bf16_seq, st.hip_module, "hv41_bf16_seq_kernel");
    if (rc != HIP_SUCCESS || !st.fn_bf16_seq) {
        set_error("hipModuleGetFunction(hv41_bf16_seq_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_bf16_w32, st.hip_module, "hv41_bf16_w32_kernel");
    if (rc != HIP_SUCCESS || !st.fn_bf16_w32) {
        set_error("hipModuleGetFunction(hv41_bf16_w32_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_sattn, st.hip_module, "hv41_sattn_kernel");
    if (rc != HIP_SUCCESS || !st.fn_sattn) {
        set_error("hipModuleGetFunction(hv41_sattn_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_act_quant32, st.hip_module, "hv41_act_quant32_kernel");
    if (rc != HIP_SUCCESS || !st.fn_act_quant32) {
        set_error("hipModuleGetFunction(hv41_act_quant32_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_shx_silu, st.hip_module, "hv41_shx_silu_kernel");
    if (rc != HIP_SUCCESS || !st.fn_shx_silu) {
        set_error("hipModuleGetFunction(hv41_shx_silu_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_e1_rmsnorm, st.hip_module, "hv41_e1_rmsnorm_kernel");
    if (rc != HIP_SUCCESS || !st.fn_e1_rmsnorm) {
        set_error("hipModuleGetFunction(hv41_e1_rmsnorm_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_plane_restore, st.hip_module, "hv41_plane_restore_kernel");
    if (rc != HIP_SUCCESS || !st.fn_plane_restore) {
        set_error("hipModuleGetFunction(hv41_plane_restore_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    rc = st.hip.moduleGetFunction(&st.fn_fp4_linear, st.hip_module, "hv41_fp4_linear_kernel");
    if (rc != HIP_SUCCESS || !st.fn_fp4_linear) {
        set_error("hipModuleGetFunction(hv41_fp4_linear_kernel) failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }

    rc = st.hip.streamCreate(&st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("hipStreamCreate failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -1;
    }
    if (st.hip.eventCreate(&st.ev_start) != HIP_SUCCESS ||
        st.hip.eventCreate(&st.ev_stop) != HIP_SUCCESS) {
        set_error("hipEventCreate failed");
        return -1;
    }

    // Pre-allocate scratch for common non-expert shapes (rows=8, K=8192, N=32768)
    ensure_scratch(st, 64ull * 8192ull * sizeof(int32_t),
                       64ull * 256ull * sizeof(float),
                       64ull * 32768ull * sizeof(float));
    ensure_pinned_buf(st, st.h_in0,  st.h_in0_bytes,  256ull * 1024ull);
    ensure_pinned_buf(st, st.h_in1,  st.h_in1_bytes,   64ull * 1024ull);
    ensure_pinned_buf(st, st.h_out0, st.h_out0_bytes, 256ull * 1024ull);
    ensure_pinned_buf(st, st.h_out1, st.h_out1_bytes,  64ull * 1024ull);
    ensure_pinned_buf(st, st.h_out2, st.h_out2_bytes,  64ull * 1024ull);
    ensure_pinned_buf(st, st.h_out3, st.h_out3_bytes, 128ull * 1024ull);

    size_t vfree = 0, vtotal = 0;
    st.hip.memGetInfo(&vfree, &vtotal);

    char info[512];
    snprintf(info, sizeof(info),
        "%s (%s HIP), %d WGPs, VRAM free %zu of %zu MiB at init",
        dev_name, isa_buf + strlen("amdgcn-amd-amdhsa--"), wgps,
        vfree >> 20, vtotal >> 20);
    st.init_info = info;
    st.last_error[0] = '\0';
    st.initialized = true;

    if (const char* exptab = getenv("HV41C_EXPTABLE")) {
        if (exptab[0]) {
            load_exptable_locked(st, exptab);
        }
    }

    if (info_buf && info_buf_size > 0) {
        snprintf(info_buf, (size_t)info_buf_size, "%s", st.init_info.c_str());
    }
    return 0;
}

extern "C" __declspec(dllexport) void* hv41c_upload(
    const uint8_t* codes,
    const uint8_t* scales,
    int N,
    int K)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);

    if (!st.initialized) {
        set_error("hv41c_upload called before hv41c_init");
        return nullptr;
    }
    if (!codes || !scales || N <= 0 || K <= 0 || (N & 31) != 0 || (K & 31) != 0) {
        set_error("hv41c_upload invalid arguments (N=%d, K=%d)", N, K);
        return nullptr;
    }

    size_t codes_bytes = (size_t)N * (size_t)K;
    size_t scales_bytes = (size_t)(N / 32) * (size_t)(K / 32);

    // Repack row-major codes [N, K] into wave32 coalesced tiles [N/32][K/32][8][32] of uint32_t
    int nnb = N >> 5;
    int nkb = K >> 5;
    std::vector<uint32_t> packed((size_t)nnb * (size_t)nkb * 8u * 32u);
    for (int nb = 0; nb < nnb; ++nb) {
        for (int lane = 0; lane < 32; ++lane) {
            int n = (nb << 5) + lane;
            const uint8_t* src_row = codes + (size_t)n * (size_t)K;
            for (int kb = 0; kb < nkb; ++kb) {
                const uint8_t* blk = src_row + ((size_t)kb << 5);
                size_t dst_base = (((size_t)nb * (size_t)nkb + (size_t)kb) << 3);
                for (int grp = 0; grp < 8; ++grp) {
                    uint32_t w4 = (uint32_t)blk[grp * 4 + 0]
                                | ((uint32_t)blk[grp * 4 + 1] << 8)
                                | ((uint32_t)blk[grp * 4 + 2] << 16)
                                | ((uint32_t)blk[grp * 4 + 3] << 24);
                    packed[((dst_base + (size_t)grp) << 5) + (size_t)lane] = w4;
                }
            }
        }
    }

    void* d_codes = nullptr;
    void* d_scales = nullptr;
    int rc = st.hip.malloc(&d_codes, codes_bytes);
    if (rc != HIP_SUCCESS || !d_codes) {
        set_error("hipMalloc d_codes (%zu bytes) failed (%d: %s)", codes_bytes, rc, st.hip.getErrorString(rc));
        return nullptr;
    }
    rc = st.hip.malloc(&d_scales, scales_bytes);
    if (rc != HIP_SUCCESS || !d_scales) {
        st.hip.free(d_codes);
        set_error("hipMalloc d_scales (%zu bytes) failed (%d: %s)", scales_bytes, rc, st.hip.getErrorString(rc));
        return nullptr;
    }

    rc = st.hip.memcpy(d_codes, packed.data(), codes_bytes, HIP_MEMCPY_HOST_TO_DEVICE);
    if (rc != HIP_SUCCESS) {
        st.hip.free(d_codes);
        st.hip.free(d_scales);
        set_error("hipMemcpy d_codes H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return nullptr;
    }
    rc = st.hip.memcpy(d_scales, scales, scales_bytes, HIP_MEMCPY_HOST_TO_DEVICE);
    if (rc != HIP_SUCCESS) {
        st.hip.free(d_codes);
        st.hip.free(d_scales);
        set_error("hipMemcpy d_scales H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return nullptr;
    }

    WeightHandle* h = new WeightHandle();
    h->N = N;
    h->K = K;
    h->d_codes = d_codes;
    h->d_scales = d_scales;

    st.resident_bytes.fetch_add(codes_bytes + scales_bytes, std::memory_order_relaxed);
    return h;
}

static int launch_fp8_m1_dev(State& st, const WeightHandle* w, void* d_av, void* d_sv, void* d_out) {
    void* p_codes = w->d_codes;
    void* p_scales = w->d_scales;
    void* p_a = d_av;
    void* p_s = d_sv;
    void* p_out = d_out;
    int p_N = w->N;
    int p_K = w->K;
    void* args[] = { &p_codes, &p_scales, &p_a, &p_s, &p_out, &p_N, &p_K };
    int nkb = w->K >> 5;
    void* fn = st.fn_m1;
    unsigned int block_x = 32u;
    if (nkb >= 8) {
        if (w->N <= 2560) {
            fn = st.fn_m1_w16;
            block_x = 512u;
        } else {
            fn = st.fn_m1_w8;
            block_x = 256u;
        }
    }
    return st.hip.moduleLaunchKernel(
        fn, (unsigned int)(w->N >> 5), 1u, 1u, block_x, 1u, 1u, 0u, st.stream, args, nullptr);
}

extern "C" __declspec(dllexport) int hv41c_linear(
    void* handle,
    const int32_t* a,
    const float* s,
    int rows,
    float* out,
    float* out_ms)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);

    if (!st.initialized) {
        set_error("hv41c_linear called before hv41c_init");
        return -1;
    }
    const WeightHandle* w = static_cast<const WeightHandle*>(handle);
    if (!w || !a || !s || !out || rows <= 0 || rows > 0xFFFF) {
        set_error("hv41c_linear invalid arguments (handle=%p, rows=%d)", handle, rows);
        return -2;
    }

    int N = w->N;
    int K = w->K;
    int nkb = K >> 5;
    size_t a_bytes = (size_t)rows * (size_t)K * sizeof(int32_t);
    size_t s_bytes = (size_t)rows * (size_t)nkb * sizeof(float);
    size_t out_bytes = (size_t)rows * (size_t)N * sizeof(float);

    if (!ensure_scratch(st, a_bytes, s_bytes, out_bytes)) {
        return -3;
    }

    const void* src_a = a;
    if (ensure_pinned_buf(st, st.h_in0, st.h_in0_bytes, a_bytes)) {
        memcpy(st.h_in0, a, a_bytes);
        src_a = st.h_in0;
    }
    const void* src_s = s;
    if (ensure_pinned_buf(st, st.h_in1, st.h_in1_bytes, s_bytes)) {
        memcpy(st.h_in1, s, s_bytes);
        src_s = st.h_in1;
    }
    bool out_pinned = ensure_pinned_buf(st, st.h_out0, st.h_out0_bytes, out_bytes);
    void* dst_out = out_pinned ? st.h_out0 : static_cast<void*>(out);

    int rc = st.hip.memcpyAsync(st.d_a, src_a, a_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("hipMemcpyAsync a H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    rc = st.hip.memcpyAsync(st.d_s, src_s, s_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("hipMemcpyAsync s H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }

    if (out_ms) {
        st.hip.eventRecord(st.ev_start, st.stream);
    }

    unsigned int grid_x = (unsigned int)(N >> 5);
    if (rows == 1) {
        rc = launch_fp8_m1_dev(st, w, st.d_a, st.d_s, st.d_out);
    } else {
        unsigned int grid_y = (unsigned int)((rows + 3) / 4);
        void* p_codes  = w->d_codes;
        void* p_scales = w->d_scales;
        void* p_a      = st.d_a;
        void* p_s      = st.d_s;
        void* p_out    = st.d_out;
        int   p_N      = N;
        int   p_K      = K;
        int   p_rows   = rows;
        void* args[] = { &p_codes, &p_scales, &p_a, &p_s, &p_out, &p_N, &p_K, &p_rows };
        rc = st.hip.moduleLaunchKernel(
            st.fn_tile4,
            grid_x, grid_y, 1u,
            32u, 1u, 1u,
            0u, st.stream, args, nullptr);
    }

    if (rc != HIP_SUCCESS) {
        set_error("hipModuleLaunchKernel failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }

    if (out_ms) {
        st.hip.eventRecord(st.ev_stop, st.stream);
    }

    rc = st.hip.memcpyAsync(dst_out, st.d_out, out_bytes, HIP_MEMCPY_DEVICE_TO_HOST, st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("hipMemcpyAsync out D2H failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }

    rc = st.hip.streamSynchronize(st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("hipStreamSynchronize failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    if (out_pinned) {
        memcpy(out, st.h_out0, out_bytes);
    }

    if (out_ms) {
        float ms = 0.0f;
        if (st.hip.eventElapsedTime(&ms, st.ev_start, st.ev_stop) == HIP_SUCCESS) {
            *out_ms = ms;
        } else {
            *out_ms = 0.0f;
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) size_t hv41c_resident(void) {
    return state().resident_bytes.load(std::memory_order_relaxed);
}

extern "C" __declspec(dllexport) size_t hv41c_vram_free(void) {
    State& st = state();
    if (!st.initialized || !st.hip.memGetInfo) return 0;
    size_t vfree = 0, vtotal = 0;
    if (st.hip.memGetInfo(&vfree, &vtotal) != HIP_SUCCESS) return 0;
    return vfree;
}

static bool ensure_dev_buf(State& st, void*& ptr, size_t& cap, size_t need_bytes) {
    if (need_bytes == 0) return true;
    if (cap >= need_bytes && ptr != nullptr) return true;
    if (ptr) {
        st.hip.free(ptr);
        ptr = nullptr;
        cap = 0;
    }
    int rc = st.hip.malloc(&ptr, need_bytes);
    if (rc != HIP_SUCCESS) {
        set_error("hipMalloc(%zu) failed (%d: %s)", need_bytes, rc, st.hip.getErrorString(rc));
        ptr = nullptr;
        cap = 0;
        return false;
    }
    cap = need_bytes;
    return true;
}

extern "C" __declspec(dllexport) void* hv41c_upload_bf16(const uint16_t* W, int N, int K) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !W || N < 1 || K < 8 || (K & 7) != 0) {
        return nullptr;
    }
    size_t bytes = (size_t)N * (size_t)K * sizeof(uint16_t);
    void* d_W = nullptr;
    int rc = st.hip.malloc(&d_W, bytes);
    if (rc != HIP_SUCCESS) {
        set_error("hv41c_upload_bf16 hipMalloc(%zu) failed (%d: %s)", bytes, rc, st.hip.getErrorString(rc));
        return nullptr;
    }
    rc = st.hip.memcpy(d_W, W, bytes, HIP_MEMCPY_HOST_TO_DEVICE);
    if (rc != HIP_SUCCESS) {
        set_error("hv41c_upload_bf16 hipMemcpy H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        st.hip.free(d_W);
        return nullptr;
    }
    auto* h = new Bf16WeightHandle();
    h->N = N;
    h->K = K;
    h->d_W = d_W;
    st.resident_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return h;
}

static int run_bf16_kernel(
    void* fn,
    void* h_void,
    const float* x,
    int M,
    int ng,
    int K_x,
    int bf16_round,
    float* y,
    float* ms_out,
    bool is_seq)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !h_void || !x || !y) return -1;
    if (M < 1 || M > 0xFFFF || ng < 1) return -2;

    const auto* h = static_cast<const Bf16WeightHandle*>(h_void);
    int N = h->N;
    int K = h->K;
    int groups = (N - 1) / ng + 1;
    if ((long long)groups * (long long)K > (long long)K_x) return -2;

    size_t x_bytes = (size_t)M * (size_t)K_x * sizeof(float);
    size_t y_bytes = (size_t)M * (size_t)N * sizeof(float);
    if (!ensure_dev_buf(st, st.d_bf16_x, st.d_bf16_x_bytes, x_bytes)) return -3;
    if (!ensure_dev_buf(st, st.d_bf16_y, st.d_bf16_y_bytes, y_bytes)) return -3;

    const void* src_x = x;
    if (ensure_pinned_buf(st, st.h_in0, st.h_in0_bytes, x_bytes)) {
        memcpy(st.h_in0, x, x_bytes);
        src_x = st.h_in0;
    }
    bool y_pinned = ensure_pinned_buf(st, st.h_out0, st.h_out0_bytes, y_bytes);
    void* dst_y = y_pinned ? st.h_out0 : static_cast<void*>(y);

    int rc = st.hip.memcpyAsync(st.d_bf16_x, src_x, x_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("bf16 H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_start, st.stream);

    void* p_W = h->d_W;
    void* p_x = st.d_bf16_x;
    void* p_y = st.d_bf16_y;
    int p_M = M, p_N = N, p_K = K, p_ng = ng, p_Kx = K_x, p_rnd = bf16_round ? 1 : 0;
    void* args[] = { &p_W, &p_x, &p_y, &p_M, &p_N, &p_K, &p_ng, &p_Kx, &p_rnd };

    if (is_seq) {
        unsigned long long total = (unsigned long long)M * (unsigned long long)N;
        unsigned int gx = (unsigned int)((total + 63ull) / 64ull);
        rc = st.hip.moduleLaunchKernel(fn, gx, 1u, 1u, 64u, 1u, 1u, 0u, st.stream, args, nullptr);
    } else {
        rc = st.hip.moduleLaunchKernel(fn, (unsigned int)N, (unsigned int)M, 1u, 32u, 1u, 1u, 0u, st.stream, args, nullptr);
    }
    if (rc != HIP_SUCCESS) {
        set_error("bf16 launch failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_stop, st.stream);

    rc = st.hip.memcpyAsync(dst_y, st.d_bf16_y, y_bytes, HIP_MEMCPY_DEVICE_TO_HOST, st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("bf16 D2H failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    rc = st.hip.streamSynchronize(st.stream);
    if (rc != HIP_SUCCESS) {
        set_error("bf16 sync failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    if (y_pinned) {
        memcpy(y, st.h_out0, y_bytes);
    }
    if (ms_out) {
        float ms = 0.0f;
        *ms_out = (st.hip.eventElapsedTime(&ms, st.ev_start, st.ev_stop) == HIP_SUCCESS) ? ms : 0.0f;
    }
    return 0;
}


extern "C" __declspec(dllexport) int hv41c_bf16_seq(
    void* h, const float* x, int M, int ng, int K_x, int bf16_round, float* y, float* ms_out)
{
    return run_bf16_kernel(state().fn_bf16_seq, h, x, M, ng, K_x, bf16_round, y, ms_out, true);
}

extern "C" __declspec(dllexport) int hv41c_bf16_w32(
    void* h, const float* x, int M, int ng, int K_x, int bf16_round, float* y, float* ms_out)
{
    return run_bf16_kernel(state().fn_bf16_w32, h, x, M, ng, K_x, bf16_round, y, ms_out, false);
}

extern "C" __declspec(dllexport) int hv41c_bf16_w32t(
    void* h, const float* x, int M, int ng, int K_x, int bf16_round, float* y, float* ms_out)
{
    return run_bf16_kernel(state().fn_bf16_w32, h, x, M, ng, K_x, bf16_round, y, ms_out, false);
}

extern "C" __declspec(dllexport) int hv41c_bf16_w32m(
    void* h, const float* x, int M, int ng, int K_x, int bf16_round, float* y, float* ms_out)
{
    return run_bf16_kernel(state().fn_bf16_w32, h, x, M, ng, K_x, bf16_round, y, ms_out, false);
}

extern "C" __declspec(dllexport) void* hv41c_seq_upload_f32(const float* data, int count) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !data || count < 1) return nullptr;
    size_t bytes = (size_t)count * sizeof(float);
    void* d_ptr = nullptr;
    int rc = st.hip.malloc(&d_ptr, bytes);
    if (rc != HIP_SUCCESS) {
        set_error("hv41c_seq_upload_f32 hipMalloc(%zu) failed (%d: %s)", bytes, rc, st.hip.getErrorString(rc));
        return nullptr;
    }
    rc = st.hip.memcpy(d_ptr, data, bytes, HIP_MEMCPY_HOST_TO_DEVICE);
    if (rc != HIP_SUCCESS) {
        set_error("hv41c_seq_upload_f32 hipMemcpy H2D failed (%d: %s)", rc, st.hip.getErrorString(rc));
        st.hip.free(d_ptr);
        return nullptr;
    }
    st.resident_bytes.fetch_add(bytes, std::memory_order_relaxed);
    return d_ptr;
}

static int load_exptable_locked(State& st, const char* path) {
    if (!path || !path[0]) return -1;
    FILE* f = fopen(path, "rb");
    if (!f) {
        set_error("sattn: cannot open exp table %s", path);
        return -4;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 8) != 0) {
        fclose(f);
        set_error("sattn: bad exp table size %ld", sz);
        return -4;
    }
    std::vector<uint8_t> buf((size_t)sz);
    if (fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        set_error("sattn: short read on %s", path);
        return -4;
    }
    fclose(f);
    if (st.d_exptable) {
        st.hip.free(st.d_exptable);
        st.d_exptable = nullptr;
        st.exptable_count = 0;
    }
    int rc = st.hip.malloc(&st.d_exptable, (size_t)sz);
    if (rc != HIP_SUCCESS) {
        set_error("sattn exptable hipMalloc failed (%d)", rc);
        return -4;
    }
    rc = st.hip.memcpy(st.d_exptable, buf.data(), (size_t)sz, HIP_MEMCPY_HOST_TO_DEVICE);
    if (rc != HIP_SUCCESS) {
        st.hip.free(st.d_exptable);
        st.d_exptable = nullptr;
        set_error("sattn exptable H2D failed (%d)", rc);
        return -4;
    }
    st.exptable_count = (int)(sz / 8);
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_sattn_exptable(const char* path, int /*flag*/) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized) return -1;
    return load_exptable_locked(st, path);
}

extern "C" __declspec(dllexport) int hv41c_sattn_table_count(void) {
    return state().exptable_count;
}

extern "C" __declspec(dllexport) int hv41c_sattn(
    const float* Q, int M, int H, int d,
    const float* K, int kv_len,
    const int* idx, int topk,
    const float* sink, float scale,
    float* O, float* ms_out)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !Q || !K || !idx || !O) return -1;
    if (M < 1 || H < 1 || d < 1 || d > 512 || kv_len < 1 || topk < 1) return -2;

    size_t q_bytes = (size_t)M * (size_t)H * (size_t)d * sizeof(float);
    size_t k_bytes = (size_t)kv_len * (size_t)d * sizeof(float);
    size_t idx_count = (size_t)M * (size_t)topk;
    for (size_t i = 0; i < idx_count; ++i) {
        if ((unsigned int)(idx[i] + 1) > (unsigned int)kv_len) {
            set_error("sattn: an index lies outside [-1, %d)", kv_len);
            return -2;
        }
    }
    size_t idx_bytes = idx_count * sizeof(int);
    size_t sink_bytes = sink ? (size_t)H * sizeof(float) : 0;

    if (!ensure_dev_buf(st, st.d_sa_q, st.d_sa_q_bytes, q_bytes)) return -3;
    if (!ensure_dev_buf(st, st.d_sa_k, st.d_sa_k_bytes, k_bytes)) return -3;
    if (!ensure_dev_buf(st, st.d_sa_idx, st.d_sa_idx_bytes, idx_bytes)) return -3;
    if (sink && !ensure_dev_buf(st, st.d_sa_sink, st.d_sa_sink_bytes, sink_bytes)) return -3;
    if (!ensure_dev_buf(st, st.d_sa_o, st.d_sa_o_bytes, q_bytes)) return -3;

    if (st.hip.memcpyAsync(st.d_sa_q, Q, q_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(st.d_sa_k, K, k_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(st.d_sa_idx, idx, idx_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS ||
        (sink && st.hip.memcpyAsync(st.d_sa_sink, sink, sink_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS)) {
        set_error("sattn H2D failed");
        return -4;
    }

    if (ms_out) st.hip.eventRecord(st.ev_start, st.stream);

    void* p_Q = st.d_sa_q;
    void* p_K = st.d_sa_k;
    void* p_idx = st.d_sa_idx;
    void* p_V = st.d_sa_k;
    void* p_sink = sink ? st.d_sa_sink : nullptr;
    void* p_O = st.d_sa_o;
    void* p_tab = st.d_exptable;
    int p_tcnt = st.exptable_count;
    int p_M = M, p_H = H, p_d = d, p_kv = kv_len, p_topk = topk;
    float p_scale = scale;
    void* args[] = { &p_Q, &p_K, &p_idx, &p_V, &p_sink, &p_O, &p_tab, &p_tcnt, &p_M, &p_H, &p_d, &p_kv, &p_topk, &p_scale };

    int rc = st.hip.moduleLaunchKernel(
        st.fn_sattn,
        (unsigned int)H, (unsigned int)M, 1u,
        64u, 1u, 1u,
        0u, st.stream, args, nullptr);
    if (rc != HIP_SUCCESS) {
        set_error("sattn launch failed (%d: %s)", rc, st.hip.getErrorString(rc));
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_stop, st.stream);

    if (st.hip.memcpyAsync(O, st.d_sa_o, q_bytes, HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS ||
        st.hip.streamSynchronize(st.stream) != HIP_SUCCESS) {
        set_error("sattn D2H/sync failed");
        return -4;
    }
    if (ms_out) {
        float ms = 0.0f;
        *ms_out = (st.hip.eventElapsedTime(&ms, st.ev_start, st.ev_stop) == HIP_SUCCESS) ? ms : 0.0f;
    }
    return 0;
}

static int launch_act_quant32_dev(State& st, void* d_x, void* d_av, void* d_sv, int K) {
    void* p_x = d_x;
    void* p_av = d_av;
    void* p_sv = d_sv;
    int p_K = K;
    void* args[] = { &p_x, &p_av, &p_sv, &p_K };
    return st.hip.moduleLaunchKernel(
        st.fn_act_quant32, (unsigned int)(K >> 5), 1u, 1u, 32u, 1u, 1u, 0u, st.stream, args, nullptr);
}

extern "C" __declspec(dllexport) int hv41c_seq_shared(
    const int* av, const float* sv,
    void* d_gate_v, void* d_up_v, void* d_down_v,
    float* y, float* ms_out)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !av || !sv || !d_gate_v || !d_up_v || !d_down_v || !y) return -1;

    const auto* w_gate = static_cast<const WeightHandle*>(d_gate_v);
    const auto* w_up   = static_cast<const WeightHandle*>(d_up_v);
    const auto* w_down = static_cast<const WeightHandle*>(d_down_v);
    int K_in = w_gate->K;
    int N_mid = w_gate->N;
    int N_out = w_down->N;
    if (w_up->K != K_in || w_up->N != N_mid || w_down->K != N_mid) return -2;

    size_t av_in_bytes = (size_t)K_in * sizeof(int);
    size_t sv_in_bytes = (size_t)(K_in >> 5) * sizeof(float);
    size_t mid_bytes   = (size_t)N_mid * sizeof(float);
    size_t av_m_bytes  = (size_t)N_mid * sizeof(int);
    size_t sv_m_bytes  = (size_t)(N_mid >> 5) * sizeof(float);
    size_t out_bytes   = (size_t)N_out * sizeof(float);

    if (!ensure_dev_buf(st, st.d_a, st.d_a_bytes, av_in_bytes) ||
        !ensure_dev_buf(st, st.d_s, st.d_s_bytes, sv_in_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp0, st.d_sq_tmp0_bytes, mid_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp1, st.d_sq_tmp1_bytes, mid_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp2, st.d_sq_tmp2_bytes, mid_bytes) ||
        !ensure_dev_buf(st, st.d_sq_av, st.d_sq_av_bytes, av_m_bytes) ||
        !ensure_dev_buf(st, st.d_sq_sv, st.d_sq_sv_bytes, sv_m_bytes) ||
        !ensure_dev_buf(st, st.d_sq_out0, st.d_sq_out0_bytes, out_bytes)) {
        return -3;
    }

    const void* src_av = av;
    if (ensure_pinned_buf(st, st.h_in0, st.h_in0_bytes, av_in_bytes)) {
        memcpy(st.h_in0, av, av_in_bytes);
        src_av = st.h_in0;
    }
    const void* src_sv = sv;
    if (ensure_pinned_buf(st, st.h_in1, st.h_in1_bytes, sv_in_bytes)) {
        memcpy(st.h_in1, sv, sv_in_bytes);
        src_sv = st.h_in1;
    }
    bool y_pinned = ensure_pinned_buf(st, st.h_out0, st.h_out0_bytes, out_bytes);
    void* dst_y = y_pinned ? st.h_out0 : static_cast<void*>(y);

    if (st.hip.memcpyAsync(st.d_a, src_av, av_in_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(st.d_s, src_sv, sv_in_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS) {
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_start, st.stream);

    if (launch_fp8_m1_dev(st, w_gate, st.d_a, st.d_s, st.d_sq_tmp0) != HIP_SUCCESS ||
        launch_fp8_m1_dev(st, w_up,   st.d_a, st.d_s, st.d_sq_tmp1) != HIP_SUCCESS) {
        return -4;
    }
    {
        void* p_g = st.d_sq_tmp0; void* p_u = st.d_sq_tmp1; void* p_o = st.d_sq_tmp2; int p_n = N_mid;
        void* args[] = { &p_g, &p_u, &p_o, &p_n };
        if (st.hip.moduleLaunchKernel(st.fn_shx_silu, (unsigned int)((N_mid + 63) / 64), 1u, 1u, 64u, 1u, 1u, 0u, st.stream, args, nullptr) != HIP_SUCCESS) {
            return -4;
        }
    }
    if (launch_act_quant32_dev(st, st.d_sq_tmp2, st.d_sq_av, st.d_sq_sv, N_mid) != HIP_SUCCESS ||
        launch_fp8_m1_dev(st, w_down, st.d_sq_av, st.d_sq_sv, st.d_sq_out0) != HIP_SUCCESS) {
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_stop, st.stream);

    if (st.hip.memcpyAsync(dst_y, st.d_sq_out0, out_bytes, HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS ||
        st.hip.streamSynchronize(st.stream) != HIP_SUCCESS) {
        return -4;
    }
    if (y_pinned) {
        memcpy(y, st.h_out0, out_bytes);
    }
    if (ms_out) {
        float ms = 0.0f;
        *ms_out = (st.hip.eventElapsedTime(&ms, st.ev_start, st.ev_stop) == HIP_SUCCESS) ? ms : 0.0f;
    }
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_seq_back(
    void* h_bf16_v, const float* x,
    int ng, int K_x, int mode, int M,
    void* d_w_fp8_v, float* y, float* ms_out)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !h_bf16_v || !x || !d_w_fp8_v || !y) return -1;
    if (M != 1 || ng < 1) return -2;

    const auto* h_bf16 = static_cast<const Bf16WeightHandle*>(h_bf16_v);
    const auto* w_fp8  = static_cast<const WeightHandle*>(d_w_fp8_v);
    int N_mid = h_bf16->N;
    int K_bf  = h_bf16->K;
    int groups = (N_mid - 1) / ng + 1;
    if ((long long)groups * (long long)K_bf > (long long)K_x) return -2;
    if (w_fp8->K != N_mid) return -2;

    size_t x_bytes   = (size_t)K_x * sizeof(float);
    size_t mid_bytes = (size_t)N_mid * sizeof(float);
    size_t av_bytes  = (size_t)N_mid * sizeof(int);
    size_t sv_bytes  = (size_t)(N_mid >> 5) * sizeof(float);
    size_t out_bytes = (size_t)w_fp8->N * sizeof(float);

    if (!ensure_dev_buf(st, st.d_bf16_x, st.d_bf16_x_bytes, x_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp0, st.d_sq_tmp0_bytes, mid_bytes) ||
        !ensure_dev_buf(st, st.d_sq_av, st.d_sq_av_bytes, av_bytes) ||
        !ensure_dev_buf(st, st.d_sq_sv, st.d_sq_sv_bytes, sv_bytes) ||
        !ensure_dev_buf(st, st.d_sq_out0, st.d_sq_out0_bytes, out_bytes)) {
        return -3;
    }

    const void* src_x = x;
    if (ensure_pinned_buf(st, st.h_in0, st.h_in0_bytes, x_bytes)) {
        memcpy(st.h_in0, x, x_bytes);
        src_x = st.h_in0;
    }
    bool y_pinned = ensure_pinned_buf(st, st.h_out0, st.h_out0_bytes, out_bytes);
    void* dst_y = y_pinned ? st.h_out0 : static_cast<void*>(y);

    if (st.hip.memcpyAsync(st.d_bf16_x, src_x, x_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS) {
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_start, st.stream);

    {
        void* p_W = h_bf16->d_W; void* p_x = st.d_bf16_x; void* p_y = st.d_sq_tmp0;
        int p_M = 1, p_N = N_mid, p_K = K_bf, p_ng = ng, p_Kx = K_x, p_rnd = 1;
        void* args[] = { &p_W, &p_x, &p_y, &p_M, &p_N, &p_K, &p_ng, &p_Kx, &p_rnd };
        int rc;
        if (mode == 0) {
            unsigned int gx = (unsigned int)((N_mid + 63) / 64);
            rc = st.hip.moduleLaunchKernel(st.fn_bf16_seq, gx, 1u, 1u, 64u, 1u, 1u, 0u, st.stream, args, nullptr);
        } else {
            rc = st.hip.moduleLaunchKernel(st.fn_bf16_w32, (unsigned int)N_mid, 1u, 1u, 32u, 1u, 1u, 0u, st.stream, args, nullptr);
        }
        if (rc != HIP_SUCCESS) return -4;
    }
    if (launch_act_quant32_dev(st, st.d_sq_tmp0, st.d_sq_av, st.d_sq_sv, N_mid) != HIP_SUCCESS ||
        launch_fp8_m1_dev(st, w_fp8, st.d_sq_av, st.d_sq_sv, st.d_sq_out0) != HIP_SUCCESS) {
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_stop, st.stream);

    if (st.hip.memcpyAsync(dst_y, st.d_sq_out0, out_bytes, HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS ||
        st.hip.streamSynchronize(st.stream) != HIP_SUCCESS) {
        return -4;
    }
    if (y_pinned) {
        memcpy(y, st.h_out0, out_bytes);
    }
    if (ms_out) {
        float ms = 0.0f;
        *ms_out = (st.hip.eventElapsedTime(&ms, st.ev_start, st.ev_stop) == HIP_SUCCESS) ? ms : 0.0f;
    }
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_seq_front(
    const int* av, const float* sv,
    void** W3, int n_w3, int M,
    const float* qn_dev, float eps,
    void** W_q, int n_q,
    float** out_ptrs, float* y_qa, float** y_q_ptrs, float* ms_out)
{
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    if (!st.initialized || !av || !sv || !W3 || n_w3 < 2 || M != 1 || !qn_dev || !W_q || n_q < 1 || !out_ptrs || !y_qa || !y_q_ptrs) {
        return -1;
    }
    const auto* w_kv  = static_cast<const WeightHandle*>(W3[0]);
    const auto* w_qa  = static_cast<const WeightHandle*>(W3[1]);
    const auto* w_qb  = static_cast<const WeightHandle*>(W_q[0]);
    const auto* w_iqb = (n_q > 1 && W_q[1]) ? static_cast<const WeightHandle*>(W_q[1]) : nullptr;
    if (!w_kv || !w_qa || !w_qb || !out_ptrs[0] || !y_q_ptrs[0]) return -1;

    int K_in = w_kv->K;
    int K_qa = w_qa->N;
    if (w_qa->K != K_in || w_qb->K != K_qa) return -2;
    if (w_iqb && w_iqb->K != K_qa) return -2;

    size_t av_in_bytes = (size_t)K_in * sizeof(int);
    size_t sv_in_bytes = (size_t)(K_in >> 5) * sizeof(float);
    size_t kv_bytes    = (size_t)w_kv->N * sizeof(float);
    size_t qa_bytes    = (size_t)K_qa * sizeof(float);
    size_t av_q_bytes  = (size_t)K_qa * sizeof(int);
    size_t sv_q_bytes  = (size_t)(K_qa >> 5) * sizeof(float);
    size_t qb_bytes    = (size_t)w_qb->N * sizeof(float);
    size_t iqb_bytes   = w_iqb ? (size_t)w_iqb->N * sizeof(float) : 0;

    if (!ensure_dev_buf(st, st.d_a, st.d_a_bytes, av_in_bytes) ||
        !ensure_dev_buf(st, st.d_s, st.d_s_bytes, sv_in_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp0, st.d_sq_tmp0_bytes, kv_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp1, st.d_sq_tmp1_bytes, qa_bytes) ||
        !ensure_dev_buf(st, st.d_sq_tmp2, st.d_sq_tmp2_bytes, qa_bytes) ||
        !ensure_dev_buf(st, st.d_sq_av, st.d_sq_av_bytes, av_q_bytes) ||
        !ensure_dev_buf(st, st.d_sq_sv, st.d_sq_sv_bytes, sv_q_bytes) ||
        !ensure_dev_buf(st, st.d_sq_out0, st.d_sq_out0_bytes, qb_bytes) ||
        (w_iqb && !ensure_dev_buf(st, st.d_sq_out1, st.d_sq_out1_bytes, iqb_bytes))) {
        return -3;
    }

    const void* src_av = av;
    if (ensure_pinned_buf(st, st.h_in0, st.h_in0_bytes, av_in_bytes)) {
        memcpy(st.h_in0, av, av_in_bytes);
        src_av = st.h_in0;
    }
    const void* src_sv = sv;
    if (ensure_pinned_buf(st, st.h_in1, st.h_in1_bytes, sv_in_bytes)) {
        memcpy(st.h_in1, sv, sv_in_bytes);
        src_sv = st.h_in1;
    }
    bool kv_pinned  = ensure_pinned_buf(st, st.h_out0, st.h_out0_bytes, kv_bytes);
    bool qa_pinned  = ensure_pinned_buf(st, st.h_out1, st.h_out1_bytes, qa_bytes);
    bool qb_pinned  = ensure_pinned_buf(st, st.h_out2, st.h_out2_bytes, qb_bytes);
    bool iqb_pinned = (w_iqb && y_q_ptrs[1]) ? ensure_pinned_buf(st, st.h_out3, st.h_out3_bytes, iqb_bytes) : false;

    void* dst_kv  = kv_pinned  ? st.h_out0 : static_cast<void*>(out_ptrs[0]);
    void* dst_qa  = qa_pinned  ? st.h_out1 : static_cast<void*>(y_qa);
    void* dst_qb  = qb_pinned  ? st.h_out2 : static_cast<void*>(y_q_ptrs[0]);
    void* dst_iqb = iqb_pinned ? st.h_out3 : static_cast<void*>(y_q_ptrs[1]);

    if (st.hip.memcpyAsync(st.d_a, src_av, av_in_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(st.d_s, src_sv, sv_in_bytes, HIP_MEMCPY_HOST_TO_DEVICE, st.stream) != HIP_SUCCESS) {
        return -4;
    }
    if (ms_out) st.hip.eventRecord(st.ev_start, st.stream);

    if (launch_fp8_m1_dev(st, w_kv, st.d_a, st.d_s, st.d_sq_tmp0) != HIP_SUCCESS ||
        launch_fp8_m1_dev(st, w_qa, st.d_a, st.d_s, st.d_sq_tmp1) != HIP_SUCCESS) {
        return -4;
    }
    {
        void* p_x = st.d_sq_tmp1; void* p_w = const_cast<float*>(qn_dev); void* p_o = st.d_sq_tmp2;
        int p_K = K_qa; float p_eps = eps;
        void* args[] = { &p_x, &p_w, &p_o, &p_K, &p_eps };
        if (st.hip.moduleLaunchKernel(st.fn_e1_rmsnorm, 1u, 1u, 1u, 64u, 1u, 1u, 0u, st.stream, args, nullptr) != HIP_SUCCESS) {
            return -4;
        }
    }
    if (launch_act_quant32_dev(st, st.d_sq_tmp2, st.d_sq_av, st.d_sq_sv, K_qa) != HIP_SUCCESS ||
        launch_fp8_m1_dev(st, w_qb, st.d_sq_av, st.d_sq_sv, st.d_sq_out0) != HIP_SUCCESS) {
        return -4;
    }
    if (w_iqb && y_q_ptrs[1]) {
        if (launch_fp8_m1_dev(st, w_iqb, st.d_sq_av, st.d_sq_sv, st.d_sq_out1) != HIP_SUCCESS) {
            return -4;
        }
    }
    if (ms_out) st.hip.eventRecord(st.ev_stop, st.stream);

    if (st.hip.memcpyAsync(dst_kv,  st.d_sq_tmp0, kv_bytes,  HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(dst_qa,  st.d_sq_tmp2, qa_bytes,  HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(dst_qb,  st.d_sq_out0, qb_bytes,  HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS ||
        (w_iqb && y_q_ptrs[1] && st.hip.memcpyAsync(dst_iqb, st.d_sq_out1, iqb_bytes, HIP_MEMCPY_DEVICE_TO_HOST, st.stream) != HIP_SUCCESS) ||
        st.hip.streamSynchronize(st.stream) != HIP_SUCCESS) {
        return -4;
    }
    if (kv_pinned)  memcpy(out_ptrs[0], st.h_out0, kv_bytes);
    if (qa_pinned)  memcpy(y_qa,        st.h_out1, qa_bytes);
    if (qb_pinned)  memcpy(y_q_ptrs[0], st.h_out2, qb_bytes);
    if (iqb_pinned) memcpy(y_q_ptrs[1], st.h_out3, iqb_bytes);
    if (ms_out) {
        float ms = 0.0f;
        *ms_out = (st.hip.eventElapsedTime(&ms, st.ev_start, st.ev_stop) == HIP_SUCCESS) ? ms : 0.0f;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase 2: hv41c_corr_* (GPU plane-restore + FP4 experts, 20 exports)
// ---------------------------------------------------------------------------
struct GcMat {
    int N;
    int K;
    uint64_t b2_off;
    uint64_t pl_off;
    uint64_t field_off;
    uint64_t table_off;
    uint64_t patch_off;
    uint32_t zs;
};

struct CorrState {
    std::mutex mu;
    bool open = false;
    char err[256] = {};
    int hs_cnt = 0;
    int dev_cnt = 0;
    uint64_t hs_bytes = 0;
    uint64_t dev_bytes = 0;
    uint64_t h2d_bytes_total = 0;
    uint64_t pinned_host_bytes = 0;

    static constexpr int MAX_SLOTS = 32;
    static constexpr int MAX_RING = 24;
    static constexpr int DEC_SLOTS = 2;
    static constexpr size_t MAT_CODES_BYTES = 0x5A0000;
    static constexpr size_t MAT_SCALES_BYTES = 0x5A000;
    static constexpr size_t MAT_STRIDE = 0x5FA000;
    static constexpr size_t DEC_SLOT_BYTES = 0x11EE000;
    static constexpr size_t SLOT_DEV_BYTES = 557056;
    static constexpr size_t SLOT_HOST_IN_BYTES = 245760;

    void* h2d_stream = nullptr;
    void* comp_stream = nullptr;
    void* host_slots[MAX_SLOTS] = {};
    void* host_stage[MAX_SLOTS] = {};
    void* dev_slots[MAX_SLOTS] = {};
    void* ev_hs[MAX_SLOTS] = {};
    bool hs_pending[MAX_SLOTS] = {};
    void* ev_dev_h2d[MAX_SLOTS] = {};
    bool dev_stage_pending[MAX_SLOTS] = {};
    void* ev_dev_comp[MAX_SLOTS] = {};
    bool dev_comp_pending[MAX_SLOTS] = {};
    uint64_t fp4_off[MAX_SLOTS][6] = {};
    uint64_t fp4_sz[MAX_SLOTS][6] = {};

    void* d_lut32kb = nullptr;
    void* d_dec[DEC_SLOTS] = {};
    float* h_yg[MAX_RING] = {};
    float* h_yu[MAX_RING] = {};
    float* h_y2[MAX_RING] = {};

    void* d_ring_scratch = nullptr;
    uint8_t* h_ring_in = nullptr;
    void* d_av13[MAX_RING] = {};
    void* d_sv13[MAX_RING] = {};
    void* d_bad13[MAX_RING] = {};
    void* d_yg[MAX_RING] = {};
    void* d_yu[MAX_RING] = {};
    void* d_av2[MAX_RING] = {};
    void* d_sv2[MAX_RING] = {};
    void* d_bad2[MAX_RING] = {};
    void* d_y2[MAX_RING] = {};

    void* h_av13[MAX_RING] = {};
    void* h_sv13[MAX_RING] = {};
    void* h_bad13[MAX_RING] = {};
    void* h_av2[MAX_RING] = {};
    void* h_sv2[MAX_RING] = {};
    void* h_bad2[MAX_RING] = {};

    void* ev_dec_start[MAX_RING] = {};
    void* ev_dec_end[MAX_RING] = {};
    bool dec_timed[MAX_RING] = {};

    void* ev_w13_start[MAX_RING] = {};
    void* ev_w13_end[MAX_RING] = {};
    bool w13_timed[MAX_RING] = {};

    void* ev_w2_start[MAX_RING] = {};
    void* ev_w2_end[MAX_RING] = {};
    bool w2_timed[MAX_RING] = {};
};

static CorrState& corr_state() {
    static CorrState* cs = new CorrState();
    return *cs;
}

static void corr_set_err(CorrState& cs, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(cs.err, sizeof(cs.err), fmt, ap);
    va_end(ap);
}

static int corr_launch_fp4_dev(
    State& st, CorrState& cs,
    int dec_slot, int mat_idx,
    void* d_av, void* d_sv, void* d_bad, void* d_y,
    int M, int N, int K,
    float* h_out)
{
    size_t y_bytes = (size_t)M * (size_t)N * sizeof(float);
    uint8_t* dec_base = static_cast<uint8_t*>(cs.d_dec[dec_slot]) + (size_t)mat_idx * CorrState::MAT_STRIDE;
    void* p_codes = dec_base;
    void* p_scales = dec_base + CorrState::MAT_CODES_BYTES;
    void* p_av = d_av;
    void* p_sv = d_sv;
    void* p_bad = d_bad;
    void* p_y = d_y;
    int p_M = M, p_N = N, p_K = K;
    void* args[] = { &p_codes, &p_scales, &p_av, &p_sv, &p_bad, &p_y, &p_M, &p_N, &p_K };
    unsigned int gx = (unsigned int)((N + 63) / 64);
    if (st.hip.moduleLaunchKernel(st.fn_fp4_linear, gx, (unsigned int)M, 1u, 64u, 1u, 1u, 0u, cs.comp_stream, args, nullptr) != HIP_SUCCESS) {
        corr_set_err(cs, "hv41_fp4_linear_kernel launch failed");
        return -1;
    }
    if (st.hip.memcpyAsync(h_out, d_y, y_bytes, HIP_MEMCPY_DEVICE_TO_HOST, cs.comp_stream) != HIP_SUCCESS) {
        corr_set_err(cs, "corr_launch_fp4 D2H failed");
        return -1;
    }
    return 0;
}

extern "C" __declspec(dllexport) const char* hv41c_corr_error() {
    return corr_state().err;
}

extern "C" __declspec(dllexport) int hv41c_corr_version() {
    return 18;
}

extern "C" __declspec(dllexport) uint64_t hv41c_corr_h2d_bytes() {
    return corr_state().h2d_bytes_total;
}

extern "C" __declspec(dllexport) int hv41c_corr_open(
    int hs_cnt, uint64_t hs_bytes,
    int dev_cnt, uint64_t dev_bytes,
    const uint8_t* tables_32kb, int floor_mib, int64_t* out_info)
{
    State& st = state();
    CorrState& cs = corr_state();
    std::lock_guard<std::mutex> lock(cs.mu);
    if (!st.initialized) {
        corr_set_err(cs, "hv41c_init not called");
        return -1;
    }
    if (cs.open) return 0;
    if (hs_cnt < 2 || hs_cnt > CorrState::MAX_SLOTS || dev_cnt < 2 || dev_cnt > CorrState::MAX_SLOTS || !tables_32kb) {
        corr_set_err(cs, "bad slot counts");
        return -1;
    }

    size_t vram_free = 0, vram_total = 0;
    st.hip.memGetInfo(&vram_free, &vram_total);
    uint64_t fixed_bytes = (uint64_t)CorrState::DEC_SLOTS * CorrState::DEC_SLOT_BYTES
                         + (uint64_t)CorrState::MAX_RING * CorrState::SLOT_DEV_BYTES
                         + 32768ull;
    uint64_t floor_bytes = (floor_mib > 0) ? ((uint64_t)floor_mib << 20) : 0ull;
    while (dev_cnt >= 2) {
        if ((uint64_t)vram_free >= floor_bytes + fixed_bytes + (uint64_t)dev_cnt * dev_bytes) break;
        --dev_cnt;
    }
    if (dev_cnt < 2) {
        corr_set_err(cs, "VRAM free %zu MiB does not fit correction ring", vram_free >> 20);
        return -2;
    }

    cs.hs_cnt = hs_cnt;
    cs.dev_cnt = dev_cnt;
    cs.hs_bytes = hs_bytes;
    cs.dev_bytes = dev_bytes;
    cs.pinned_host_bytes = 0;

    if (st.hip.streamCreate(&cs.h2d_stream) != HIP_SUCCESS ||
        st.hip.streamCreate(&cs.comp_stream) != HIP_SUCCESS) {
        corr_set_err(cs, "streamCreate failed");
        return -1;
    }

    for (int i = 0; i < hs_cnt; ++i) {
        if (st.hip.hostMalloc(&cs.host_slots[i], (size_t)hs_bytes, 0u) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_hs[i]) != HIP_SUCCESS) {
            corr_set_err(cs, "host slot alloc failed");
            return -1;
        }
        cs.pinned_host_bytes += hs_bytes;
    }
    for (int i = 0; i < dev_cnt; ++i) {
        if (st.hip.hostMalloc(&cs.host_stage[i], (size_t)dev_bytes, 0u) != HIP_SUCCESS ||
            st.hip.malloc(&cs.dev_slots[i], (size_t)dev_bytes) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_dev_h2d[i]) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_dev_comp[i]) != HIP_SUCCESS) {
            corr_set_err(cs, "device slot alloc failed");
            return -1;
        }
        cs.pinned_host_bytes += dev_bytes;
    }
    if (st.hip.malloc(&cs.d_lut32kb, 32768) != HIP_SUCCESS ||
        st.hip.memcpy(cs.d_lut32kb, tables_32kb, 32768, HIP_MEMCPY_HOST_TO_DEVICE) != HIP_SUCCESS) {
        corr_set_err(cs, "lut32kb alloc/copy failed");
        return -1;
    }
    if (st.hip.malloc(&cs.d_ring_scratch, (size_t)CorrState::MAX_RING * CorrState::SLOT_DEV_BYTES) != HIP_SUCCESS ||
        st.hip.hostMalloc((void**)&cs.h_ring_in, (size_t)CorrState::MAX_RING * CorrState::SLOT_HOST_IN_BYTES, 0u) != HIP_SUCCESS) {
        corr_set_err(cs, "scratch ring alloc failed");
        return -1;
    }
    for (int i = 0; i < CorrState::DEC_SLOTS; ++i) {
        if (st.hip.malloc(&cs.d_dec[i], CorrState::DEC_SLOT_BYTES) != HIP_SUCCESS) {
            corr_set_err(cs, "dec slot alloc failed");
            return -1;
        }
    }
    for (int i = 0; i < CorrState::MAX_RING; ++i) {
        uint8_t* db = static_cast<uint8_t*>(cs.d_ring_scratch) + (size_t)i * CorrState::SLOT_DEV_BYTES;
        cs.d_av13[i]  = db + 0;
        cs.d_sv13[i]  = db + 163840;
        cs.d_bad13[i] = db + 168960;
        cs.d_yg[i]    = db + 169216;
        cs.d_yu[i]    = db + 242944;
        cs.d_av2[i]   = db + 316672;
        cs.d_sv2[i]   = db + 390400;
        cs.d_bad2[i]  = db + 392960;
        cs.d_y2[i]    = db + 393216;

        uint8_t* hb = cs.h_ring_in + (size_t)i * CorrState::SLOT_HOST_IN_BYTES;
        cs.h_av13[i]  = hb + 0;
        cs.h_sv13[i]  = hb + 163840;
        cs.h_bad13[i] = hb + 168960;
        cs.h_av2[i]   = hb + 169216;
        cs.h_sv2[i]   = hb + 242944;
        cs.h_bad2[i]  = hb + 245504;

        if (st.hip.hostMalloc((void**)&cs.h_yg[i], 8u * 2304u * sizeof(float), 0u) != HIP_SUCCESS ||
            st.hip.hostMalloc((void**)&cs.h_yu[i], 8u * 2304u * sizeof(float), 0u) != HIP_SUCCESS ||
            st.hip.hostMalloc((void**)&cs.h_y2[i], 8u * 5120u * sizeof(float), 0u) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_dec_start[i]) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_dec_end[i]) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_w13_start[i]) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_w13_end[i]) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_w2_start[i]) != HIP_SUCCESS ||
            st.hip.eventCreate(&cs.ev_w2_end[i]) != HIP_SUCCESS) {
            corr_set_err(cs, "ring alloc failed");
            return -1;
        }
    }

    size_t vram_after = 0;
    st.hip.memGetInfo(&vram_after, &vram_total);
    uint64_t used = (vram_free > vram_after) ? (vram_free - vram_after) : (fixed_bytes + (uint64_t)dev_cnt * dev_bytes);
    if (out_info) {
        out_info[0] = (int64_t)used;
        out_info[1] = (int64_t)(vram_after >> 20);
        out_info[2] = (int64_t)cs.pinned_host_bytes;
        out_info[3] = (int64_t)dev_cnt;
    }
    cs.open = true;
    return 0;
}

extern "C" __declspec(dllexport) uint8_t* hv41c_corr_host(int hs) {
    CorrState& cs = corr_state();
    if (!cs.open || hs < 0 || hs >= cs.hs_cnt) return nullptr;
    return static_cast<uint8_t*>(cs.host_slots[hs]);
}

extern "C" __declspec(dllexport) void hv41c_corr_hs_wait(int hs) {
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open || hs < 0 || hs >= cs.hs_cnt) return;
    if (cs.hs_pending[hs]) {
        st.hip.eventSynchronize(cs.ev_hs[hs]);
        cs.hs_pending[hs] = false;
    }
}

extern "C" __declspec(dllexport) int hv41c_corr_load_gc(
    int dev_slot, int hs, const uint8_t** ptrs, const uint64_t* sizes, uint64_t total_bytes)
{
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open) {
        corr_set_err(cs, "load_gc not open");
        return -1;
    }
    std::lock_guard<std::mutex> lock(cs.mu);
    if (dev_slot < 0 || dev_slot >= cs.dev_cnt || hs < 0 || hs >= cs.hs_cnt || !ptrs || !sizes) {
        corr_set_err(cs, "load_gc bad args dev=%d/%d hs=%d/%d", dev_slot, cs.dev_cnt, hs, cs.hs_cnt);
        return -1;
    }

    uint64_t pl_total = sizes[0] + sizes[1] + sizes[2];
    if (pl_total + total_bytes > cs.dev_bytes || total_bytes > cs.hs_bytes) {
        corr_set_err(cs, "load_gc overflow");
        return -3;
    }
    if (cs.dev_stage_pending[dev_slot]) {
        st.hip.eventSynchronize(cs.ev_dev_h2d[dev_slot]);
        cs.dev_stage_pending[dev_slot] = false;
    }
    if (cs.hs_pending[hs]) {
        st.hip.eventSynchronize(cs.ev_hs[hs]);
        cs.hs_pending[hs] = false;
    }

    uint8_t* stg = static_cast<uint8_t*>(cs.host_stage[dev_slot]);
    memcpy(stg, ptrs[0], (size_t)sizes[0]);
    memcpy(stg + sizes[0], ptrs[1], (size_t)sizes[1]);
    memcpy(stg + sizes[0] + sizes[1], ptrs[2], (size_t)sizes[2]);

    if (cs.dev_comp_pending[dev_slot]) {
        st.hip.streamWaitEvent(cs.h2d_stream, cs.ev_dev_comp[dev_slot], 0u);
    }
    uint8_t* dst = static_cast<uint8_t*>(cs.dev_slots[dev_slot]);
    if (st.hip.memcpyAsync(dst, stg, (size_t)pl_total, HIP_MEMCPY_HOST_TO_DEVICE, cs.h2d_stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(dst + pl_total, cs.host_slots[hs], (size_t)total_bytes, HIP_MEMCPY_HOST_TO_DEVICE, cs.h2d_stream) != HIP_SUCCESS) {
        corr_set_err(cs, "load_gc H2D failed");
        return -1;
    }
    st.hip.eventRecord(cs.ev_dev_h2d[dev_slot], cs.h2d_stream);
    cs.dev_stage_pending[dev_slot] = true;
    st.hip.eventRecord(cs.ev_hs[hs], cs.h2d_stream);
    cs.hs_pending[hs] = true;
    cs.h2d_bytes_total += pl_total + total_bytes;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_load_fp4(
    int dev_slot, const uint8_t** ptrs, const uint64_t* sizes)
{
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open) {
        corr_set_err(cs, "load_fp4 not open");
        return -1;
    }
    std::lock_guard<std::mutex> lock(cs.mu);
    if (dev_slot < 0 || dev_slot >= cs.dev_cnt || !ptrs || !sizes) {
        corr_set_err(cs, "load_fp4 bad args dev=%d/%d", dev_slot, cs.dev_cnt);
        return -1;
    }

    uint64_t off = 0;
    for (int i = 0; i < 6; ++i) {
        cs.fp4_off[dev_slot][i] = off;
        cs.fp4_sz[dev_slot][i] = sizes[i];
        off += (sizes[i] + 15ull) & ~15ull;
    }
    if (off > cs.dev_bytes) {
        corr_set_err(cs, "load_fp4 overflow");
        return -3;
    }
    if (cs.dev_stage_pending[dev_slot]) {
        st.hip.eventSynchronize(cs.ev_dev_h2d[dev_slot]);
        cs.dev_stage_pending[dev_slot] = false;
    }
    uint8_t* hbuf = static_cast<uint8_t*>(cs.host_stage[dev_slot]);
    for (int i = 0; i < 6; ++i) {
        memcpy(hbuf + cs.fp4_off[dev_slot][i], ptrs[i], (size_t)sizes[i]);
    }
    if (cs.dev_comp_pending[dev_slot]) {
        st.hip.streamWaitEvent(cs.h2d_stream, cs.ev_dev_comp[dev_slot], 0u);
    }
    if (st.hip.memcpyAsync(cs.dev_slots[dev_slot], hbuf, (size_t)off, HIP_MEMCPY_HOST_TO_DEVICE, cs.h2d_stream) != HIP_SUCCESS) {
        corr_set_err(cs, "load_fp4 H2D failed");
        return -1;
    }
    st.hip.eventRecord(cs.ev_dev_h2d[dev_slot], cs.h2d_stream);
    cs.dev_stage_pending[dev_slot] = true;
    cs.h2d_bytes_total += off;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_decode(
    int dev_slot, int ring_slot, const GcMat* mats, int ev_slot)
{
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open) {
        corr_set_err(cs, "corr_decode not open");
        return -1;
    }
    std::lock_guard<std::mutex> lock(cs.mu);
    if (dev_slot < 0 || dev_slot >= cs.dev_cnt || ring_slot < 0 || ring_slot >= CorrState::DEC_SLOTS ||
        ev_slot < 0 || ev_slot >= CorrState::MAX_RING || !mats) {
        corr_set_err(cs, "corr_decode bad args dev=%d dec=%d ev=%d", dev_slot, ring_slot, ev_slot);
        return -1;
    }

    if (cs.dev_stage_pending[dev_slot]) {
        st.hip.streamWaitEvent(cs.comp_stream, cs.ev_dev_h2d[dev_slot], 0u);
    }
    st.hip.eventRecord(cs.ev_dec_start[ev_slot], cs.comp_stream);

    const uint8_t* dev_base = static_cast<const uint8_t*>(cs.dev_slots[dev_slot]);
    uint8_t* dec_base = static_cast<uint8_t*>(cs.d_dec[ring_slot]);

    for (int i = 0; i < 3; ++i) {
        const GcMat& m = mats[i];
        void* p_field = const_cast<uint8_t*>(dev_base + m.b2_off);
        const uint8_t* pl = dev_base + m.pl_off;
        void* p_plane1 = const_cast<uint8_t*>(pl + m.field_off);
        void* p_pidx   = const_cast<uint8_t*>(pl + m.table_off);
        void* p_pbytes = const_cast<uint8_t*>(pl + m.patch_off);
        void* p_lut    = cs.d_lut32kb;
        int p_N = m.N, p_K = m.K, p_zs = (int)m.zs;
        uint8_t* out_c = dec_base + (size_t)i * CorrState::MAT_STRIDE;
        void* p_out_c = out_c;
        void* p_out_s = out_c + ((size_t)m.N * (size_t)m.K >> 1);
        void* args[] = { &p_field, &p_plane1, &p_pidx, &p_pbytes, &p_lut, &p_N, &p_K, &p_zs, &p_out_c, &p_out_s };
        unsigned int gx = (unsigned int)(m.N >> 3);
        if (st.hip.moduleLaunchKernel(st.fn_plane_restore, gx, 1u, 1u, 256u, 1u, 1u, 0u, cs.comp_stream, args, nullptr) != HIP_SUCCESS) {
            corr_set_err(cs, "hv41_plane_restore_kernel launch failed");
            return -1;
        }
    }

    st.hip.eventRecord(cs.ev_dec_end[ev_slot], cs.comp_stream);
    st.hip.eventRecord(cs.ev_dev_comp[dev_slot], cs.comp_stream);
    cs.dev_comp_pending[dev_slot] = true;
    cs.dec_timed[ev_slot] = true;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_decode_fp4(
    int dev_slot, int ring_slot, int ev_slot)
{
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open) {
        corr_set_err(cs, "decode_fp4 not open");
        return -1;
    }
    std::lock_guard<std::mutex> lock(cs.mu);
    if (dev_slot < 0 || dev_slot >= cs.dev_cnt || ring_slot < 0 || ring_slot >= CorrState::DEC_SLOTS ||
        ev_slot < 0 || ev_slot >= CorrState::MAX_RING) {
        corr_set_err(cs, "decode_fp4 bad args dev=%d dec=%d ev=%d", dev_slot, ring_slot, ev_slot);
        return -1;
    }

    if (cs.dev_stage_pending[dev_slot]) {
        st.hip.streamWaitEvent(cs.comp_stream, cs.ev_dev_h2d[dev_slot], 0u);
    }
    st.hip.eventRecord(cs.ev_dec_start[ev_slot], cs.comp_stream);

    const uint8_t* src_base = static_cast<const uint8_t*>(cs.dev_slots[dev_slot]);
    uint8_t* dec_base = static_cast<uint8_t*>(cs.d_dec[ring_slot]);
    for (int i = 0; i < 3; ++i) {
        uint8_t* dst = dec_base + (size_t)i * CorrState::MAT_STRIDE;
        if (st.hip.memcpyAsync(dst, src_base + cs.fp4_off[dev_slot][2 * i], (size_t)cs.fp4_sz[dev_slot][2 * i], HIP_MEMCPY_DEVICE_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS ||
            st.hip.memcpyAsync(dst + CorrState::MAT_CODES_BYTES, src_base + cs.fp4_off[dev_slot][2 * i + 1], (size_t)cs.fp4_sz[dev_slot][2 * i + 1], HIP_MEMCPY_DEVICE_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS) {
            corr_set_err(cs, "decode_fp4 D2D failed");
            return -1;
        }
    }

    st.hip.eventRecord(cs.ev_dec_end[ev_slot], cs.comp_stream);
    st.hip.eventRecord(cs.ev_dev_comp[dev_slot], cs.comp_stream);
    cs.dev_comp_pending[dev_slot] = true;
    cs.dec_timed[ev_slot] = true;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_w13(
    int dec_slot, int out_slot, const int* av, const float* sv, const uint8_t* bad, int M)
{
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open || M < 1 || M > 8 || dec_slot < 0 || dec_slot >= CorrState::DEC_SLOTS ||
        out_slot < 0 || out_slot >= CorrState::MAX_RING || !av || !sv || !bad) {
        corr_set_err(cs, "corr_w13 bad args dec=%d out=%d M=%d", dec_slot, out_slot, M);
        return -1;
    }
    std::lock_guard<std::mutex> lock(cs.mu);

    size_t av_bytes = (size_t)M * 5120u * sizeof(int);
    size_t sv_bytes = (size_t)M * 160u * sizeof(float);
    memcpy(cs.h_av13[out_slot], av, av_bytes);
    memcpy(cs.h_sv13[out_slot], sv, sv_bytes);
    memcpy(cs.h_bad13[out_slot], bad, (size_t)M);

    if (st.hip.memcpyAsync(cs.d_av13[out_slot], cs.h_av13[out_slot], av_bytes, HIP_MEMCPY_HOST_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(cs.d_sv13[out_slot], cs.h_sv13[out_slot], sv_bytes, HIP_MEMCPY_HOST_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(cs.d_bad13[out_slot], cs.h_bad13[out_slot], (size_t)M, HIP_MEMCPY_HOST_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS) {
        corr_set_err(cs, "corr_w13 H2D failed");
        return -1;
    }

    st.hip.eventRecord(cs.ev_w13_start[out_slot], cs.comp_stream);
    if (corr_launch_fp4_dev(st, cs, dec_slot, 0, cs.d_av13[out_slot], cs.d_sv13[out_slot], cs.d_bad13[out_slot], cs.d_yg[out_slot], M, 2304, 5120, cs.h_yg[out_slot]) != 0 ||
        corr_launch_fp4_dev(st, cs, dec_slot, 2, cs.d_av13[out_slot], cs.d_sv13[out_slot], cs.d_bad13[out_slot], cs.d_yu[out_slot], M, 2304, 5120, cs.h_yu[out_slot]) != 0) {
        return -1;
    }
    st.hip.eventRecord(cs.ev_w13_end[out_slot], cs.comp_stream);
    cs.w13_timed[out_slot] = true;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_w2(
    int dec_slot, int out_slot, const int* av, const float* sv, const uint8_t* bad, int M)
{
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open || M < 1 || M > 8 || dec_slot < 0 || dec_slot >= CorrState::DEC_SLOTS ||
        out_slot < 0 || out_slot >= CorrState::MAX_RING || !av || !sv || !bad) {
        corr_set_err(cs, "corr_w2 bad args dec=%d out=%d M=%d", dec_slot, out_slot, M);
        return -1;
    }
    std::lock_guard<std::mutex> lock(cs.mu);

    size_t av_bytes = (size_t)M * 2304u * sizeof(int);
    size_t sv_bytes = (size_t)M * 72u * sizeof(float);
    memcpy(cs.h_av2[out_slot], av, av_bytes);
    memcpy(cs.h_sv2[out_slot], sv, sv_bytes);
    memcpy(cs.h_bad2[out_slot], bad, (size_t)M);

    if (st.hip.memcpyAsync(cs.d_av2[out_slot], cs.h_av2[out_slot], av_bytes, HIP_MEMCPY_HOST_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(cs.d_sv2[out_slot], cs.h_sv2[out_slot], sv_bytes, HIP_MEMCPY_HOST_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS ||
        st.hip.memcpyAsync(cs.d_bad2[out_slot], cs.h_bad2[out_slot], (size_t)M, HIP_MEMCPY_HOST_TO_DEVICE, cs.comp_stream) != HIP_SUCCESS) {
        corr_set_err(cs, "corr_w2 H2D failed");
        return -1;
    }

    st.hip.eventRecord(cs.ev_w2_start[out_slot], cs.comp_stream);
    if (corr_launch_fp4_dev(st, cs, dec_slot, 1, cs.d_av2[out_slot], cs.d_sv2[out_slot], cs.d_bad2[out_slot], cs.d_y2[out_slot], M, 5120, 2304, cs.h_y2[out_slot]) != 0) {
        return -1;
    }
    st.hip.eventRecord(cs.ev_w2_end[out_slot], cs.comp_stream);
    cs.w2_timed[out_slot] = true;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_wait13(int out_slot) {
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open || out_slot < 0 || out_slot >= CorrState::MAX_RING) return -1;
    return (st.hip.eventSynchronize(cs.ev_w13_end[out_slot]) == HIP_SUCCESS) ? 0 : -1;
}

extern "C" __declspec(dllexport) int hv41c_corr_wait2(int out_slot) {
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open || out_slot < 0 || out_slot >= CorrState::MAX_RING) return -1;
    return (st.hip.eventSynchronize(cs.ev_w2_end[out_slot]) == HIP_SUCCESS) ? 0 : -1;
}

extern "C" __declspec(dllexport) const float* hv41c_corr_yg(int out_slot) {
    CorrState& cs = corr_state();
    if (!cs.open || out_slot < 0 || out_slot >= CorrState::MAX_RING) return nullptr;
    return cs.h_yg[out_slot];
}

extern "C" __declspec(dllexport) const float* hv41c_corr_yu(int out_slot) {
    CorrState& cs = corr_state();
    if (!cs.open || out_slot < 0 || out_slot >= CorrState::MAX_RING) return nullptr;
    return cs.h_yu[out_slot];
}

extern "C" __declspec(dllexport) const float* hv41c_corr_y2(int out_slot) {
    CorrState& cs = corr_state();
    if (!cs.open || out_slot < 0 || out_slot >= CorrState::MAX_RING) return nullptr;
    return cs.h_y2[out_slot];
}

extern "C" __declspec(dllexport) int hv41c_corr_sync() {
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open) return 0;
    if (cs.h2d_stream && st.hip.streamSynchronize(cs.h2d_stream) != HIP_SUCCESS) return -1;
    if (cs.comp_stream && st.hip.streamSynchronize(cs.comp_stream) != HIP_SUCCESS) return -1;
    return 0;
}

extern "C" __declspec(dllexport) int hv41c_corr_read_decoded(int ring_slot, uint8_t* dst) {
    State& st = state();
    CorrState& cs = corr_state();
    if (!cs.open || ring_slot < 0 || ring_slot >= CorrState::DEC_SLOTS || !dst) {
        corr_set_err(cs, "read_decoded bad args dec=%d", ring_slot);
        return -1;
    }
    if (st.hip.streamSynchronize(cs.comp_stream) != HIP_SUCCESS) return -1;
    return (st.hip.memcpy(dst, cs.d_dec[ring_slot], CorrState::DEC_SLOT_BYTES, HIP_MEMCPY_DEVICE_TO_HOST) == HIP_SUCCESS) ? 0 : -1;
}

extern "C" __declspec(dllexport) void hv41c_corr_take_times(double* out3) {
    State& st = state();
    CorrState& cs = corr_state();
    if (!out3) return;
    std::lock_guard<std::mutex> lock(cs.mu);
    double t0 = 0.0, t1 = 0.0, t2 = 0.0;
    for (int i = 0; i < CorrState::MAX_RING; ++i) {
        if (cs.dec_timed[i]) {
            float ms = 0.0f;
            if (st.hip.eventElapsedTime(&ms, cs.ev_dec_start[i], cs.ev_dec_end[i]) == HIP_SUCCESS) {
                t0 += (double)ms;
                cs.dec_timed[i] = false;
            }
        }
        if (cs.w13_timed[i]) {
            float ms = 0.0f;
            if (st.hip.eventElapsedTime(&ms, cs.ev_w13_start[i], cs.ev_w13_end[i]) == HIP_SUCCESS) {
                t1 += (double)ms;
                cs.w13_timed[i] = false;
            }
        }
        if (cs.w2_timed[i]) {
            float ms = 0.0f;
            if (st.hip.eventElapsedTime(&ms, cs.ev_w2_start[i], cs.ev_w2_end[i]) == HIP_SUCCESS) {
                t2 += (double)ms;
                cs.w2_timed[i] = false;
            }
        }
    }
    out3[0] += t0;
    out3[1] += t1;
    out3[2] += t2;
}

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved) {
    (void)lpvReserved;
    if (fdwReason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinstDLL);
    }
    return TRUE;
}
