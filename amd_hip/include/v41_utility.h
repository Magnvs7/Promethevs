// Compatibility shims & V41Utility interface for linking PROMETHEVS_0.2 under VS 2022 (MSVC 14.44)
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

struct V41UtilityChoice {
    int width = 1;
    bool explore = false;
    double useful = 0.0;
    double seconds = 0.0;
};

struct V41Utility {
    V41UtilityChoice choose(int source, int supplied, double setup_seconds) {
        (void)source;
        (void)setup_seconds;
        V41UtilityChoice c;
        c.width = std::max(1, supplied);
        return c;
    }

    void observe(int source, int n_lanes, int accepted, bool ended, double work_seconds) {
        (void)source;
        (void)n_lanes;
        (void)accepted;
        (void)ended;
        (void)work_seconds;
    }
};

extern "C" {

void __cdecl __std_rotate(void * first, void * mid, void * last) noexcept {
    uint8_t * f = static_cast<uint8_t *>(first);
    uint8_t * m = static_cast<uint8_t *>(mid);
    uint8_t * l = static_cast<uint8_t *>(last);
    auto rev = [](uint8_t * a, uint8_t * b) {
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

void * __cdecl __std_unique_4(void * first, void * last) noexcept {
    uint32_t * f = static_cast<uint32_t *>(first);
    uint32_t * l = static_cast<uint32_t *>(last);
    if (f == l) return l;
    uint32_t * dest = f;
    while (++f != l) {
        if (*dest != *f && ++dest != f) *dest = *f;
    }
    return ++dest;
}

void * __cdecl __std_unique_8(void * first, void * last) noexcept {
    uint64_t * f = static_cast<uint64_t *>(first);
    uint64_t * l = static_cast<uint64_t *>(last);
    if (f == l) return l;
    uint64_t * dest = f;
    while (++f != l) {
        if (*dest != *f && ++dest != f) *dest = *f;
    }
    return ++dest;
}

const void * __cdecl __std_find_not_ch_1(const void * first, const void * last, uint8_t val) noexcept {
    const uint8_t * p = static_cast<const uint8_t *>(first);
    const uint8_t * e = static_cast<const uint8_t *>(last);
    for (; p != e; ++p) {
        if (*p != val) return p;
    }
    return e;
}

size_t __cdecl __std_regex_transform_primary_char(
    char * first1, char * last1, const char * first2, const char * last2, const void * ploc) noexcept {
    (void)first1; (void)last1; (void)first2; (void)last2; (void)ploc;
    return ~size_t{ 0 };
}

size_t __cdecl __std_regex_transform_primary_wchar_t(
    wchar_t * first1, wchar_t * last1, const wchar_t * first2, const wchar_t * last2, const void * ploc) noexcept {
    (void)first1; (void)last1; (void)first2; (void)last2; (void)ploc;
    return ~size_t{ 0 };
}

} // extern "C"
