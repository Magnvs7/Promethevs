// Reconstructed minimal ggml header matching PROMETHEVS_0.2 deepseek41.cpp.obj (.debug$T)
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_DUP = 1,
    GGML_OP_ADD = 2,
    GGML_OP_MUL_MAT = 29,
    GGML_OP_MUL_MAT_ID = 30,
    GGML_OP_UNARY = 91,
    GGML_OP_CUSTOM = 95,
    GGML_OP_GLU = 100,
    GGML_OP_COUNT = 101
};

struct ggml_tensor {
    int32_t type;
    int32_t _pad0;
    void * buffer;
    int64_t ne[4];
    size_t nb[4];
    enum ggml_op op;
    int32_t op_params[16];
    int32_t flags;
    struct ggml_tensor * src[10];
    struct ggml_tensor * view_src;
    size_t view_offs;
    void * data;
    char name[64];
    void * extra;
    char padding[8];
};

static_assert(sizeof(struct ggml_tensor) == 0x150, "ggml_tensor size must match deepseek41.cpp.obj (336 bytes)");
static_assert(offsetof(struct ggml_tensor, op) == 0x50, "ggml_tensor::op offset mismatch");
static_assert(offsetof(struct ggml_tensor, op_params) == 0x54, "ggml_tensor::op_params offset mismatch");
static_assert(offsetof(struct ggml_tensor, src) == 0x98, "ggml_tensor::src offset mismatch");
static_assert(offsetof(struct ggml_tensor, data) == 0xf8, "ggml_tensor::data offset mismatch");
static_assert(offsetof(struct ggml_tensor, name) == 0x100, "ggml_tensor::name offset mismatch");

struct ggml_cgraph;

typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);

const char * ggml_op_name(enum ggml_op op);
const char * ggml_op_desc(const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif
