// Reconstructed minimal llama.cpp header matching PROMETHEVS_0.2 deepseek41.cpp.obj (.debug$T) and llama.lib
#pragma once

#include "ggml.h"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;

struct llama_vocab;
struct llama_model;
struct llama_context;
struct llama_memory_i;
typedef struct llama_memory_i * llama_memory_t;

struct llama_model_params {
    void * devices;
    const void * tensor_buft_overrides;
    int32_t n_gpu_layers;
    int32_t split_mode;
    int32_t load_mode;
    int32_t lazy_mode;
    int32_t main_gpu;
    int32_t _pad0;
    const float * tensor_split;
    bool (*progress_callback)(float progress, void * user_data);
    void * progress_callback_user_data;
    const void * kv_overrides;
    bool vocab_only;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
    bool load_mtp;
    uint8_t _pad1[2];
};

static_assert(sizeof(struct llama_model_params) == 0x50, "llama_model_params size must match deepseek41.cpp.obj (80 bytes)");
static_assert(offsetof(struct llama_model_params, n_gpu_layers) == 0x10, "llama_model_params::n_gpu_layers offset mismatch");
static_assert(offsetof(struct llama_model_params, load_mtp) == 0x4d, "llama_model_params::load_mtp offset mismatch");

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    uint32_t n_outputs_max_per_seq;
    int32_t n_threads;
    int32_t n_threads_batch;
    int32_t ctx_type;
    int32_t rope_scaling_type;
    int32_t pooling_type;
    int32_t attention_type;
    int32_t flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    int32_t type_k;
    int32_t type_v;
    bool (*abort_callback)(void * data);
    void * abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    uint8_t _pad0[2];
    void * samplers;
    size_t n_samplers;
    void * ctx_other;
};

static_assert(sizeof(struct llama_context_params) == 0xA0, "llama_context_params size must match deepseek41.cpp.obj (160 bytes)");
static_assert(offsetof(struct llama_context_params, cb_eval) == 0x58, "llama_context_params::cb_eval offset mismatch");
static_assert(offsetof(struct llama_context_params, cb_eval_user_data) == 0x60, "llama_context_params::cb_eval_user_data offset mismatch");
static_assert(offsetof(struct llama_context_params, ctx_other) == 0x98, "llama_context_params::ctx_other offset mismatch");

struct llama_batch {
    int32_t n_tokens;
    int32_t _pad0;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
};

static_assert(sizeof(struct llama_batch) == 0x38, "llama_batch size must match deepseek41.cpp.obj (56 bytes)");
static_assert(offsetof(struct llama_batch, logits) == 0x30, "llama_batch::logits offset mismatch");

void llama_backend_init(void);

struct llama_model_params llama_model_default_params(void);
struct llama_context_params llama_context_default_params(void);

struct llama_model * llama_model_load_from_file(const char * path_model, struct llama_model_params params);
const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);

int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);
llama_token llama_vocab_eos(const struct llama_vocab * vocab);

int32_t llama_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);

int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
    llama_token token,
    char * buf,
    int32_t length,
    int32_t lstrip,
    bool special);

struct llama_context * llama_init_from_model(struct llama_model * model, struct llama_context_params params);
void llama_free(struct llama_context * ctx);

void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

struct llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
void llama_batch_free(struct llama_batch batch);

int32_t llama_decode(struct llama_context * ctx, struct llama_batch batch);
float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

llama_memory_t llama_get_memory(const struct llama_context * ctx);
void llama_memory_clear(llama_memory_t mem, bool data);
llama_pos llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id);

size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id);
size_t llama_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id);
size_t llama_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id);

size_t llama_state_seq_save_file(
    struct llama_context * ctx,
    const char * filepath,
    llama_seq_id seq_id,
    const llama_token * tokens,
    size_t n_token_count);

size_t llama_state_seq_load_file(
    struct llama_context * ctx,
    const char * filepath,
    llama_seq_id dest_seq_id,
    llama_token * tokens_out,
    size_t n_token_capacity,
    size_t * n_token_count_out);

#ifdef __cplusplus
}
#endif
