#include "ggml-decoder.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Local execution-cache key. A match still needs the ModelParams compatibility
// check; this key alone does not identify weights or a compiled model.
struct graph_key {
    int n_nodes;
    std::string first_node_name;
    std::string last_node_name;
    std::vector<std::string> input_src_names;

    graph_key(const ggml_cgraph * cgraph) : n_nodes(cgraph->n_nodes) {
        if (n_nodes > 0) {
            first_node_name = cgraph->nodes[0]->name;
            last_node_name = cgraph->nodes[n_nodes - 1]->name;
        }

        std::unordered_map<const ggml_tensor *, std::string> names;
        auto get_input_key_name = [&names](const ggml_cgraph * graph, const ggml_tensor * tensor) {
            auto it = names.find(tensor);
            if (it == names.end()) {
                it = names.emplace(tensor, GgmlOvDecoder::get_tensor_name(graph, tensor)).first;
            }
            return it->second;
        };

        std::vector<std::string> node_names;
        node_names.reserve(cgraph->n_nodes);
        for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
            node_names.emplace_back(cgraph->nodes[node_idx]->name);
        }

        for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
            const ggml_tensor * node = cgraph->nodes[node_idx];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; src_idx++) {
                const ggml_tensor * src = node->src[src_idx];
                if (src == nullptr || src->name[0] == '\0') {
                    continue;
                }

                const std::string src_name = get_input_key_name(cgraph, src);
                if (std::find(node_names.begin(), node_names.end(), src_name) != node_names.end()) {
                    continue;
                }
                if (src_name.find("weight") != std::string::npos) {
                    continue;
                }

                input_src_names.push_back(std::to_string(node_idx) + ":" + std::to_string(src_idx) + ":" + src_name);
            }
        }
    }

    bool operator==(const graph_key & other) const {
        return n_nodes == other.n_nodes && first_node_name == other.first_node_name &&
               last_node_name == other.last_node_name && input_src_names == other.input_src_names;
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key & key) const {
        size_t hash = std::hash<int>{}(key.n_nodes);
        if (key.n_nodes > 0) {
            hash ^= std::hash<std::string>{}(key.first_node_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<std::string>{}(key.last_node_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        for (const auto & input_src_name : key.input_src_names) {
            hash ^= std::hash<std::string>{}(input_src_name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

// Plain host-side (pointer, byte_size) signature for an I/O port's currently-bound tensor.
// GGML_OPENVINO_NPU_SKIP_REBIND (experimental, default off): used instead of comparing against
// InferRequest::get_*_tensor(), which itself carries significant per-call overhead on at least
// one NPU driver tested -- see ov_runtime_context::bound_input_sig_cache.
struct port_signature {
    const void * data = nullptr;
    size_t bytes = 0;
    bool operator==(const port_signature & o) const { return data == o.data && bytes == o.bytes; }
};

struct decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}

    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
    // Attention sizes the cached prefill model was built with; decode calls overwrite the decoder's.
    int prefill_attention_size = -1;
    int prefill_attention_size_swa = -1;
};

struct ov_compiled_graph {
    ov::CompiledModel decode;
    ov::CompiledModel prefill;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::string> prefill_input_names;
    std::vector<std::string> prefill_output_names;
    // GGML_OPENVINO_NPU_SPLIT_LM_HEAD (experimental, default off): the final output-projection
    // (lm_head) matmul, compiled as its own standalone model instead of being bundled into
    // decode/prefill. Empty (operator bool() == false) when the flag is off or the split wasn't
    // performed (e.g. a graph with no matching tail pattern). The body model (decode/prefill
    // above) then exposes the pre-lm_head hidden state as an extra output named
    // lm_head_boundary_name instead of the real logits output.
    ov::CompiledModel decode_lm_head;
    ov::CompiledModel prefill_lm_head;
    std::string lm_head_boundary_name;
    std::string lm_head_output_name;
};

// Only compilation and cache publication use this mutex. Requests, decoders and
// sequence state belong to individual backend contexts and never enter this cache.
struct ov_compiled_model_cache {
    std::mutex mutex;
    std::unordered_map<std::string, ov_compiled_graph> graphs;
    size_t backend_count = 0;
};

// Private to one backend instance. Only compiled_cache is shared with other
// instances; clearing these local caches cannot invalidate their requests.
struct ov_runtime_context {
    // Serializes calls on this backend only, not inference in other contexts.
    std::mutex execution_mutex;
    std::shared_ptr<ov_compiled_model_cache> compiled_cache;
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache_prefill;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache_prefill;
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_input_name_map;

    // GGML_OPENVINO_NPU_SPLIT_LM_HEAD (experimental, default off, decode only): the standalone
    // lm_head InferRequest for this key (absent/null when the split wasn't performed for this
    // graph), plus the boundary/output tensor names needed to wire body -> lm_head at execution
    // time. Populated alongside infer_request_cache; see ov_graph_compute_static.
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_lm_head;
    std::unordered_map<graph_key, std::string, graph_key_hash> lm_head_boundary_name_cache;
    std::unordered_map<graph_key, std::string, graph_key_hash> lm_head_output_name_cache;

    // GGML_OPENVINO_NPU_SKIP_REBIND (experimental, default off): last-bound (pointer, byte_size)
    // per I/O port index for this key, so set_input_tensor/set_output_tensor can be skipped when
    // nothing changed since the previous call. Only ever touched while execution_mutex is held
    // (see ov_graph_compute), so no separate lock is needed here.
    std::unordered_map<graph_key, std::vector<port_signature>, graph_key_hash> bound_input_sig_cache;
    std::unordered_map<graph_key, std::vector<port_signature>, graph_key_hash> bound_output_sig_cache;

    // GGML_OPENVINO_NPU_SPLIT_LM_HEAD (experimental, default off): the lm_head sub-request is
    // kicked off with start_async() instead of a blocking infer(), so its NPU execution can
    // overlap with whatever host-side work ggml-backend's caller does between dispatching this
    // graph_compute and actually reading the output tensor (ggml_backend_tensor_get_async falls
    // back to ggml_backend_synchronize() + a blocking copy when .get_tensor_async is NULL, which
    // is exactly the hook below uses to perform the deferred wait()). Any pending request here
    // is also waited on defensively at the start of the next ov_graph_compute call.
    std::shared_ptr<ov::InferRequest> pending_lm_head_wait;

    ov_runtime_context() : device("CPU"), stateful(false), stateful_kv_size(0) {}

    void clear_caches_locked() {
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_input_names_cache.clear();
        ov_output_names_cache.clear();
        ov_input_names_cache_prefill.clear();
        ov_output_names_cache_prefill.clear();
        kv_state_input_name_map.clear();
        stateful_kv_size = 0;
        infer_request_cache_lm_head.clear();
        lm_head_boundary_name_cache.clear();
        lm_head_output_name_cache.clear();
        bound_input_sig_cache.clear();
        bound_output_sig_cache.clear();
    }

    void clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        clear_caches_locked();
    }
};

enum ggml_status ov_graph_compute(struct ggml_cgraph * cgraph, ggml_backend_t backend);

enum ggml_status ov_graph_compute_dynamic(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);
enum ggml_status ov_graph_compute_static(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);

size_t checksum(const void * data, size_t size);

bool save_ggml_tensor_data_to_txt(const ggml_tensor * tensor, const std::string & file_path);

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor);

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst);

template <typename T>
std::vector<T> pad_input(const T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_value) {
    std::vector<T> padded(padded_rows * padded_cols, pad_value);

    for (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    return padded;
}

template <typename T>
std::vector<T> pad_input(const ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_value) {
    return pad_input<T>(reinterpret_cast<const T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_value);
}

const ggml_tensor * get_inp_pos_tensor(struct ggml_cgraph * cgraph);

int64_t get_inp_pos_n_tokens(struct ggml_cgraph * cgraph, const ggml_tensor * inp_pos);

bool get_is_prefill(struct ggml_cgraph * cgraph, const ggml_tensor * inp_pos);

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index);

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor);

bool is_naive(struct ggml_cgraph * cgraph);

/**
 * @brief Heuristically checks whether the given computation graph is a split-model fragment.
 * @param cgraph Pointer to the GGML computation graph to analyze.
 * @return true if the graph is identified as split; otherwise false.
 */
bool is_model_splitted(struct ggml_cgraph * cgraph);

enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config,
                               ov_compiled_model_cache & cache);
