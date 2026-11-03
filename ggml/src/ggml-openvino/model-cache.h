#pragma once

// Frontend-level compiled-model cache (GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR).
//
// The OpenVINO plugin's own ov::cache_dir caches the compiled blob keyed by the
// *OV model*, but producing that model still runs the full frontend every time:
// weight requantization (incl. the large token_embd F32 transient) and the
// ggml->OV graph conversion. This cache keys off a fingerprint computed directly
// from the ggml cgraph, so a hit skips requant + convert + compile entirely and
// instead imports a previously exported CompiledModel blob.
//
// Opt-in and independent from GGML_OPENVINO_CACHE_DIR. Default off.
// On NPU, GGML_OPENVINO_SELF_CONTAINED_BLOB=1 exports embedded-weight prefill/decode
// blobs. A warm hit skips weight requantization, graph conversion, and compilation,
// at the cost of roughly model-size storage per blob and separate NPUW weight banks.
// After both blobs exist, GGML_OPENVINO_SELF_CONTAINED_MMAP=1 additionally wraps the
// GGUF mmap instead of allocating the private OpenVINO model buffer. This mode is
// import-only and fails clearly if either required blob is missing or invalid.
// GGML_OPENVINO_NPU_WEIGHTLESS_BANK=1 is the same raw-weight flow, but exports weightless
// prefill/decode blobs plus ONE shared <prefill>-<decode>.weights.bin. Warm import reads the
// weights from that file (ov::weights_path) and NPUW deduplicates the two models into one bank
// (no OpenVINO patch needed), so the weights are stored once on disk and resident once.

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ov {
class Model;
namespace op {
namespace v0 {
class Constant;
}
}  // namespace op
}  // namespace ov

// Returns the compiled-model cache directory from GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR,
// or empty if unset/disabled. When empty, callers must not use the cache.
std::string ggml_openvino_model_cache_dir();

// Compute a stable 64-bit fingerprint identifying the model+config that a cgraph
// would compile to. Combines graph topology, a sampled hash of every weight
// tensor (name/shape/dtype + bounded byte sample), and the config that changes
// the produced blob (device, flash-attention, rope params, the compile-memory
// flags, stateful, and the OpenVINO version). `device` is the resolved device
// string; `fa` is the flash-attention flag; `rope_params`/`rope_len` cover the
// model's rope configuration; `extra_cfg` folds in any other blob-affecting bits.
uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg);

// Path to the compiled-blob file for a fingerprint (<dir>/<hex>.blob).
std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint);

// Path to the sidecar manifest (<dir>/<hex>.manifest) holding the per-weight
// fingerprints, used to re-verify a hit before trusting the blob.
std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint);

// Write/read the manifest. The manifest is a newline-separated list of
// "name ne0 ne1 ne2 ne3 type sample_hash" lines plus a header line with the
// fingerprint and OV version. Returns false on I/O error.
bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint);

// Verify that the cgraph's weights still match the stored manifest (guards the
// sampled-hash collision risk: a blob is only trusted if every weight's
// name/shape/type/sample-hash matches what was cached). Returns true on match.
bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint);

// GGML_OPENVINO_NPU_WEIGHTLESS_BANK: a weights file shared by weightless blobs. Every weightless-tagged
// Constant is one region keyed by its data pointer (NPUW maps a Constant to its file offset by pointer).
struct ggml_openvino_weights_layout {
    std::string path;
    // data pointer -> (offset, size) of its region in the file.
    std::unordered_map<const void *, std::pair<size_t, size_t>> regions;
    // Keeps the laid-out Constants alive so no other Constant can reuse one of their data pointers.
    std::vector<std::shared_ptr<ov::op::v0::Constant>> constants;
};

// <dir>/<a>-<b>.weights.bin (NPUW's weightless import needs the .bin extension).
std::string ggml_openvino_model_cache_weights_path(const std::string & dir, uint64_t fp_a, uint64_t fp_b);

// Lay out all weightless-tagged Constants of |models| in one file, retag their WeightlessCacheAttribute with the
// real offsets and write the file. Call before compiling. Returns nullptr on I/O error.
std::shared_ptr<ggml_openvino_weights_layout> ggml_openvino_model_cache_write_weights(
    const std::string & path,
    const std::vector<std::shared_ptr<ov::Model>> & models);

// Point the tagged Constants of |model| that live in |layout| at their regions and untag the rest, which NPUW
// then embeds in the blob. Returns the number of Constants resolved through the file.
size_t ggml_openvino_model_cache_retag_weights(const std::shared_ptr<ov::Model> & model,
                                               const ggml_openvino_weights_layout & layout);

// True if the weights file is complete (its size matches the record written after it).
bool ggml_openvino_model_cache_weights_valid(const std::string & path);

// Remove a weights file and its size record.
void ggml_openvino_model_cache_remove_weights(const std::string & path);
