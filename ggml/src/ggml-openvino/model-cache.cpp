#include "model-cache.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "openvino/rt_info/weightless_caching_attributes.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <openvino/core/model.hpp>
#include <openvino/core/version.hpp>
#include <openvino/op/constant.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#    include <direct.h>
#endif

namespace {

// 64-bit FNV-1a, the mixing primitive for all fingerprints here.
inline uint64_t fnv1a(uint64_t h, const void * data, size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

inline uint64_t fnv1a_u64(uint64_t h, uint64_t v) {
    return fnv1a(h, &v, sizeof(v));
}

constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;

// Bytes sampled from each end of a weight tensor for the sampled hash. The whole
// model is never hashed (that would cost seconds every run); instead we sample a
// bounded window from the head and tail of each weight's bytes. The manifest
// re-verify (same sample) guards the residual collision risk.
constexpr size_t WEIGHT_SAMPLE_BYTES = 4096;

// Is this src a model weight, mirroring create_weight_nodes()'s selection:
// non-view tensor whose buffer is USAGE_WEIGHTS or whose type is quantized.
bool is_weight_src(const ggml_tensor * src) {
    if (src == nullptr || src->view_src != nullptr || src->buffer == nullptr) {
        return false;
    }
    return src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type);
}

// Per-weight sampled fingerprint: identity (name/shape/type) + a bounded byte
// sample. Returns FNV offset basis if data is unavailable (kept deterministic).
uint64_t weight_fingerprint(const ggml_tensor * t) {
    uint64_t h = FNV_OFFSET;
    h = fnv1a(h, t->name, strlen(t->name));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        h = fnv1a_u64(h, static_cast<uint64_t>(t->ne[i]));
    }
    h = fnv1a_u64(h, static_cast<uint64_t>(t->type));
    const size_t nbytes = ggml_nbytes(t);
    h = fnv1a_u64(h, nbytes);
    if (t->data != nullptr && nbytes > 0) {
        const size_t head = nbytes < WEIGHT_SAMPLE_BYTES ? nbytes : WEIGHT_SAMPLE_BYTES;
        h = fnv1a(h, t->data, head);
        if (nbytes > WEIGHT_SAMPLE_BYTES) {
            const size_t tail = nbytes < 2 * WEIGHT_SAMPLE_BYTES ? nbytes - WEIGHT_SAMPLE_BYTES : WEIGHT_SAMPLE_BYTES;
            h = fnv1a(h, static_cast<const uint8_t *>(t->data) + (nbytes - tail), tail);
        }
    }
    return h;
}

// Walk the cgraph and invoke fn(weight_tensor) for each distinct weight, in node
// order. De-duplicates by tensor pointer so a weight used by several nodes is
// fingerprinted once, deterministically.
template <typename F>
void for_each_weight(const ggml_cgraph * cgraph, F && fn) {
    std::vector<const ggml_tensor *> seen;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (!is_weight_src(src)) {
                continue;
            }
            bool dup = false;
            for (const auto * p : seen) {
                if (p == src) {
                    dup = true;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            seen.push_back(src);
            fn(src);
        }
    }
}

std::string ov_version_string() {
    const ov::Version v = ov::get_openvino_version();
    return std::string(v.buildNumber ? v.buildNumber : "unknown");
}

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
    return std::string(buf);
}

// Portable mkdir for a single path component. Returns true if the directory
// exists after the call (created now or already present).
bool make_dir(const std::string & path) {
#if defined(_WIN32)
    int rc = _mkdir(path.c_str());
#else
    int rc = ::mkdir(path.c_str(), 0755);
#endif
    if (rc == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

// Create `path` and any missing parents (like `mkdir -p`). Best-effort:
// returns true only if the full directory exists afterwards.
bool make_dirs(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::string acc;
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        acc.push_back(c);
        const bool sep = (c == '/'
#if defined(_WIN32)
                          || c == '\\'
#endif
        );
        // Create each intermediate component (skip a leading "/" root).
        if (sep && acc.size() > 1) {
            std::string component = acc.substr(0, acc.size() - 1);
            if (!make_dir(component)) {
                return false;
            }
        }
    }
    return make_dir(path);
}

}  // namespace

std::string ggml_openvino_model_cache_dir() {
    const char * dir = ggml_openvino_getenv_str("GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR");
    if (!dir || strlen(dir) == 0) {
        return std::string();
    }
    std::string path(dir);
    // Create the cache directory (and parents) on first use so callers don't
    // have to pre-create it; a missing dir would otherwise silently disable the
    // cache (manifest/blob writes fail with no directory to write into).
    if (!make_dirs(path)) {
        GGML_LOG_WARN("ggml-openvino: could not create model cache dir '%s' (errno=%d); caching disabled\n",
                      path.c_str(), errno);
        return std::string();
    }
    return path;
}

uint64_t ggml_openvino_model_fingerprint(const ggml_cgraph * cgraph,
                                         const std::string & device,
                                         bool fa,
                                         const int32_t * rope_params,
                                         int rope_len,
                                         uint64_t extra_cfg) {
    uint64_t h = FNV_OFFSET;

    // Topology: node count + each node's op and name (cheap, and distinguishes
    // graphs that share weights but differ structurally).
    h = fnv1a_u64(h, static_cast<uint64_t>(cgraph->n_nodes));
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        h = fnv1a_u64(h, static_cast<uint64_t>(node->op));
        h = fnv1a(h, node->name, strlen(node->name));
    }

    // Weights: the model identity.
    for_each_weight(cgraph, [&](const ggml_tensor * t) { h = fnv1a_u64(h, weight_fingerprint(t)); });

    // Config that changes the produced blob.
    h = fnv1a(h, device.data(), device.size());
    h = fnv1a_u64(h, fa ? 1u : 0u);
    if (rope_params && rope_len > 0) {
        h = fnv1a(h, rope_params, sizeof(int32_t) * static_cast<size_t>(rope_len));
    }
    h = fnv1a_u64(h, extra_cfg);
    const std::string ver = ov_version_string();
    h = fnv1a(h, ver.data(), ver.size());

    return h;
}

std::string ggml_openvino_model_cache_blob_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".blob";
}

std::string ggml_openvino_model_cache_manifest_path(const std::string & dir, uint64_t fingerprint) {
    return dir + "/" + hex64(fingerprint) + ".manifest";
}

bool ggml_openvino_model_cache_write_manifest(const std::string & path,
                                              const ggml_cgraph * cgraph,
                                              uint64_t fingerprint) {
    std::ofstream f(path, std::ios::trunc);
    if (!f.is_open()) {
        return false;
    }
    f << "fingerprint " << hex64(fingerprint) << "\n";
    f << "ov_version " << ov_version_string() << "\n";
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        f << t->name << " " << t->ne[0] << " " << t->ne[1] << " " << t->ne[2] << " " << t->ne[3] << " "
          << static_cast<int>(t->type) << " " << hex64(weight_fingerprint(t)) << "\n";
    });
    return f.good();
}

bool ggml_openvino_model_cache_verify_manifest(const std::string & path,
                                               const ggml_cgraph * cgraph,
                                               uint64_t fingerprint) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::string tag, val;
    // header: fingerprint
    if (!(f >> tag >> val) || tag != "fingerprint" || val != hex64(fingerprint)) {
        return false;
    }
    // header: ov_version
    if (!(f >> tag >> val) || tag != "ov_version" || val != ov_version_string()) {
        return false;
    }

    // Build the expected per-weight lines from the live cgraph, then require an
    // exact match (same set, same order) against the manifest.
    std::vector<std::string> expected;
    for_each_weight(cgraph, [&](const ggml_tensor * t) {
        expected.push_back(std::string(t->name) + " " + std::to_string(t->ne[0]) + " " + std::to_string(t->ne[1]) +
                           " " + std::to_string(t->ne[2]) + " " + std::to_string(t->ne[3]) + " " +
                           std::to_string(static_cast<int>(t->type)) + " " + hex64(weight_fingerprint(t)));
    });

    size_t idx = 0;
    std::string line;
    std::getline(f, line);  // consume rest of ov_version line
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        if (idx >= expected.size() || line != expected[idx]) {
            return false;
        }
        ++idx;
    }
    return idx == expected.size();
}

std::string ggml_openvino_model_cache_weights_path(const std::string & dir, uint64_t fp_a, uint64_t fp_b) {
    return dir + "/" + hex64(fp_a) + "-" + hex64(fp_b) + ".weights.bin";
}

std::shared_ptr<ggml_openvino_weights_layout> ggml_openvino_model_cache_write_weights(
    const std::string & path,
    const std::vector<std::shared_ptr<ov::Model>> & models) {
    struct region {
        const void * data;
        size_t size;
        size_t offset;
    };
    constexpr size_t alignment = 64;
    auto layout = std::make_shared<ggml_openvino_weights_layout>();
    layout->path = path;
    std::vector<region> regions;
    std::unordered_map<const void *, size_t> region_of;
    for (const auto & model : models) {
        if (!model) {
            continue;
        }
        for (const auto & node : model->get_ordered_ops()) {
            auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
            if (!cnst || cnst->get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()) == 0 ||
                cnst->get_data_ptr() == nullptr || cnst->get_byte_size() == 0) {
                continue;
            }
            layout->constants.push_back(cnst);
            auto [it, inserted] = region_of.emplace(cnst->get_data_ptr(), regions.size());
            if (inserted) {
                regions.push_back({ cnst->get_data_ptr(), cnst->get_byte_size(), 0 });
            } else {
                regions[it->second].size = std::max(regions[it->second].size, cnst->get_byte_size());
            }
        }
    }

    size_t total = 0;
    for (auto & r : regions) {
        r.offset = total;
        total = (total + r.size + alignment - 1) / alignment * alignment;
        layout->regions[r.data] = { r.offset, r.size };
    }
    for (const auto & cnst : layout->constants) {
        cnst->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] = ov::WeightlessCacheAttribute(
            cnst->get_byte_size(), layout->regions.at(cnst->get_data_ptr()).first, cnst->get_element_type());
    }

    // Publish the data file first and its size record last, so a valid record always describes a complete file.
    const std::string data_tmp = path + ".tmp";
    const std::string meta_path = path + ".meta";
    const std::string meta_tmp = meta_path + ".tmp";
    {
        std::ofstream f(data_tmp, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            return nullptr;
        }
        static const char zeros[alignment] = {};
        size_t pos = 0;
        for (const auto & r : regions) {
            f.write(zeros, static_cast<std::streamsize>(r.offset - pos));
            f.write(static_cast<const char *>(r.data), static_cast<std::streamsize>(r.size));
            pos = r.offset + r.size;
        }
        f.write(zeros, static_cast<std::streamsize>(total - pos));
        f.close();
        if (!f.good()) {
            std::remove(data_tmp.c_str());
            return nullptr;
        }
    }
    {
        std::ofstream f(meta_tmp, std::ios::trunc);
        f << "size " << total << "\nregions " << regions.size() << "\n";
        f.close();
        if (!f.good()) {
            std::remove(data_tmp.c_str());
            std::remove(meta_tmp.c_str());
            return nullptr;
        }
    }
    // std::rename does not replace an existing file on Windows.
    ggml_openvino_model_cache_remove_weights(path);
    if (std::rename(data_tmp.c_str(), path.c_str()) != 0 || std::rename(meta_tmp.c_str(), meta_path.c_str()) != 0) {
        std::remove(data_tmp.c_str());
        std::remove(meta_tmp.c_str());
        return nullptr;
    }
    GGML_LOG_INFO("ggml-openvino: model cache WROTE %s (%zu regions, %.1f MiB)\n", path.c_str(), regions.size(),
                  total / 1024.0 / 1024.0);
    return layout;
}

size_t ggml_openvino_model_cache_retag_weights(const std::shared_ptr<ov::Model> & model,
                                               const ggml_openvino_weights_layout & layout) {
    size_t resolved = 0;
    for (const auto & node : model->get_ordered_ops()) {
        auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!cnst) {
            continue;
        }
        auto & rt_info = cnst->get_rt_info();
        auto attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (attr == rt_info.end()) {
            continue;
        }
        auto it = layout.regions.find(cnst->get_data_ptr());
        if (it != layout.regions.end() && cnst->get_byte_size() <= it->second.second) {
            attr->second = ov::WeightlessCacheAttribute(cnst->get_byte_size(), it->second.first, cnst->get_element_type());
            ++resolved;
        } else {
            rt_info.erase(attr);
        }
    }
    return resolved;
}

bool ggml_openvino_model_cache_weights_valid(const std::string & path) {
    std::ifstream meta(path + ".meta");
    std::string key;
    uint64_t size = 0;
    if (!(meta >> key >> size) || key != "size") {
        return false;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.is_open() && static_cast<uint64_t>(f.tellg()) == size;
}

void ggml_openvino_model_cache_remove_weights(const std::string & path) {
    std::remove((path + ".meta").c_str());
    std::remove(path.c_str());
}
