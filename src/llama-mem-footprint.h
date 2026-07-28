#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

struct llama_mem_footprint_snapshot {
    long long vm_size_kib = 0;
    long long vm_rss_kib = 0;
    long long vm_hwm_kib = 0;
    long long rss_anon_kib = 0;
    long long rss_file_kib = 0;
    long long rss_shmem_kib = 0;
    bool valid = false;
};

static inline bool llama_mem_footprint_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("LLAMA_MEM_FOOTPRINT");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static inline bool llama_mem_footprint_parse_kib(const std::string & line, const char * key, long long & value) {
    const size_t key_len = std::strlen(key);
    if (line.compare(0, key_len, key) != 0) {
        return false;
    }

    const char * ptr = line.c_str() + key_len;
    while (*ptr == ' ' || *ptr == '\t') {
        ++ptr;
    }

    char * end = nullptr;
    const long long parsed = std::strtoll(ptr, &end, 10);
    if (end == ptr) {
        return false;
    }

    value = parsed;
    return true;
}

static inline llama_mem_footprint_snapshot llama_mem_footprint_read() {
    llama_mem_footprint_snapshot result;
    std::ifstream status("/proc/self/status");
    if (!status) {
        return result;
    }

    std::string line;
    while (std::getline(status, line)) {
        llama_mem_footprint_parse_kib(line, "VmSize:",  result.vm_size_kib)  ||
        llama_mem_footprint_parse_kib(line, "VmRSS:",   result.vm_rss_kib)   ||
        llama_mem_footprint_parse_kib(line, "VmHWM:",   result.vm_hwm_kib)   ||
        llama_mem_footprint_parse_kib(line, "RssAnon:", result.rss_anon_kib) ||
        llama_mem_footprint_parse_kib(line, "RssFile:", result.rss_file_kib) ||
        llama_mem_footprint_parse_kib(line, "RssShmem:", result.rss_shmem_kib);
    }
    result.valid = true;
    return result;
}

static inline double llama_mem_footprint_kib_to_mib(long long kib) {
    return kib / 1024.0;
}

static inline void llama_mem_footprint_print(const char * label) {
    if (!llama_mem_footprint_enabled()) {
        return;
    }

    const auto mem = llama_mem_footprint_read();
    if (!mem.valid) {
        std::fprintf(stderr, "llama_mem_footprint: %-72s unavailable\n", label);
        return;
    }

    std::fprintf(stderr,
            "llama_mem_footprint: %-72s VmRSS=%10.2f MiB VmHWM=%10.2f MiB VmSize=%10.2f MiB RssAnon=%10.2f MiB RssFile=%10.2f MiB RssShmem=%10.2f MiB\n",
            label,
            llama_mem_footprint_kib_to_mib(mem.vm_rss_kib),
            llama_mem_footprint_kib_to_mib(mem.vm_hwm_kib),
            llama_mem_footprint_kib_to_mib(mem.vm_size_kib),
            llama_mem_footprint_kib_to_mib(mem.rss_anon_kib),
            llama_mem_footprint_kib_to_mib(mem.rss_file_kib),
            llama_mem_footprint_kib_to_mib(mem.rss_shmem_kib));
}