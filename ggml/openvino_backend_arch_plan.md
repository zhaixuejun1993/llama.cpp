## Plan: OpenVINO Backend Architecture Optimization

TL;DR: The OpenVINO backend is a translation-based ggml backend: ggml scheduler selects supported graph fragments, OpenVINO buffer code prepares tensor/weight extras, GgmlOvDecoder exposes ggml_cgraph as a custom OpenVINO frontend input model, TranslateSession lowers ops via op_table translators into ov::Model, then runtime compiles/caches InferRequest and executes. Recommended optimization direction: preserve this architecture, but make graph identity, tensor lifetime, op capability checks, and model-specific pattern handling more declarative and less brittle.

**Steps**
1. Establish an architecture baseline by documenting the current layers and data flow: backend registration, device config, buffers, runtime graph execution, decoder, frontend translation, op translators, quantization, passes, and scheduler capability checks.
2. Harden graph/runtime caching. Replace the current graph_key heuristics with a more explicit cache signature that includes relevant shapes, dynamic/static mode, stateful mode, device, compile config, KV buffer identity, and important op params such as RoPE. Keep first/last node name as a fast debugging hint, not as the primary identity.
3. Split backend core responsibilities. Move pattern-specific unsupported cases out of ggml_backend_openvino_device_supports_op into a capability registry or rule table grouped by op/device/type/model-pattern. Keep the backend interface layer thin.
4. Improve tensor lifetime ownership. Make tensor->extra usage explicit through a small RAII owner/handle abstraction, so weight constants, quantized extracted tensors, remote tensors, and host tensors have one visible ownership model. Preserve ggml buffer integration.
5. Make GgmlOvDecoder less string-pattern driven. Centralize name-based predicates such as KV cache, inp_pos, inp_mask, output_idx, SWA, layer extraction into a small semantic tensor classifier. Prefer graph flags and structural roles where available; keep names as fallback.
6. Expand translation normalization before op lowering. Add a pre-lowering canonicalization layer that normalizes VIEW/RESHAPE/PERMUTE/in-place aliases, reducing per-op special cases in op translators.
7. Convert more ad hoc translator cases into OpenVINO passes. Keep op translators close to semantic lowering, then apply custom passes for NPU/GPU layout fixes, MatMul squeezing, attention fusion, stateful KV shaping, and decompression attributes.
8. Optimize quantized weight path. Preserve pre-built Constant/subgraph extras, but add measurable cache/reuse controls for extracted/requantized tensors and separate extraction policy from extraction implementation.
9. Add architecture-level verification. Use backend op tests, selected LLM prompt eval/decode runs on CPU/GPU/NPU, cache hit profiling, IR dumps, and memory/profiling toggles to verify each refactor.

**Relevant files**
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/include/ggml-openvino.h` - public backend API.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-backend-reg.cpp` - backend registry integration via GGML_USE_OPENVINO.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/ggml-openvino.cpp` - backend interface, buffer type, device interface, supports_op rules, registration.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/ggml-openvino-extra.h` and `.cpp` - device/env singleton, remote context, OpenCL queue, tensor extra classes.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/utils.h` and `.cpp` - runtime graph compute, graph caches, dynamic/static execution, InferRequest input/output binding.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/ggml-decoder.h` and `.cpp` - GgmlOvDecoder graph analysis, model inputs/outputs, op cases, shape/type/stride API.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/openvino/frontend.cpp`, `input_model.cpp`, `translate_session.cpp`, `node_context.h` - custom OpenVINO frontend and lowering session.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/openvino/op_table.cpp` and `openvino/op/*.cpp` - op dispatch table and per-op translators.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/ggml-quants.h` and `.cpp` - quantized weight extraction, requantization, OpenVINO weight subgraphs.
- `/home/openvino-288v002/xuejun/llama.cpp/ggml/src/ggml-openvino/openvino/pass/*.cpp` - custom OpenVINO graph transformation passes.

**Verification**
1. Build with `GGML_OPENVINO=ON` for CPU first, then GPU/NPU if devices are available.
2. Run focused backend op tests for OpenVINO-supported ops, especially VIEW/CPY/SET/MUL_MAT/MUL_MAT_ID/ROPE/FLASH_ATTN_EXT/SSM/GATED_DELTA_NET.
3. Run at least one prompt eval and decode benchmark on a small GGUF model, with `GGML_OPENVINO_PROFILING=1`, and compare cache hit behavior before/after cache changes.
4. Use `GGML_OPENVINO_DUMP_CGRAPH=1` and `GGML_OPENVINO_DUMP_IR=1` selectively to compare graph boundaries and generated OV IR.
5. Validate CPU fallback boundaries by checking unsupported cases still route away from OpenVINO instead of failing inside translation or compile_model.

**Decisions**
- Keep the translation-based architecture; do not switch to eager per-op OpenVINO execution.
- Treat CPU/GPU/NPU differences as capability/configuration data where possible, not scattered conditionals.
- Prioritize correctness and maintainability before aggressive fusion.
- Defer custom OpenVINO ops unless a repeated subgraph is proven expensive and unsupported by passes.

**Further Considerations**
1. Cache invalidation is the highest-risk optimization: implement it first behind diagnostics and compare IR/cache hit rates.
2. Stateful KV execution can be a major performance gain, but should remain opt-in until multi-request semantics and reset behavior are robust.
3. Model-specific exceptions should gradually move into named rules with comments/tests, so future llama.cpp graph changes do not silently break backend placement.