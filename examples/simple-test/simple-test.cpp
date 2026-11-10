#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// fill the batch with tokens at consecutive positions starting from pos_0, output logits only for the last one
static void batch_set_tokens(llama_batch_ext * batch, const llama_token * tokens, int32_t n_tokens, llama_pos pos_0) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = pos_0 + i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-fa 0|1|auto] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "hello, what is sdpa?";
    int n_predict = 128;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    {
        int i = 1;
        for (; i < argc; i++) {
            const bool has_value = i + 1 < argc;
            if (strcmp(argv[i], "-m") == 0 && has_value) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && has_value) {
                try {
                    n_predict = std::stoi(argv[++i]);
                } catch (...) {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if ((strcmp(argv[i], "-fa") == 0 || strcmp(argv[i], "--flash-attn") == 0) && has_value) {
                const std::string v = argv[++i];
                if (v == "1" || v == "on") {
                    fa = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                } else if (v == "0" || v == "off") {
                    fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                } else if (v == "auto") {
                    fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (argv[i][0] == '-') {
                print_usage(argc, argv);
                return 1;
            } else {
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // wrap the prompt with the model's chat template so phi-4 answers instead of continuing the text
    std::string formatted = prompt;
    if (const char * tmpl = llama_model_chat_template(model, nullptr)) {
        llama_chat_message msg = { "user", prompt.c_str() };
        std::vector<char> buf(prompt.size() * 4 + 256);
        int32_t len = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
        if (len > (int32_t) buf.size()) {
            buf.resize(len);
            len = llama_chat_apply_template(tmpl, &msg, 1, true, buf.data(), buf.size());
        }
        if (len > 0) {
            formatted.assign(buf.data(), len);
        }
    }

    const int n_prompt = -llama_tokenize(vocab, formatted.c_str(), formatted.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, formatted.c_str(), formatted.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = n_prompt + n_predict;
    ctx_params.n_batch         = n_prompt;
    ctx_params.flash_attn_type = fa;
    ctx_params.no_perf         = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s: flash_attn = %s, n_prompt = %d, n_predict = %d\n",
            __func__, llama_flash_attn_type_name(fa), n_prompt, n_predict);

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        printf("%s", std::string(buf, n).c_str());
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    int n_tokens = n_prompt;
    batch_set_tokens(batch, prompt_tokens.data(), n_prompt, 0);

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;

    for (int n_pos = 0; n_pos + n_tokens < n_prompt + n_predict; ) {
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }
        n_pos += n_tokens;

        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        printf("%s", std::string(buf, n).c_str());
        fflush(stdout);

        batch_set_tokens(batch, &new_token_id, 1, n_pos);
        n_tokens = 1;
        n_decode += 1;
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_batch_ext_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
