#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstring>
#include <cstdint>
#include <cmath>
#include <memory>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/subtract.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// Reimplementation of GGML_OP_SOFT_MAX semantics for OpenVINO backend:
// 1) logits = src0 * scale
// 2) logits += mask (if provided)
// 3) if attention sinks are provided, include one per-head sink logit in the softmax denominator
// 4) softmax over the last dimension
OutputVector translate_soft_max(const NodeContext & context) {
    num_inputs_check(context, 1, 3);

    float scale = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&max_bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    ov::Output<ov::Node> logits = context.get_input(0);

    // Apply scale first: logits = src0 * scale
    if (scale != 1.0f) {
        auto scale_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
        logits = std::make_shared<ov::op::v1::Multiply>(logits, scale_const);
    }

    FRONT_END_CHECK_IMPLEMENTED(!(max_bias > 0.0f && context.get_input_size() < 2),
                                "OpenVINO softmax ALiBi path requires mask input");

    // Optional mask add: logits += mask
    // For max_bias > 0 (ALiBi), apply per-head slope to mask before adding.
    if (context.get_input_size() > 1) {
        ov::Output<ov::Node> mask = context.get_input(1);

        // For stateful
        std::string mask_name = "KQ_mask_sliced";
        if (context.get_input_names()[1].find("swa") != std::string::npos) {
            mask_name = "KQ_mask_swa_sliced";
        }
        if (context.has_input(mask_name)) {
            mask = context.get_input(mask_name);
        }

        if (mask.get_element_type() != logits.get_element_type()) {
            mask = std::make_shared<ov::op::v0::Convert>(mask, logits.get_element_type());
        }

        if (max_bias > 0.0f) {
            auto out_shape = context.get_output_shape().to_shape();
            FRONT_END_CHECK_IMPLEMENTED(out_shape.size() == 4,
                                        "OpenVINO softmax ALiBi path expects rank-4 tensor");

            const uint32_t n_head = static_cast<uint32_t>(out_shape[1]);
            FRONT_END_CHECK_IMPLEMENTED(n_head > 0, "OpenVINO softmax ALiBi path expects n_head > 0");

            const uint32_t n_head_log2 = 1u << static_cast<uint32_t>(std::floor(std::log2(static_cast<float>(n_head))));
            const float m0 = std::pow(2.0f, -(max_bias) / static_cast<float>(n_head_log2));
            const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / static_cast<float>(n_head_log2));

            std::vector<float> slopes(n_head);
            for (uint32_t h = 0; h < n_head; ++h) {
                slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<float>(h + 1))
                                             : std::pow(m1, static_cast<float>(2 * (h - n_head_log2) + 1));
            }

            ov::Output<ov::Node> slope_node =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_head}, slopes);
            if (slope_node.get_element_type() != mask.get_element_type()) {
                slope_node = std::make_shared<ov::op::v0::Convert>(slope_node, mask.get_element_type());
            }

            auto slope_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                       std::vector<int64_t>{1, static_cast<int64_t>(n_head), 1, 1});
            auto slope_4d = std::make_shared<ov::op::v1::Reshape>(slope_node, slope_shape, false);
            mask = std::make_shared<ov::op::v1::Multiply>(mask, slope_4d);
        }

        logits = std::make_shared<ov::op::v1::Add>(logits, mask);
    }

    ov::Output<ov::Node> res;
    if (context.get_input_size() > 2) {
        auto output_shape = context.get_output_shape();
        FRONT_END_CHECK_IMPLEMENTED(output_shape.rank().is_static() && output_shape.rank().get_length() == 4,
                                    "OpenVINO softmax sinks path expects rank-4 tensor");
        FRONT_END_CHECK_IMPLEMENTED(output_shape[1].is_static(),
                                    "OpenVINO softmax sinks path expects static head dimension");

        ov::Output<ov::Node> sinks = context.get_input(2);
        if (sinks.get_element_type() != logits.get_element_type()) {
            sinks = std::make_shared<ov::op::v0::Convert>(sinks, logits.get_element_type());
        }

        const int64_t n_head = output_shape[1].get_length();
        auto sinks_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, n_head, 1, 1});
        auto sinks_4d = std::make_shared<ov::op::v1::Reshape>(sinks, sinks_shape, false);

        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto softmax = std::make_shared<ov::op::v8::Softmax>(logits, -1);
        auto logits_max = std::make_shared<ov::op::v1::ReduceMax>(logits, axes, true);
        auto row_max = std::make_shared<ov::op::v1::Maximum>(logits_max, sinks_4d);
        auto exp_logits = std::make_shared<ov::op::v0::Exp>(std::make_shared<ov::op::v1::Subtract>(logits, row_max));
        auto sum_logits = std::make_shared<ov::op::v1::ReduceSum>(exp_logits, axes, true);
        auto exp_sink = std::make_shared<ov::op::v0::Exp>(std::make_shared<ov::op::v1::Subtract>(sinks_4d, row_max));
        auto denom = std::make_shared<ov::op::v1::Add>(sum_logits, exp_sink);
        auto correction = std::make_shared<ov::op::v1::Divide>(sum_logits, denom);
        res = std::make_shared<ov::op::v1::Multiply>(softmax, correction);
    } else {
        // Softmax along last dimension (equivalent to ggml softmax over ne[0]).
        res = std::make_shared<ov::op::v8::Softmax>(logits, -1);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
