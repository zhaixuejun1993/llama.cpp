#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_swiglu(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        auto combined = context.get_input(0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis   = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step   = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    const int32_t swapped = params[1];
    if (swapped) {
        std::swap(src0, src1);
    }

    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(src0);
    auto silu = std::make_shared<ov::op::v1::Multiply>(src0, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_glu_swiglu_oai(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        auto combined = context.get_input(0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis   = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step   = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    const float * params_f32 = reinterpret_cast<const float *>(params);
    const float alpha = params_f32[2];
    const float limit = params_f32[3];

    const auto input_type = src0.get_element_type();
    if (src0.get_element_type() != ov::element::f32) {
        src0 = std::make_shared<ov::op::v0::Convert>(src0, ov::element::f32);
    }
    if (src1.get_element_type() != ov::element::f32) {
        src1 = std::make_shared<ov::op::v0::Convert>(src1, ov::element::f32);
    }

    auto limit_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {limit});
    auto neg_limit_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-limit});
    auto alpha_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {alpha});
    auto one_node = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});

    auto x = std::make_shared<ov::op::v1::Minimum>(src0, limit_node);
    auto g_upper = std::make_shared<ov::op::v1::Minimum>(src1, limit_node);
    auto g = std::make_shared<ov::op::v1::Maximum>(g_upper, neg_limit_node);
    auto alpha_x = std::make_shared<ov::op::v1::Multiply>(x, alpha_node);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(alpha_x);
    auto silu = std::make_shared<ov::op::v1::Multiply>(x, sigmoid);
    auto one_plus_g = std::make_shared<ov::op::v1::Add>(one_node, g);
    ov::Output<ov::Node> res = std::make_shared<ov::op::v1::Multiply>(silu, one_plus_g);
    if (res.get_element_type() != input_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, input_type);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
