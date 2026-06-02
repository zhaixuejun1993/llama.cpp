#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_add_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto input = process_view_input_new(context, 0);
    auto bias = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    if (input.get_element_type() != ov::element::f32) {
        input = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    }
    if (bias.get_element_type() != ov::element::f32) {
        bias = std::make_shared<ov::op::v0::Convert>(bias, ov::element::f32);
    }
    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    auto bias_shape = std::make_shared<ov::op::v3::ShapeOf>(bias, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);

    auto bias_rank = bias.get_partial_shape().rank();
    auto ids_rank = ids.get_partial_shape().rank();
    auto bias_shape_2d = bias_rank.is_static() && bias_rank.get_length() == 2 ? bias_shape : get_dimensions(bias_shape, {2, 3});
    auto ids_shape_2d = ids_rank.is_static() && ids_rank.get_length() == 2 ? ids_shape_4d : get_dimensions(ids_shape_4d, {2, 3});

    bias = std::make_shared<ov::op::v1::Reshape>(bias, bias_shape_2d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_bias = std::make_shared<ov::op::v8::Gather>(bias, ids, gather_axis);

    auto selected_bias_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(input_shape, {0}),
            ids_shape_2d,
            get_dimensions(input_shape, {3}),
        },
        0);
    selected_bias = std::make_shared<ov::op::v1::Reshape>(selected_bias, selected_bias_shape, false);

    ov::Output<ov::Node> result = std::make_shared<ov::op::v1::Add>(input, selected_bias);
    if (result.get_element_type() != context.get_output_type()) {
        result = std::make_shared<ov::op::v0::Convert>(result, context.get_output_type());
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov