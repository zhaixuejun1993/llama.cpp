#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_elements_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

std::shared_ptr<ov::op::v0::Constant> const_i64(const std::vector<int64_t> & values) {
    return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

void check_rank_4(const ov::PartialShape & shape, const char * name) {
    FRONT_END_OP_CONVERSION_CHECK(shape.rank().is_static() && shape.rank().get_length() == 4,
                                  "SET_ROWS expects rank-4 ", name, " tensor, got ", shape);
}

int64_t get_static_dim(const ov::PartialShape & shape, size_t axis, const char * name) {
    FRONT_END_OP_CONVERSION_CHECK(shape[axis].is_static(), "SET_ROWS requires static ", name, " dimension ", axis,
                                  ", got ", shape);
    return shape[axis].get_length();
}

void check_same_dim_if_static(const ov::PartialShape & lhs,
                              const ov::PartialShape & rhs,
                              size_t axis,
                              const char * lhs_name,
                              const char * rhs_name) {
    if (lhs[axis].is_static() && rhs[axis].is_static()) {
        FRONT_END_OP_CONVERSION_CHECK(lhs[axis].get_length() == rhs[axis].get_length(), "SET_ROWS ", lhs_name,
                                      " and ", rhs_name, " shapes are incompatible: ", lhs, " vs ", rhs);
    }
}

size_t shape_elements(const ov::Shape & shape) {
    size_t result = 1;
    for (const auto dim : shape) {
        result *= dim;
    }
    return result;
}

bool dst_view_can_reshape_back_to_base(const NodeContext & context, size_t input_index) {
    const size_t view_count = context.get_view_input_size(input_index);
    if (view_count == 0) {
        return false;
    }

    const ov::Shape view_shape = context.get_view_input_ggml_shape(input_index, 0);
    const ov::Shape base_shape = context.get_view_input_src_ggml_shape(input_index, view_count - 1);
    return !view_shape.empty() && !base_shape.empty() && shape_elements(view_shape) == shape_elements(base_shape);
}

}  // namespace

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = process_view_input_new(context, 0);
    auto indices = process_view_input_new(context, 1);
    auto dst = context.get_input(2);
    auto dst_for_scatter = process_view_input_new(context, 2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    Output<Node> res;
    if (context.is_stateful()) {
        int concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } else {
        const auto dst_shape = context.get_input_shape(2);
        const auto data_shape = context.get_input_shape(0);
        const auto indices_shape = context.get_input_shape(1);

        check_rank_4(dst_shape, "destination");
        check_rank_4(data_shape, "data");
        check_rank_4(indices_shape, "indices");

        const int64_t dst_dim0 = get_static_dim(dst_shape, 0, "destination");
        const int64_t dst_dim1 = get_static_dim(dst_shape, 1, "destination");
        const int64_t indices_dim0 = get_static_dim(indices_shape, 0, "indices");
        const int64_t indices_dim1 = get_static_dim(indices_shape, 1, "indices");
        const int64_t indices_dim2 = get_static_dim(indices_shape, 2, "indices");

        FRONT_END_OP_CONVERSION_CHECK(indices_dim0 == 1, "SET_ROWS indices must have ggml ne[3] == 1");
        FRONT_END_OP_CONVERSION_CHECK(indices_dim1 > 0 && indices_dim2 > 0 && dst_dim0 % indices_dim1 == 0 &&
                                          dst_dim1 % indices_dim2 == 0,
                                      "SET_ROWS indices do not broadcast over destination dims");

        check_same_dim_if_static(data_shape, dst_shape, 0, "data", "destination");
        check_same_dim_if_static(data_shape, dst_shape, 1, "data", "destination");
        check_same_dim_if_static(data_shape, dst_shape, 3, "data", "destination");
        if (indices_shape[3].is_static() && data_shape[2].is_static()) {
            FRONT_END_OP_CONVERSION_CHECK(indices_shape[3].get_length() == data_shape[2].get_length(),
                                          "SET_ROWS indices length must match update rows: ", indices_shape, " vs ",
                                          data_shape);
        }

        auto indices_3d = std::make_shared<ov::op::v0::Squeeze>(indices, const_i64({0}));
        auto indices_tiled = std::make_shared<ov::op::v0::Tile>(
            indices_3d, const_i64({dst_dim0 / indices_dim1, dst_dim1 / indices_dim2, 1}));
        auto indices_4d = std::make_shared<ov::op::v0::Unsqueeze>(indices_tiled, const_i64({3}));
        auto indices_full = std::make_shared<ov::op::v3::Broadcast>(
            indices_4d, std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64), ov::op::BroadcastType::BIDIRECTIONAL);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst_for_scatter, indices_full, data, axes);

        if (dst_view_can_reshape_back_to_base(context, 2)) {
            res = std::make_shared<ov::op::v1::Reshape>(
                res, std::make_shared<ov::op::v3::ShapeOf>(dst, ov::element::i64), false);
        }
    }

    if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
