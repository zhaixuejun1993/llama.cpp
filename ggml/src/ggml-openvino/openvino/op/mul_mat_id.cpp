#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

static float e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000 << x;
    } else {
        bits = static_cast<uint32_t>(x - 1) << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static ov::Output<ov::Node> slice_and_squeeze_last_dim(ov::Output<ov::Node> input, int64_t start, int64_t stop) {
    auto start_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {start});
    auto stop_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {stop});
    auto step_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axis_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

    auto sliced = std::make_shared<ov::op::v8::Slice>(input, start_node, stop_node, step_node, axis_node);
    return std::make_shared<ov::op::v0::Squeeze>(sliced, squeeze_axis);
}

static ov::Output<ov::Node> selected_mxfp4_dot(ov::Output<ov::Node> selected_blocks,
                                               ov::Output<ov::Node> activations) {
    static constexpr int mxfp4_values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    std::vector<float> scales(256);
    std::vector<float> low_values(256);
    std::vector<float> high_values(256);
    for (int i = 0; i < 256; ++i) {
        scales[i] = e8m0_to_fp32_half(static_cast<uint8_t>(i));
        low_values[i] = static_cast<float>(mxfp4_values[i & 0x0F]);
        high_values[i] = static_cast<float>(mxfp4_values[i >> 4]);
    }

    ov::Output<ov::Node> scale_bytes = slice_and_squeeze_last_dim(selected_blocks, 0, 1);
    scale_bytes = std::make_shared<ov::op::v0::Convert>(scale_bytes, ov::element::i32);

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto scale_table = ov::op::v0::Constant::create(ov::element::f32, {scales.size()}, scales);
    auto low_table = ov::op::v0::Constant::create(ov::element::f32, {low_values.size()}, low_values);
    auto high_table = ov::op::v0::Constant::create(ov::element::f32, {high_values.size()}, high_values);

    ov::Output<ov::Node> scale = std::make_shared<ov::op::v8::Gather>(scale_table, scale_bytes, gather_axis);

    auto selected_shape = std::make_shared<ov::op::v3::ShapeOf>(selected_blocks, ov::element::i64);
    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto block_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {32});
    auto activations_block_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0, 1}),
            get_dimensions(selected_shape, {3}),
            block_size,
        },
        0);
    auto activation_blocks = std::make_shared<ov::op::v1::Reshape>(activations, activations_block_shape, false);

    auto reduce_k_blocks_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto unsqueeze_row_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    ov::Output<ov::Node> accumulated;
    for (int packed_byte = 0; packed_byte < 16; ++packed_byte) {
        ov::Output<ov::Node> packed = slice_and_squeeze_last_dim(selected_blocks, packed_byte + 1, packed_byte + 2);
        packed = std::make_shared<ov::op::v0::Convert>(packed, ov::element::i32);

        ov::Output<ov::Node> low = std::make_shared<ov::op::v8::Gather>(low_table, packed, gather_axis);
        ov::Output<ov::Node> high = std::make_shared<ov::op::v8::Gather>(high_table, packed, gather_axis);

        ov::Output<ov::Node> act_low = slice_and_squeeze_last_dim(activation_blocks, packed_byte, packed_byte + 1);
        act_low = std::make_shared<ov::op::v0::Unsqueeze>(act_low, unsqueeze_row_axis);
        ov::Output<ov::Node> act_high = slice_and_squeeze_last_dim(activation_blocks, packed_byte + 16, packed_byte + 17);
        act_high = std::make_shared<ov::op::v0::Unsqueeze>(act_high, unsqueeze_row_axis);

        auto low_product = std::make_shared<ov::op::v1::Multiply>(low, act_low);
        auto high_product = std::make_shared<ov::op::v1::Multiply>(high, act_high);
        ov::Output<ov::Node> pair_sum = std::make_shared<ov::op::v1::Add>(low_product, high_product);
        pair_sum = std::make_shared<ov::op::v1::Multiply>(pair_sum, scale);
        auto partial = std::make_shared<ov::op::v1::ReduceSum>(pair_sum, reduce_k_blocks_axis, false);

        if (accumulated.get_node_shared_ptr() == nullptr) {
            accumulated = partial;
        } else {
            accumulated = std::make_shared<ov::op::v1::Add>(accumulated, partial);
        }
    }

    return accumulated;
}

}  // namespace

OutputVector translate_mul_mat_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto expert_weights = process_view_input_new(context, 0);
    auto activations = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    if (context.has_input("token_len_per_seq")) {
        auto token_len_per_seq = context.get_input("token_len_per_seq");
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto activations_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto ids_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

        activations = std::make_shared<ov::op::v8::Slice>(activations, zero, token_len_per_seq, one, activations_axis);
        ids = std::make_shared<ov::op::v8::Slice>(ids, zero, token_len_per_seq, one, ids_axis);
    }

    // OpenVINO sees GGML tensors in reversed dimension order:
    //   weights: [1, n_expert, m, k] or packed MXFP4 [1, n_expert, m, k_blocks, 17]
    //   activations: [1, n_tokens, n_used_or_1, k]
    //   ids: [1, 1, n_tokens, n_used]
    // Rebuild the logical ranks explicitly from the 4D inputs instead of relying
    // on fixed squeeze axes: real graphs can arrive through VIEW/RESHAPE chains
    // where singleton axes are still represented differently at this point.
    auto expert_weights_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(expert_weights, ov::element::i64);
    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);

    const bool is_packed_mxfp4 = expert_weights.get_element_type() == ov::element::u8 &&
                                 expert_weights.get_partial_shape().rank().is_static() &&
                                 expert_weights.get_partial_shape().rank().get_length() == 5;
    auto expert_weights_shape = is_packed_mxfp4 ? get_dimensions(expert_weights_shape_4d, {1, 2, 3, 4}) :
                                                  get_dimensions(expert_weights_shape_4d, {1, 2, 3});
    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, expert_weights_shape, false);
    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    const auto output_type = context.get_output_type();
    if (activations.get_element_type() != ov::element::f32) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, ov::element::f32);
    }

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted = std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims,
                                                                                     ov::op::BroadcastType::BIDIRECTIONAL);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto output_shape = context.get_output_shape();
    FRONT_END_OP_CONVERSION_CHECK(output_shape.rank().is_static() && output_shape.rank().get_length() == 4,
                                  "Unexpected MUL_MAT_ID output rank");
    FRONT_END_OP_CONVERSION_CHECK(output_shape[3].is_static(),
                                  "Expected static row dimension for MUL_MAT_ID output");
    const auto row_dim_value = output_shape[3].get_length();
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {row_dim_value});

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    auto used_axis_i32 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto used_axis_i64 = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    const auto ids_ps = ids.get_partial_shape();
    const bool can_split_by_used =
        ids_ps.rank().is_static() && ids_ps.rank().get_length() == 2 && ids_ps[1].is_static() && ids_ps[1].get_length() > 1;
    bool should_split_mxfp4_by_used = false;
    if (can_split_by_used) {
        const auto weights_ps = expert_weights.get_partial_shape();
        if (weights_ps.rank().is_static() && weights_ps.rank().get_length() == 4 && ids_ps[0].is_static() &&
            weights_ps[1].is_static() && weights_ps[2].is_static() && weights_ps[3].is_static()) {
            size_t selected_bytes = 1;
            auto checked_mul_dim = [](size_t a, int64_t b, size_t & out) {
                if (b < 0) {
                    return false;
                }
                const size_t ub = static_cast<size_t>(b);
                if (ub != 0 && a > SIZE_MAX / ub) {
                    return false;
                }
                out = a * ub;
                return true;
            };

            should_split_mxfp4_by_used =
                checked_mul_dim(selected_bytes, ids_ps[0].get_length(), selected_bytes) &&
                checked_mul_dim(selected_bytes, ids_ps[1].get_length(), selected_bytes) &&
                checked_mul_dim(selected_bytes, weights_ps[1].get_length(), selected_bytes) &&
                checked_mul_dim(selected_bytes, weights_ps[2].get_length(), selected_bytes) &&
                checked_mul_dim(selected_bytes, weights_ps[3].get_length(), selected_bytes) &&
                selected_bytes > (256ULL << 20);
        }
    }

    ov::Output<ov::Node> result;
    if (is_packed_mxfp4) {
        // Process one used-slot at a time to avoid materializing
        // [n_tokens, n_used, rows, k_blocks, 17] selected blocks in one Gather.
        if (should_split_mxfp4_by_used) {
            const auto n_used = static_cast<int64_t>(ids_ps[1].get_length());
            ov::OutputVector per_used_results;
            per_used_results.reserve(static_cast<size_t>(n_used));

            for (int64_t i = 0; i < n_used; ++i) {
                auto used_index = ov::op::v0::Constant::create(ov::element::i64, {1}, {i});

                ov::Output<ov::Node> ids_i = std::make_shared<ov::op::v8::Gather>(ids, used_index, used_axis_i32);
                ov::Output<ov::Node> acts_i = std::make_shared<ov::op::v8::Gather>(acts_broadcasted, used_index, used_axis_i32);
                ov::Output<ov::Node> selected_blocks_i =
                    std::make_shared<ov::op::v8::Gather>(expert_weights, ids_i, gather_axis);

                per_used_results.push_back(selected_mxfp4_dot(selected_blocks_i, acts_i));
            }

            result = std::make_shared<ov::op::v0::Concat>(per_used_results, 1);
        } else {
            ov::Output<ov::Node> selected_blocks = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
            result = selected_mxfp4_dot(selected_blocks, acts_broadcasted);
        }
    } else {
        // Process one used-slot at a time to avoid materializing
        // [n_tokens, n_used, rows, k] selected weights in a single Gather.
        if (can_split_by_used) {
            const auto n_used = static_cast<int64_t>(ids_ps[1].get_length());
            ov::OutputVector per_used_results;
            per_used_results.reserve(static_cast<size_t>(n_used));

            for (int64_t i = 0; i < n_used; ++i) {
                auto used_index = ov::op::v0::Constant::create(ov::element::i64, {1}, {i});

                ov::Output<ov::Node> ids_i = std::make_shared<ov::op::v8::Gather>(ids, used_index, used_axis_i32);
                ids_i = std::make_shared<ov::op::v0::Squeeze>(ids_i, used_axis_i64);

                ov::Output<ov::Node> selected_weights_i =
                    std::make_shared<ov::op::v8::Gather>(expert_weights, ids_i, gather_axis);
                if (selected_weights_i.get_element_type() != ov::element::f32) {
                    selected_weights_i = std::make_shared<ov::op::v0::Convert>(selected_weights_i, ov::element::f32);
                }

                ov::Output<ov::Node> activations_i = std::make_shared<ov::op::v8::Gather>(acts_broadcasted, used_index, used_axis_i32);
                per_used_results.push_back(std::make_shared<ov::op::v0::MatMul>(activations_i, selected_weights_i, false, true));
            }

            result = std::make_shared<ov::op::v0::Concat>(per_used_results, 1);
        } else {
            ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);
            if (selected_weights.get_element_type() != ov::element::f32) {
                selected_weights = std::make_shared<ov::op::v0::Convert>(selected_weights, ov::element::f32);
            }

            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
            auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, unsqueeze_axes);
            result = std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);
        }
    }

    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            batch_dim,
            get_dimensions(ids_shape, {0, 1}),
            row_dim,
        },
        0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);

    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov