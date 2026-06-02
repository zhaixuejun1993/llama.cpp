#include "../op_table.h"
#include "../utils.h"
namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    return {process_view_input_new(context, 0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
