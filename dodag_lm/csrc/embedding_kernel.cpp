#include <torch/extension.h>

namespace dodag {

torch::Tensor bilinear_forward(torch::Tensor parent, torch::Tensor child);

}  // namespace dodag

TORCH_LIBRARY(dodag, m) {
  m.def("bilinear_forward", &dodag::bilinear_forward);
}

TORCH_LIBRARY_IMPL(dodag, CUDA, m) {
  m.impl("bilinear_forward", &dodag::bilinear_forward);
}

TORCH_LIBRARY_IMPL(dodag, CPU, m) {
  m.impl("bilinear_forward", [](torch::Tensor parent, torch::Tensor child) {
    TORCH_CHECK(false, "CUDA implementation required but not available");
    return torch::Tensor();
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dodag_bilinear_forward", &dodag::bilinear_forward, "DoDAG bilinear forward (CUDA)");
}
