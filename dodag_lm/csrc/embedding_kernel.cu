#include <torch/extension.h>

namespace dodag {
namespace {

template <typename scalar_t>
__global__ void bilinear_forward_kernel(
    const scalar_t* __restrict__ parent,
    const scalar_t* __restrict__ child,
    scalar_t* __restrict__ output,
    const int64_t rows,
    const int64_t dim) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  scalar_t acc = static_cast<scalar_t>(0);
  const scalar_t* parent_row = parent + row * dim;
  const scalar_t* child_row = child + row * dim;
  for (int64_t idx = 0; idx < dim; ++idx) {
    acc += parent_row[idx] * child_row[idx];
  }
  output[row] = acc;
}

}  // namespace

torch::Tensor bilinear_forward(torch::Tensor parent, torch::Tensor child) {
  TORCH_CHECK(parent.is_cuda(), "parent tensor must be CUDA");
  TORCH_CHECK(child.is_cuda(), "child tensor must be CUDA");
  TORCH_CHECK(parent.sizes() == child.sizes(), "parent and child shapes must match");
  TORCH_CHECK(parent.dim() == 2, "Inputs must be 2D matrices");

  auto rows = parent.size(0);
  auto dim = parent.size(1);
  auto output = torch::empty({rows}, parent.options());

  const int threads = 256;
  const int blocks = (rows + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(parent.scalar_type(), "dodag_bilinear_forward", [&] {
    bilinear_forward_kernel<scalar_t><<<blocks, threads>>>(
        parent.data_ptr<scalar_t>(),
        child.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        rows,
        dim);
  });

  return output;
}

}  // namespace dodag
