// Copyright (c) Megvii Inc. All rights reserved.
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>

extern THCState *state;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int average_voxel_pooling_forward_wrapper(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    at::Tensor geom_xyz_tensor, at::Tensor input_features_tensor, at::Tensor input_pos_tensor, at::Tensor output_features_tensor, at::Tensor output_pos_tensor, at::Tensor pos_memo_tensor);

void average_voxel_pooling_forward_kernel_launcher(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *input_features, const int *input_pos, float *output_features, float *output_pos, int *pos_memo, cudaStream_t stream);

int average_voxel_pooling_forward_wrapper(int batch_size, int num_points, int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    at::Tensor geom_xyz_tensor, at::Tensor input_features_tensor, at::Tensor input_pos_tensor, at::Tensor output_features_tensor, at::Tensor output_pos_tensor, at::Tensor pos_memo_tensor) {
    CHECK_INPUT(geom_xyz_tensor);
    CHECK_INPUT(input_features_tensor);
    CHECK_INPUT(input_pos_tensor);
    const int *geom_xyz = geom_xyz_tensor.data_ptr<int>();
    const float *input_features = input_features_tensor.data_ptr<float>();
    const int *input_pos = input_pos_tensor.data_ptr<int>();
    float *output_features = output_features_tensor.data_ptr<float>();
    float *output_pos = output_pos_tensor.data_ptr<float>();
    int *pos_memo = pos_memo_tensor.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    average_voxel_pooling_forward_kernel_launcher(batch_size, num_points, num_channels, num_voxel_x, num_voxel_y, num_voxel_z,
        geom_xyz, input_features, input_pos, output_features, output_pos, pos_memo, stream);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("average_voxel_pooling_forward_wrapper", &average_voxel_pooling_forward_wrapper, "average_voxel_pooling_forward_wrapper");
}
