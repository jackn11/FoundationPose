# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from setuptools import setup
import os,sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import load

code_dir = os.path.dirname(os.path.realpath(__file__))


nvcc_flags = ['-Xcompiler', '-O3', '-std=c++14', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__']
c_flags = ['-O3', '-std=c++14']

setup(
    name='common',
    extra_cflags=c_flags,
    extra_cuda_cflags=nvcc_flags,
    ext_modules=[
        CUDAExtension('common', [
            'bindings.cpp',
            'common.cu',
        ],extra_compile_args={'gcc': c_flags, 'nvcc': nvcc_flags}),
        CUDAExtension('gridencoder', [
            f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
            f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
        ],extra_compile_args={'gcc': c_flags, 'nvcc': nvcc_flags}),
    ],
    include_dirs=[
        # "/usr/local/include/eigen3",
        # "/usr/include/eigen3",
        "/home/jacknaimer/anaconda3/envs/foundationpose/include/eigen/Eigen"
    ],
    cmdclass={
        'build_ext': BuildExtension
})
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.





# from setuptools import setup
# import os
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# code_dir = os.path.dirname(os.path.realpath(__file__))

# nvcc_flags = [
#     '-Xcompiler', '-O3', '-std=c++14',
#     '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
#     '--expt-relaxed-constexpr',
#     '--compiler-options', '-fPIC',
#     '-I/usr/local/cuda/include',  # CUDA include path
# ]
# c_flags = ['-O3', '-std=c++14', '-fPIC']

# setup(
#     name='common',
#     ext_modules=[
#         CUDAExtension(
#             'common', [
#                 '/home/jacknaimer/SchoelligLabProjects/FoundationPoseTest/src/FoundationPose/bundlesdf/mycuda/bindings.cpp',
#                 '/home/jacknaimer/SchoelligLabProjects/FoundationPoseTest/src/FoundationPose/bundlesdf/mycuda/common.cu',
#             ],
#             extra_compile_args={
#                 'cxx': c_flags,
#                 'nvcc': nvcc_flags
#             }
#         ),
#         CUDAExtension(
#             'gridencoder', [
#                 f"{code_dir}/torch_ngp_grid_encoder/gridencoder.cu",
#                 f"{code_dir}/torch_ngp_grid_encoder/bindings.cpp",
#             ],
#             extra_compile_args={
#                 'cxx': c_flags,
#                 'nvcc': nvcc_flags
#             }
#         ),
#     ],
#     include_dirs=[
#         "/usr/include/eigen3",
#         "/usr/local/cuda/include",  # Ensure CUDA include directory is added
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )
