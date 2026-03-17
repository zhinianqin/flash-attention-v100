cmake_minimum_required(VERSION 3.26)

project(vllm_flash_attn LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FA2_ENABLED ON)

# CUDA by default, can be overridden by using -DVLLM_TARGET_DEVICE=... (used by setup.py)
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Supported python versions. These should be kept in sync with setup.py.
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12" "3.13")

#
# Supported/expected torch versions for CUDA/ROCm.
#
# Currently, having an incorrect pytorch version results in a warning
# rather than an error.
#
# Note: these should be kept in sync with the torch version in setup.py.
# Likely should also be in sync with the vLLM version.
#
set(TORCH_SUPPORTED_VERSION_CUDA "2.10.0")

find_python_constrained_versions(${PYTHON_SUPPORTED_VERSIONS})

#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
message(DEBUG "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
if (NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()

set(VLLM_GPU_LANG "CUDA")

#
# Query torch for additional GPU compilation flags for the given
# `VLLM_GPU_LANG`.
# The final set of arches is stored in `VLLM_FA_GPU_FLAGS`.
#
get_torch_gpu_compiler_flags(VLLM_FA_GPU_FLAGS ${VLLM_GPU_LANG})

#
# Set nvcc parallelism.
#
if (NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
    list(APPEND VLLM_FA_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif ()


# Other flags
list(APPEND VLLM_FA_GPU_FLAGS --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math)

# If CUTLASS is compiled on NVCC >= 12.5, it by default uses
# cudaGetDriverEntryPointByVersion as a wrapper to avoid directly calling the
# driver API. This causes problems when linking with earlier versions of CUDA.
# Setting this variable sidesteps the issue by calling the driver directly.
list(APPEND VLLM_FA_GPU_FLAGS -DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

# Replace instead of appending, nvcc doesn't like duplicate -O flags.
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

#
# _C extension
#

if (FA2_ENABLED)
    # Only build sparse kernels for faster iteration during sparse development
    file(GLOB FA2_GEN_SRCS "csrc/flash_attn/src/flash_fwd_sparse_*.cu")

    # For CUDA we set the architectures on a per file basis
    if (VLLM_GPU_LANG STREQUAL "CUDA")
        set(FA2_ARCHS "7.0")
        message(STATUS "FA2_ARCHS: ${FA2_ARCHS}")

        set_gencode_flags_for_srcs(
            SRCS "${FA2_GEN_SRCS}"
            CUDA_ARCHS "${FA2_ARCHS}")
    endif()

    define_gpu_extension_target(
        _vllm_fa2_C
        DESTINATION vllm_flash_attn
        LANGUAGE ${VLLM_GPU_LANG}
        SOURCES
            csrc/flash_attn/flash_api.cpp
            csrc/flash_attn/flash_api_sparse.cpp
            csrc/flash_attn/flash_api_torch_lib.cpp
            ${FA2_GEN_SRCS}
        COMPILE_FLAGS ${VLLM_FA_GPU_FLAGS}
        USE_SABI 3
        WITH_SOABI)

    target_include_directories(_vllm_fa2_C PRIVATE
        csrc/flash_attn
        csrc/flash_attn/src
        csrc/common
        csrc/cutlass/include)

    # custom definitions
    target_compile_definitions(_vllm_fa2_C PRIVATE
        FLASHATTENTION_DISABLE_BACKWARD
        FLASHATTENTION_DISABLE_DROPOUT
        # FLASHATTENTION_DISABLE_ALIBI
        # FLASHATTENTION_DISABLE_SOFTCAP
        # FLASHATTENTION_DISABLE_UNEVEN_K
        # FLASHATTENTION_DISABLE_LOCAL
        FLASHATTENTION_DISABLE_PYBIND
        FLASHATTENTION_DISABLE_DENSE
    )
endif ()