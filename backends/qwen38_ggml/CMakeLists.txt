cmake_minimum_required(VERSION 3.18)
project(dial_qwen38_ggml LANGUAGES C CXX)

# Use the ggml directory of llama.cpp b10837, not the chat/RPC server.
set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp b10837 sources")
if(NOT EXISTS "${LLAMA_CPP_DIR}/ggml/include/ggml.h")
    message(FATAL_ERROR "Set LLAMA_CPP_DIR to the extracted llama.cpp-b10837 directory")
endif()
# The GGUF DeltaNet layout and graph ABI are pinned together. In particular,
# pre-reorder Qwen3.5 GGUFs must not be silently interpreted as this format.
file(SHA256 "${LLAMA_CPP_DIR}/ggml/include/ggml.h" DIAL_GGML_HEADER_HASH)
file(SHA256 "${LLAMA_CPP_DIR}/src/models/qwen35.cpp" DIAL_QWEN_GRAPH_HASH)
file(SHA256 "${LLAMA_CPP_DIR}/conversion/qwen.py" DIAL_QWEN_CONVERTER_HASH)
if(NOT DIAL_GGML_HEADER_HASH STREQUAL "34192eac913444df9dc1d23025a618448cdee1ee7d78597af4979d8edc77b695"
   OR NOT DIAL_QWEN_GRAPH_HASH STREQUAL "ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671"
   OR NOT DIAL_QWEN_CONVERTER_HASH STREQUAL "b783f53598e1a65cb0ef7fbe0d0f52da472e2784f63acf13a5aea00f623cfaa6")
    message(FATAL_ERROR "DIAL adapter requires unmodified llama.cpp b10837 header/model/converter sources")
endif()
option(GGML_CUDA "Build NVIDIA device backend" ON)
# ggml/ alone defaults this OFF; llama.cpp's top-level project enables it.
# Match that runtime optimization when embedding GGML instead of llama-server.
option(GGML_CUDA_GRAPHS "Reuse CUDA graphs for stable decode shapes" ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Portable CPU fallback for testing" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}/ggml" ggml)
add_library(dial_qwen38_ggml SHARED adapter.cpp)
target_compile_features(dial_qwen38_ggml PRIVATE cxx_std_17)
target_link_libraries(dial_qwen38_ggml PRIVATE ggml ggml-cpu)
target_compile_options(dial_qwen38_ggml PRIVATE -Wall -Wextra)
if(GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(dial_qwen38_ggml PRIVATE DIAL_GGML_CUDA=1)
    target_link_libraries(dial_qwen38_ggml PRIVATE ggml-cuda CUDA::cudart)
endif()
set_target_properties(dial_qwen38_ggml PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
