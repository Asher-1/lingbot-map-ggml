cmake_minimum_required(VERSION 3.21)
project(lingbot_map_ggml LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINGBOT_GGML_CUDA "Enable the ggml CUDA backend" OFF)
option(LINGBOT_GGML_VULKAN "Enable the ggml Vulkan backend" OFF)
option(LINGBOT_GGML_BUILD_TOOLS "Build lingbot-map-cli and lingbot-map-bench" ON)
set(LINGBOT_GGML_CUDA_ARCHITECTURES "" CACHE STRING "CUDA architectures passed to ggml (empty keeps ggml default)")
option(LINGBOT_GGML_CUDA_FORCE_MMQ "Use ggml MMQ kernels for quantized CUDA weights" OFF)
option(LINGBOT_GGML_CUDA_FORCE_CUBLAS "Use cuBLAS f32 accumulation for quantized CUDA weights instead of MMQ" ON)

if(LINGBOT_GGML_CUDA_FORCE_MMQ AND LINGBOT_GGML_CUDA_FORCE_CUBLAS)
    message(FATAL_ERROR "LINGBOT_GGML_CUDA_FORCE_MMQ and LINGBOT_GGML_CUDA_FORCE_CUBLAS are mutually exclusive")
endif()

if(LINGBOT_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_NO_VMM ON CACHE BOOL "" FORCE)
    if(LINGBOT_GGML_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "${LINGBOT_GGML_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
    endif()
    if(LINGBOT_GGML_CUDA_FORCE_MMQ)
        set(GGML_CUDA_FORCE_MMQ ON CACHE BOOL "" FORCE)
    endif()
    if(LINGBOT_GGML_CUDA_FORCE_CUBLAS)
        set(GGML_CUDA_FORCE_CUBLAS ON CACHE BOOL "" FORCE)
    endif()
endif()
if(LINGBOT_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

set(LINGBOT_GGML_DIR "${CMAKE_SOURCE_DIR}/third_party/ggml")
if(NOT EXISTS "${LINGBOT_GGML_DIR}/include/ggml.h")
    message(FATAL_ERROR "ggml v0.21.0 submodule is missing. Run: git submodule update --init --recursive")
endif()

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")
include(apply_ggml_patches)
lingbot_apply_ggml_patches("${LINGBOT_GGML_DIR}" "${CMAKE_SOURCE_DIR}/third_party/patches")

add_subdirectory("${LINGBOT_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml")

add_library(lingbot_map_ggml STATIC
    src/backend.cpp
    src/gguf_loader.cpp
    src/model.cpp
    src/skyseg.cpp
)
target_include_directories(lingbot_map_ggml PUBLIC include)
target_link_libraries(lingbot_map_ggml PUBLIC ggml)
# model.cpp uses std::thread; on glibc < 2.34 libpthread is a separate DSO that
# consumers of the static library must receive on the link line.
find_package(Threads REQUIRED)
target_link_libraries(lingbot_map_ggml PUBLIC Threads::Threads)
target_compile_options(lingbot_map_ggml PRIVATE -Wall -Wextra -Wpedantic)

if(LINGBOT_GGML_BUILD_TOOLS)
    add_executable(lingbot-map-cli tools/lingbot-map-cli.cpp tools/stb_impl.cpp)
    target_link_libraries(lingbot-map-cli PRIVATE lingbot_map_ggml)
    target_include_directories(lingbot-map-cli PRIVATE src tools)
    target_compile_options(lingbot-map-cli PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(lingbot-map-bench tools/lingbot-map-bench.cpp)
    target_link_libraries(lingbot-map-bench PRIVATE lingbot_map_ggml)
    target_compile_options(lingbot-map-bench PRIVATE -Wall -Wextra -Wpedantic)

    # Standalone sky-segmentation runner: dumps the raw probability map for
    # parity checks against onnxruntime and times the native inference.
    add_executable(skyseg-cli tools/skyseg-cli.cpp tools/stb_impl.cpp)
    target_link_libraries(skyseg-cli PRIVATE lingbot_map_ggml)
    target_include_directories(skyseg-cli PRIVATE tools src)
    target_compile_options(skyseg-cli PRIVATE -Wall -Wextra -Wpedantic -g -O1)
endif()
