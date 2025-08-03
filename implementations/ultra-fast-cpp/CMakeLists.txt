# Ultra-Fast Knowledge Graph C++ Implementation
# Based on Kuzu architecture with extreme performance optimizations

cmake_minimum_required(VERSION 3.20)
project(UltraFastKnowledgeGraph
    VERSION 1.0.0
    DESCRIPTION "The fastest knowledge graph database ever built"
    LANGUAGES CXX C ASM
)

# Set C++ standard and compile flags for maximum performance
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type optimization
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations for maximum performance
if(CMAKE_CXX_COMPILER_ID STREQUAL "GCC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-stack-protector -fno-plt")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto=thin -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -fomit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx512f -mavx512cd -mavx512vl -mavx512bw -mavx512dq")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-stack-protector")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /arch:AVX512 /fp:fast /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
endif()

# Link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Enable all warnings and treat warnings as errors in debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GCC" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic -Werror")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address,undefined")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")
    endif()
endif()

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Optional dependencies
find_package(OpenMP)
find_package(TBB)
find_package(MPI)

# CUDA support for GPU acceleration
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
else()
    set(ENABLE_CUDA OFF)
    message(STATUS "CUDA not found, GPU acceleration disabled")
endif()

# Intel MKL for optimized linear algebra
find_package(MKL QUIET)
if(MKL_FOUND)
    set(ENABLE_MKL ON)
    message(STATUS "Intel MKL found")
else()
    set(ENABLE_MKL OFF)
    message(STATUS "Intel MKL not found")
endif()

# Check for AVX-512 support
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    __m512 b = _mm512_set1_ps(2.0f);
    __m512 c = _mm512_add_ps(a, b);
    return 0;
}
" HAVE_AVX512)

if(HAVE_AVX512)
    message(STATUS "AVX-512 support detected")
    add_compile_definitions(HAVE_AVX512)
else()
    message(WARNING "AVX-512 not supported on this platform")
endif()

# Memory allocator options
option(USE_JEMALLOC "Use jemalloc allocator" ON)
option(USE_MIMALLOC "Use mimalloc allocator" OFF)
option(USE_TCMALLOC "Use tcmalloc allocator" OFF)

if(USE_JEMALLOC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(JEMALLOC jemalloc)
    if(JEMALLOC_FOUND)
        message(STATUS "Using jemalloc allocator")
        add_compile_definitions(USE_JEMALLOC)
    endif()
elseif(USE_MIMALLOC)
    find_package(mimalloc QUIET)
    if(mimalloc_FOUND)
        message(STATUS "Using mimalloc allocator")
        add_compile_definitions(USE_MIMALLOC)
    endif()
elseif(USE_TCMALLOC)
    find_library(TCMALLOC_LIBRARY tcmalloc)
    if(TCMALLOC_LIBRARY)
        message(STATUS "Using tcmalloc allocator")
        add_compile_definitions(USE_TCMALLOC)
    endif()
endif()

# Performance profiling support
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(ENABLE_PROFILING)
    add_compile_definitions(ENABLE_PROFILING)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GCC" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/generated
)

# Source files organization
set(CORE_SOURCES
    src/core/graph.cpp
    src/core/node.cpp
    src/core/edge.cpp
    src/core/property.cpp
)

set(STORAGE_SOURCES
    src/storage/csr_matrix.cpp
    src/storage/node_storage.cpp
    src/storage/edge_storage.cpp
    src/storage/page_manager.cpp
    src/storage/buffer_pool.cpp
    src/storage/memory_manager.cpp
)

set(QUERY_SOURCES
    src/query/query_engine.cpp
    src/query/pattern_matcher.cpp
    src/query/query_planner.cpp
    src/query/execution_engine.cpp
    src/query/result_processor.cpp
)

set(ALGORITHMS_SOURCES
    src/algorithms/traversal.cpp
    src/algorithms/shortest_path.cpp
    src/algorithms/centrality.cpp
    src/algorithms/pagerank.cpp
    src/algorithms/community_detection.cpp
)

set(SIMD_SOURCES
    src/simd/avx512_operations.cpp
    src/simd/vectorized_algorithms.cpp
    src/simd/simd_utils.cpp
)

set(DISTRIBUTED_SOURCES
    src/distributed/shard_manager.cpp
    src/distributed/network_layer.cpp
    src/distributed/load_balancer.cpp
    src/distributed/fault_tolerance.cpp
)

set(UTILS_SOURCES
    src/utils/hash.cpp
    src/utils/serialization.cpp
    src/utils/compression.cpp
    src/utils/threading.cpp
    src/utils/metrics.cpp
    src/utils/logging.cpp
)

# Assembly sources for hand-optimized critical paths
set(ASM_SOURCES)
if(HAVE_AVX512)
    list(APPEND ASM_SOURCES
        src/asm/avx512_matrix_multiply.S
        src/asm/avx512_distance_update.S
        src/asm/avx512_neighbor_count.S
    )
endif()

# CUDA sources for GPU acceleration
set(CUDA_SOURCES)
if(ENABLE_CUDA)
    list(APPEND CUDA_SOURCES
        src/cuda/gpu_algorithms.cu
        src/cuda/gpu_memory_manager.cu
        src/cuda/gpu_pagerank.cu
        src/cuda/gpu_traversal.cu
    )
endif()

# Combine all sources
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${STORAGE_SOURCES}
    ${QUERY_SOURCES}
    ${ALGORITHMS_SOURCES}
    ${SIMD_SOURCES}
    ${DISTRIBUTED_SOURCES}
    ${UTILS_SOURCES}
    ${ASM_SOURCES}
    ${CUDA_SOURCES}
)

# Create the main library
add_library(ultra_fast_kg STATIC ${ALL_SOURCES})

# Set target properties for maximum performance
set_target_properties(ultra_fast_kg PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Compiler-specific optimizations
target_compile_options(ultra_fast_kg PRIVATE
    $<$<CXX_COMPILER_ID:GCC>:-Wall -Wextra -Wpedantic -ffast-math>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -ffast-math>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /fp:fast>
)

# Link libraries
target_link_libraries(ultra_fast_kg
    Threads::Threads
)

# Optional library linking
if(OpenMP_FOUND)
    target_link_libraries(ultra_fast_kg OpenMP::OpenMP_CXX)
    target_compile_definitions(ultra_fast_kg PRIVATE HAVE_OPENMP)
endif()

if(TBB_FOUND)
    target_link_libraries(ultra_fast_kg TBB::tbb)
    target_compile_definitions(ultra_fast_kg PRIVATE HAVE_TBB)
endif()

if(MPI_FOUND)
    target_link_libraries(ultra_fast_kg MPI::MPI_CXX)
    target_compile_definitions(ultra_fast_kg PRIVATE HAVE_MPI)
endif()

if(JEMALLOC_FOUND)
    target_link_libraries(ultra_fast_kg ${JEMALLOC_LIBRARIES})
    target_include_directories(ultra_fast_kg PRIVATE ${JEMALLOC_INCLUDE_DIRS})
endif()

if(mimalloc_FOUND)
    target_link_libraries(ultra_fast_kg mimalloc)
endif()

if(TCMALLOC_LIBRARY)
    target_link_libraries(ultra_fast_kg ${TCMALLOC_LIBRARY})
endif()

if(ENABLE_CUDA)
    set_property(TARGET ultra_fast_kg PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89 90)
    target_link_libraries(ultra_fast_kg CUDA::cudart CUDA::cublas CUDA::cusparse)
    target_compile_definitions(ultra_fast_kg PRIVATE ENABLE_CUDA)
endif()

if(ENABLE_MKL)
    target_link_libraries(ultra_fast_kg ${MKL_LIBRARIES})
    target_include_directories(ultra_fast_kg PRIVATE ${MKL_INCLUDE_DIRS})
    target_compile_definitions(ultra_fast_kg PRIVATE ENABLE_MKL)
endif()

# Create executable targets
add_executable(kg_server src/main/server.cpp)
target_link_libraries(kg_server ultra_fast_kg)

add_executable(kg_benchmark src/main/benchmark.cpp)
target_link_libraries(kg_benchmark ultra_fast_kg)

add_executable(kg_test src/main/test.cpp)
target_link_libraries(kg_test ultra_fast_kg)

# Installation
install(TARGETS ultra_fast_kg kg_server kg_benchmark kg_test
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Testing framework
enable_testing()

# Google Test for unit testing
find_package(GTest QUIET)
if(GTest_FOUND)
    add_subdirectory(tests)
endif()

# Google Benchmark for performance testing
find_package(benchmark QUIET)
if(benchmark_FOUND)
    add_subdirectory(benchmarks)
endif()

# Documentation generation
find_package(Doxygen QUIET)
if(Doxygen_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    add_custom_target(doc_doxygen ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/UltraFastKnowledgeGraphConfigVersion.cmake"
    VERSION "${version}"
    COMPATIBILITY AnyNewerVersion
)

# CPack configuration for distribution
set(CPACK_PACKAGE_NAME "UltraFastKnowledgeGraph")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The fastest knowledge graph database ever built")
set(CPACK_PACKAGE_VENDOR "Ultra-Fast Knowledge Graph Team")
set(CPACK_PACKAGE_CONTACT "support@ultrafast-kg.com")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

include(CPack)

# Print configuration summary
message(STATUS "")
message(STATUS "Ultra-Fast Knowledge Graph Configuration Summary:")
message(STATUS "================================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "")
message(STATUS "Performance Features:")
message(STATUS "  AVX-512 Support: ${HAVE_AVX512}")
message(STATUS "  CUDA Support: ${ENABLE_CUDA}")
message(STATUS "  Intel MKL: ${ENABLE_MKL}")
message(STATUS "  OpenMP: ${OpenMP_FOUND}")
message(STATUS "  Intel TBB: ${TBB_FOUND}")
message(STATUS "  MPI: ${MPI_FOUND}")
message(STATUS "")
message(STATUS "Memory Allocator:")
message(STATUS "  jemalloc: ${USE_JEMALLOC}")
message(STATUS "  mimalloc: ${USE_MIMALLOC}")
message(STATUS "  tcmalloc: ${USE_TCMALLOC}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Profiling: ${ENABLE_PROFILING}")
message(STATUS "  LTO: ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")
message(STATUS "")

# Performance warning if not using optimized build
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "")
    message(WARNING "⚠️  Building in ${CMAKE_BUILD_TYPE} mode!")
    message(WARNING "   For maximum performance, use: cmake -DCMAKE_BUILD_TYPE=Release")
    message(WARNING "")
endif()

if(NOT HAVE_AVX512)
    message(WARNING "")
    message(WARNING "⚠️  AVX-512 not detected!")
    message(WARNING "   Performance will be significantly reduced without SIMD support")
    message(WARNING "")
endif()