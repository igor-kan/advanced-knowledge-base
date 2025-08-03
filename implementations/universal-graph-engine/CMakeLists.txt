cmake_minimum_required(VERSION 3.15)

# Universal Graph Engine - The Ultimate Graph Database
project(UniversalGraphEngine 
    VERSION 1.0.0
    DESCRIPTION "The most complex and flexible graph database ever conceived"
    LANGUAGES C CXX
)

# ============================================================================
# Project Configuration
# ============================================================================

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add project root to include path
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================================
# Build Options
# ============================================================================

option(UG_BUILD_SHARED "Build shared library" ON)
option(UG_BUILD_STATIC "Build static library" ON)
option(UG_BUILD_CLI "Build command line interface" ON)
option(UG_BUILD_TESTS "Build test suite" ON)
option(UG_BUILD_EXAMPLES "Build examples" ON)
option(UG_BUILD_BENCHMARKS "Build benchmarks" OFF)

# Advanced features
option(UG_ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(UG_ENABLE_OPENCL "Enable OpenCL acceleration" OFF)
option(UG_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(UG_ENABLE_QUANTUM "Enable quantum graph features" OFF)
option(UG_ENABLE_TEMPORAL "Enable temporal features" ON)
option(UG_ENABLE_STREAMING "Enable streaming features" ON)
option(UG_ENABLE_DISTRIBUTED "Enable distributed features" OFF)
option(UG_ENABLE_GENETIC_ALGORITHMS "Enable genetic algorithms" OFF)
option(UG_ENABLE_NEUROMORPHIC "Enable neuromorphic computing" OFF)

# Language bindings
option(UG_BUILD_CPP_BINDINGS "Build C++ bindings" ON)
option(UG_BUILD_RUST_BINDINGS "Build Rust bindings" OFF)
option(UG_BUILD_GO_BINDINGS "Build Go bindings" OFF)
option(UG_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

# Development options
option(UG_ENABLE_PROFILING "Enable profiling support" OFF)
option(UG_ENABLE_DEBUGGING "Enable debugging support" OFF)
option(UG_ENABLE_SANITIZERS "Enable address/memory sanitizers" OFF)

# ============================================================================
# Platform Detection
# ============================================================================

if(WIN32)
    add_definitions(-DUG_PLATFORM_WINDOWS)
    set(UG_PLATFORM_LIBS "")
elseif(APPLE)
    add_definitions(-DUG_PLATFORM_MACOS)
    set(UG_PLATFORM_LIBS pthread)
elseif(UNIX)
    add_definitions(-DUG_PLATFORM_LINUX)
    set(UG_PLATFORM_LIBS pthread rt)
endif()

# ============================================================================
# Compiler Flags
# ============================================================================

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(UG_C_FLAGS
        -Wall -Wextra -Wpedantic
        -Wformat=2 -Wformat-security
        -Wnull-dereference
        -Wstack-protector
        -Wvla
        -Warray-bounds
        -Warray-bounds-pointer-arithmetic
        -Wconditional-uninitialized
        -Wloop-analysis
        -Wshift-sign-overflow
        -Wswitch-enum
        -Wthread-safety-analysis
        -Wunused-variable
        -Wunused-parameter
        -Wunused-function
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        list(APPEND UG_C_FLAGS
            -O3 -DNDEBUG
            -march=native
            -ffast-math
            -funroll-loops
            -finline-functions
            -flto
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND UG_C_FLAGS
            -O0 -g3
            -DDEBUG
            -fno-omit-frame-pointer
        )
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(UG_CXX_FLAGS ${UG_C_FLAGS}
        -Woverloaded-virtual
        -Wnon-virtual-dtor
        -Wold-style-cast
    )
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Find required packages
find_package(Threads REQUIRED)

# Optional dependencies
if(UG_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        list(APPEND UG_PLATFORM_LIBS ${OpenMP_C_LIBRARIES})
        add_definitions(-DUG_ENABLE_OPENMP)
    endif()
endif()

if(UG_ENABLE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        enable_language(CUDA)
        add_definitions(-DUG_ENABLE_CUDA)
    endif()
endif()

if(UG_ENABLE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        add_definitions(-DUG_ENABLE_OPENCL)
        list(APPEND UG_PLATFORM_LIBS ${OpenCL_LIBRARIES})
        include_directories(${OpenCL_INCLUDE_DIRS})
    endif()
endif()

# ============================================================================
# Source Files
# ============================================================================

set(UG_CORE_SOURCES
    src/universal_graph.c
    src/universal_value.c
    src/universal_node.c
    src/universal_relationship.c
    src/universal_property.c
    src/universal_query.c
    src/universal_traversal.c
    src/universal_export.c
    src/universal_memory.c
    src/universal_hash.c
)

set(UG_ADVANCED_SOURCES
    src/universal_temporal.c
    src/universal_quantum.c
    src/universal_streaming.c
    src/universal_hypergraph.c
    src/universal_meta.c
    src/universal_causal.c
    src/universal_evolution.c
)

set(UG_HEADERS
    include/universal_graph.h
    include/universal_types.h
    include/universal_config.h
    include/universal_error.h
)

# Create source lists based on enabled features
set(UG_ALL_SOURCES ${UG_CORE_SOURCES})

if(UG_ENABLE_TEMPORAL)
    list(APPEND UG_ALL_SOURCES src/universal_temporal.c)
    add_definitions(-DUG_ENABLE_TEMPORAL)
endif()

if(UG_ENABLE_QUANTUM)
    list(APPEND UG_ALL_SOURCES src/universal_quantum.c)
    add_definitions(-DUG_ENABLE_QUANTUM)
endif()

if(UG_ENABLE_STREAMING)
    list(APPEND UG_ALL_SOURCES src/universal_streaming.c)
    add_definitions(-DUG_ENABLE_STREAMING)
endif()

if(UG_ENABLE_DISTRIBUTED)
    list(APPEND UG_ALL_SOURCES src/universal_distributed.c)
    add_definitions(-DUG_ENABLE_DISTRIBUTED)
endif()

if(UG_ENABLE_GENETIC_ALGORITHMS)
    list(APPEND UG_ALL_SOURCES src/universal_evolution.c)
    add_definitions(-DUG_ENABLE_GENETIC_ALGORITHMS)
endif()

# ============================================================================
# Core Library
# ============================================================================

# Shared library
if(UG_BUILD_SHARED)
    add_library(universal_graph SHARED ${UG_ALL_SOURCES})
    target_compile_options(universal_graph PRIVATE ${UG_C_FLAGS})
    target_link_libraries(universal_graph ${UG_PLATFORM_LIBS} Threads::Threads)
    
    set_target_properties(universal_graph PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
        PUBLIC_HEADER "${UG_HEADERS}"
    )
    
    # Export symbols
    if(WIN32)
        target_compile_definitions(universal_graph PRIVATE UG_BUILDING_DLL)
        target_compile_definitions(universal_graph INTERFACE UG_USING_DLL)
    endif()
endif()

# Static library
if(UG_BUILD_STATIC)
    add_library(universal_graph_static STATIC ${UG_ALL_SOURCES})
    target_compile_options(universal_graph_static PRIVATE ${UG_C_FLAGS})
    target_link_libraries(universal_graph_static ${UG_PLATFORM_LIBS} Threads::Threads)
    
    set_target_properties(universal_graph_static PROPERTIES
        OUTPUT_NAME universal_graph
        PUBLIC_HEADER "${UG_HEADERS}"
    )
endif()

# ============================================================================
# C++ Bindings
# ============================================================================

if(UG_BUILD_CPP_BINDINGS)
    add_library(universal_graph_cpp INTERFACE)
    target_include_directories(universal_graph_cpp INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp>
        $<INSTALL_INTERFACE:include/universal_graph>
    )
    
    if(UG_BUILD_SHARED)
        target_link_libraries(universal_graph_cpp INTERFACE universal_graph)
    else()
        target_link_libraries(universal_graph_cpp INTERFACE universal_graph_static)
    endif()
    
    # Install C++ headers
    install(FILES bindings/cpp/universal_graph.hpp
        DESTINATION include/universal_graph
    )
endif()

# ============================================================================
# CLI Application
# ============================================================================

if(UG_BUILD_CLI)
    # CLI sources
    set(UG_CLI_SOURCES
        src/cli/main.c
        src/cli/ug_cli.c
    )
    
    # Create CLI executable
    add_executable(ug_cli ${UG_CLI_SOURCES})
    target_include_directories(ug_cli PRIVATE src/cli)
    target_compile_options(ug_cli PRIVATE ${UG_C_FLAGS})
    
    # Link against universal graph library
    if(UG_BUILD_SHARED)
        target_link_libraries(ug_cli universal_graph ${UG_PLATFORM_LIBS} Threads::Threads)
    else()
        target_link_libraries(ug_cli universal_graph_static ${UG_PLATFORM_LIBS} Threads::Threads)
    endif()
    
    # Install CLI executable
    install(TARGETS ug_cli
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# ============================================================================
# Examples
# ============================================================================

if(UG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ============================================================================
# Tests
# ============================================================================

if(UG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(UG_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install libraries
if(UG_BUILD_SHARED)
    install(TARGETS universal_graph
        EXPORT UniversalGraphEngineTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/universal_graph
    )
endif()

if(UG_BUILD_STATIC)
    install(TARGETS universal_graph_static
        EXPORT UniversalGraphEngineTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/universal_graph
    )
endif()

if(UG_BUILD_CPP_BINDINGS)
    install(TARGETS universal_graph_cpp
        EXPORT UniversalGraphEngineTargets
    )
endif()

# Install headers
install(FILES ${UG_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/universal_graph
)

# Create and install CMake config files
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/UniversalGraphEngineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT UniversalGraphEngineTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/UniversalGraphEngineTargets.cmake"
    NAMESPACE UniversalGraphEngine::
)

configure_file(cmake/UniversalGraphEngineConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/UniversalGraphEngineConfig.cmake"
    COPYONLY
)

install(EXPORT UniversalGraphEngineTargets
    FILE UniversalGraphEngineTargets.cmake
    NAMESPACE UniversalGraphEngine::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/UniversalGraphEngine
)

install(FILES
    cmake/UniversalGraphEngineConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/UniversalGraphEngineConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/UniversalGraphEngine
)

# ============================================================================
# Package Configuration
# ============================================================================

set(CPACK_PACKAGE_NAME "UniversalGraphEngine")
set(CPACK_PACKAGE_VENDOR "Universal Graph Engine Project")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The most complex and flexible graph database ever conceived")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "UniversalGraphEngine")

if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop;TGZ")
else()
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
endif()

include(CPack)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Universal Graph Engine Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Libraries:")
message(STATUS "  Shared: ${UG_BUILD_SHARED}")
message(STATUS "  Static: ${UG_BUILD_STATIC}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  CUDA: ${UG_ENABLE_CUDA}")
message(STATUS "  OpenCL: ${UG_ENABLE_OPENCL}")
message(STATUS "  OpenMP: ${UG_ENABLE_OPENMP}")
message(STATUS "  Quantum: ${UG_ENABLE_QUANTUM}")
message(STATUS "  Temporal: ${UG_ENABLE_TEMPORAL}")
message(STATUS "  Streaming: ${UG_ENABLE_STREAMING}")
message(STATUS "  Distributed: ${UG_ENABLE_DISTRIBUTED}")
message(STATUS "  Genetic Algorithms: ${UG_ENABLE_GENETIC_ALGORITHMS}")
message(STATUS "")
message(STATUS "Language Bindings:")
message(STATUS "  C++: ${UG_BUILD_CPP_BINDINGS}")
message(STATUS "  Rust: ${UG_BUILD_RUST_BINDINGS}")
message(STATUS "  Go: ${UG_BUILD_GO_BINDINGS}")
message(STATUS "  Python: ${UG_BUILD_PYTHON_BINDINGS}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  CLI: ${UG_BUILD_CLI}")
message(STATUS "  Tests: ${UG_BUILD_TESTS}")
message(STATUS "  Examples: ${UG_BUILD_EXAMPLES}")
message(STATUS "  Benchmarks: ${UG_BUILD_BENCHMARKS}")
message(STATUS "  Profiling: ${UG_ENABLE_PROFILING}")
message(STATUS "  Debugging: ${UG_ENABLE_DEBUGGING}")
message(STATUS "  Sanitizers: ${UG_ENABLE_SANITIZERS}")
message(STATUS "")