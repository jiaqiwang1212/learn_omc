cmake_minimum_required(VERSION 3.20)
project(LearnOMCPasses CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default MLIR_DIR to our build directory
set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../build/lib/cmake/mlir"
    CACHE PATH "Path to MLIRConfig.cmake (default: ../build/lib/cmake/mlir)")

find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
message(STATUS "Found MLIR in: ${MLIR_DIR}")
message(STATUS "MLIR version: ${MLIR_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddLLVM)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(phase2)
