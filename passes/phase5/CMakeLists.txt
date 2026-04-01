# Shared compile options — same as phase2
set(PASS_COMPILE_OPTIONS
  -fno-rtti
  -Wall
  -Wno-unused-parameter
)

if(APPLE)
  set(PLUGIN_LINK_FLAGS "-undefined dynamic_lookup")
else()
  set(PLUGIN_LINK_FLAGS "")
endif()

# Phase 5 exercise 1: Define myml dialect + register it as a pass plugin
# Usage: mlir-opt --load-pass-plugin=./libEx1DefineDialect.so --ex1-load-myml input.mlir
add_library(Ex1DefineDialect MODULE ex1_define_dialect.cpp)
target_compile_options(Ex1DefineDialect PRIVATE ${PASS_COMPILE_OPTIONS})
target_link_libraries(Ex1DefineDialect PRIVATE ${PLUGIN_LINK_FLAGS})
target_include_directories(Ex1DefineDialect PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Phase 5 exercise 2: Lower myml ops to linalg.generic
# Usage: mlir-opt --load-pass-plugin=./libEx2LowerToLinalg.so --ex2-lower-myml-to-linalg input.mlir
add_library(Ex2LowerToLinalg MODULE ex2_lower_to_linalg.cpp)
target_compile_options(Ex2LowerToLinalg PRIVATE ${PASS_COMPILE_OPTIONS})
target_link_libraries(Ex2LowerToLinalg PRIVATE ${PLUGIN_LINK_FLAGS})
target_include_directories(Ex2LowerToLinalg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
