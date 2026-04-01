# Shared compile options for all pass plugins
set(PASS_COMPILE_OPTIONS
  -fno-rtti          # MLIR requires -fno-rtti for RTTI-less builds; adjust if needed
  -Wall
  -Wno-unused-parameter
)

# On macOS, MODULE libraries need -undefined dynamic_lookup so that
# MLIR symbols are resolved at load time from the host binary (mlir-opt),
# rather than being linked statically into the plugin (which would create
# a duplicate, private copy of the pass registry).
if(APPLE)
  set(PLUGIN_LINK_FLAGS "-undefined dynamic_lookup")
else()
  set(PLUGIN_LINK_FLAGS "")
endif()

# ---------------------------------------------------------------------------
# Exercise 1: No-op pass
# Usage: mlir-opt --load-pass-plugin=./libEx1NoopPass.so --ex1-noop input.mlir
# ---------------------------------------------------------------------------
add_library(Ex1NoopPass MODULE ex1_noop_pass.cpp)
target_compile_options(Ex1NoopPass PRIVATE ${PASS_COMPILE_OPTIONS})
target_link_libraries(Ex1NoopPass PRIVATE ${PLUGIN_LINK_FLAGS})

# ---------------------------------------------------------------------------
# Exercise 2: Counter pass
# Usage: mlir-opt --load-pass-plugin=./libEx2CounterPass.so --ex2-counter input.mlir
# ---------------------------------------------------------------------------
add_library(Ex2CounterPass MODULE ex2_counter_pass.cpp)
target_compile_options(Ex2CounterPass PRIVATE ${PASS_COMPILE_OPTIONS})
target_link_libraries(Ex2CounterPass PRIVATE ${PLUGIN_LINK_FLAGS})

# ---------------------------------------------------------------------------
# Exercise 3: Rewrite pass
# Usage: mlir-opt --load-pass-plugin=./libEx3RewritePass.so --ex3-rewrite input.mlir
# ---------------------------------------------------------------------------
add_library(Ex3RewritePass MODULE ex3_rewrite_pass.cpp)
target_compile_options(Ex3RewritePass PRIVATE ${PASS_COMPILE_OPTIONS})
target_link_libraries(Ex3RewritePass PRIVATE ${PLUGIN_LINK_FLAGS})
