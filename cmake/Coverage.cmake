# Instrument a target for line coverage when DP_ENABLE_COVERAGE is ON (GCC or Clang, gcov format).
#
# The flags are attached PUBLIC so anything linking the library (tests) is instrumented too: inline
# code in the public headers is then counted wherever it is called from, and the test executable
# links the coverage runtime. Use an unoptimised build, or lines are merged and reordered.
function(dp_enable_coverage target)
    if(NOT DP_ENABLE_COVERAGE)
        return()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(WARNING "DP_ENABLE_COVERAGE is only supported with GCC and Clang")
        return()
    endif()

    target_compile_options(${target} PUBLIC --coverage -O0)
    target_link_options(${target} PUBLIC --coverage)
endfunction()
