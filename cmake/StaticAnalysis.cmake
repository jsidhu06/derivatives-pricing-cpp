# clang-tidy integration, enabled per target when DP_ENABLE_CLANG_TIDY is ON.
#
# A per-target property is used rather than the global CMAKE_CXX_CLANG_TIDY so third-party sources
# built via FetchContent (spdlog, Catch2) are not analysed. HeaderFilterRegex in .clang-tidy limits
# header diagnostics to our own tree.
if(DP_ENABLE_CLANG_TIDY)
    find_program(DP_CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-18 clang-tidy-17)
    if(DP_CLANG_TIDY_EXE)
        message(STATUS "clang-tidy: ${DP_CLANG_TIDY_EXE}")
    else()
        message(WARNING "DP_ENABLE_CLANG_TIDY is ON but clang-tidy was not found")
    endif()
endif()

# Usage: dp_enable_clang_tidy(<target>)
function(dp_enable_clang_tidy target)
    if(DP_ENABLE_CLANG_TIDY AND DP_CLANG_TIDY_EXE)
        set(tidy_command "${DP_CLANG_TIDY_EXE}" -extra-arg=-Wno-unknown-warning-option)
        if(DP_WARNINGS_AS_ERRORS)
            list(APPEND tidy_command "-warnings-as-errors=*")
        endif()
        set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${tidy_command}")
    endif()
endfunction()
