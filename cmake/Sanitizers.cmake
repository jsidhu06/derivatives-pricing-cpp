# Enable ASan + UBSan on a target when DP_ENABLE_SANITIZERS is ON.
#
# The flags are attached PUBLIC so anything linking the library (tests) is instrumented consistently
# — mixing instrumented and non-instrumented objects produces spurious reports.
function(dp_enable_sanitizers target)
    if(NOT DP_ENABLE_SANITIZERS)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} PUBLIC /fsanitize=address)
        return()
    endif()

    set(sanitizer_flags -fsanitize=address,undefined -fno-omit-frame-pointer
                        -fno-sanitize-recover=undefined
    )
    target_compile_options(${target} PUBLIC ${sanitizer_flags})
    target_link_options(${target} PUBLIC ${sanitizer_flags})
    message(STATUS "Sanitizers enabled on ${target}: address, undefined")
endfunction()
