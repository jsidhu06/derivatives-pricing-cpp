# Enable sanitizers on a target when DP_ENABLE_SANITIZERS is ON. DP_SANITIZER picks the set:
# "address,undefined" (ASan + UBSan, the default) or "thread" (TSan, which cannot be combined with
# ASan).
#
# The flags are attached PUBLIC so anything linking the library (tests) is instrumented consistently
# — mixing instrumented and non-instrumented objects produces spurious reports.
function(dp_enable_sanitizers target)
    if(NOT DP_ENABLE_SANITIZERS)
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(NOT DP_SANITIZER STREQUAL "address,undefined")
            message(FATAL_ERROR "DP_SANITIZER=${DP_SANITIZER} is not supported with MSVC")
        endif()
        target_compile_options(${target} PUBLIC /fsanitize=address)
        return()
    endif()

    if(DP_SANITIZER STREQUAL "address,undefined")
        set(sanitizer_flags -fsanitize=address,undefined -fno-omit-frame-pointer
                            -fno-sanitize-recover=undefined
        )
    elseif(DP_SANITIZER STREQUAL "thread")
        set(sanitizer_flags -fsanitize=thread -fno-omit-frame-pointer)
    else()
        message(FATAL_ERROR "DP_SANITIZER must be \"address,undefined\" or \"thread\", got "
                            "\"${DP_SANITIZER}\""
        )
    endif()
    target_compile_options(${target} PUBLIC ${sanitizer_flags})
    target_link_options(${target} PUBLIC ${sanitizer_flags})
    message(STATUS "Sanitizers enabled on ${target}: ${DP_SANITIZER}")
endfunction()
