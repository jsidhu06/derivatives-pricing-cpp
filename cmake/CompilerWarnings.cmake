# Attach a strict, compiler-specific warning set to a target.
#
# Usage: dp_set_project_warnings(<target>)
function(dp_set_project_warnings target)
    set(gcc_clang_warning_flags
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
    )

    set(gcc_only_warning_flags -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches
                               -Wlogical-op -Wuseless-cast
    )

    set(win_warning_flags
        /W4
        /permissive-
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /we4289
        /w14296
        /w14311
        /w14545
        /w14546
        /w14547
        /w14549
        /w14555
        /w14619
        /w14640
        /w14826
        /w14905
        /w14906
        /w14928
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(warnings ${win_warning_flags})
        if(DP_WARNINGS_AS_ERRORS)
            list(APPEND warnings /WX)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(warnings ${gcc_clang_warning_flags})
        if(DP_WARNINGS_AS_ERRORS)
            list(APPEND warnings -Werror)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(warnings ${gcc_clang_warning_flags} ${gcc_only_warning_flags})
        if(DP_WARNINGS_AS_ERRORS)
            list(APPEND warnings -Werror)
        endif()
    else()
        message(AUTHOR_WARNING "No warning set configured for compiler '${CMAKE_CXX_COMPILER_ID}'")
    endif()

    target_compile_options(${target} PRIVATE ${warnings})
endfunction()
