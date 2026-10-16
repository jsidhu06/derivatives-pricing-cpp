# A minimal consumer of the library, used by CI to check both ways of using it:
#
# installed package:  cmake -S tests/package -B build/package -DCMAKE_PREFIX_PATH=<prefix>
# FetchContent:       cmake -S tests/package -B build/package -DDP_CONSUME=fetchcontent
# -DDP_SOURCE_DIR=<checkout>
cmake_minimum_required(VERSION 3.25)
project(dp_package_test LANGUAGES CXX)

set(DP_CONSUME "find_package" CACHE STRING
                                    "How to consume the library: find_package or fetchcontent"
)

if(DP_CONSUME STREQUAL "find_package")
    find_package(derivatives_pricing 0.1 REQUIRED)
elseif(DP_CONSUME STREQUAL "fetchcontent")
    include(FetchContent)
    # SOURCE_DIR stands in for the GIT_REPOSITORY / GIT_TAG pair a real consumer would use.
    fetchcontent_declare(derivatives_pricing SOURCE_DIR "${DP_SOURCE_DIR}")
    fetchcontent_makeavailable(derivatives_pricing)
else()
    message(FATAL_ERROR "DP_CONSUME must be find_package or fetchcontent, got '${DP_CONSUME}'")
endif()

add_executable(dp_package_test main.cpp)
target_link_libraries(dp_package_test PRIVATE dp::derivatives_pricing)
