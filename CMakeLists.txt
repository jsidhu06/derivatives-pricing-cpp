cmake_minimum_required(VERSION 3.25)

project(
    derivatives_pricing
    VERSION 0.1.0
    DESCRIPTION "C++20 derivatives pricing library: analytical, tree, PDE and Monte Carlo engines"
    HOMEPAGE_URL "https://github.com/jsidhu06/derivatives-pricing-cpp"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(DP_DEFAULT_BUILD_TESTS ON)
else()
    set(DP_DEFAULT_BUILD_TESTS OFF)
endif()

option(DP_BUILD_TESTS "Build the unit tests" ${DP_DEFAULT_BUILD_TESTS})
option(DP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(DP_ENABLE_SANITIZERS "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(DP_ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" OFF)
option(DP_ENABLE_LTO "Enable link-time optimisation for Release builds" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)
include(StaticAnalysis)
include(Dependencies)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(derivatives_pricing STATIC)
add_library(dp::derivatives_pricing ALIAS derivatives_pricing)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/dp/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/dp/version.hpp" @ONLY
)

target_sources(
    derivatives_pricing
    PRIVATE src/contracts.cpp
            src/dates.cpp
            src/diagnostics.cpp
            src/discount_curve.cpp
            src/dividends.cpp
            src/engines/binomial.cpp
            src/engines/bsm.cpp
            src/enums.cpp
            src/forward.cpp
            src/market_data.cpp
            src/math/normal.cpp
            src/params.cpp
            src/underlying.cpp
            src/valuation.cpp
)

target_include_directories(
    derivatives_pricing PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                               $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(derivatives_pricing PUBLIC Eigen3::Eigen Boost::math PRIVATE spdlog::spdlog)

target_compile_features(derivatives_pricing PUBLIC cxx_std_20)
dp_set_project_warnings(derivatives_pricing)
dp_enable_sanitizers(derivatives_pricing)
dp_enable_clang_tidy(derivatives_pricing)

if(DP_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT dp_ipo_supported OUTPUT dp_ipo_message)
    if(dp_ipo_supported)
        set_property(TARGET derivatives_pricing PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    else()
        message(WARNING "LTO requested but not supported: ${dp_ipo_message}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Personal scratch programs (untracked; see practice/CMakeLists.txt)
# ---------------------------------------------------------------------------
if(PROJECT_IS_TOP_LEVEL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/practice/CMakeLists.txt")
    add_subdirectory(practice)
endif()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "derivatives_pricing ${PROJECT_VERSION}")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests:             ${DP_BUILD_TESTS}")
message(STATUS "  Sanitizers:        ${DP_ENABLE_SANITIZERS}")
message(STATUS "  clang-tidy:        ${DP_ENABLE_CLANG_TIDY}")
message(STATUS "  Warnings as errors ${DP_WARNINGS_AS_ERRORS}")
message(STATUS "")
