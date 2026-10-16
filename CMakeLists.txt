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

# Only as the top-level project: a consumer's build type is the consumer's choice.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(DP_DEFAULT_BUILD_TESTS ON)
else()
    set(DP_DEFAULT_BUILD_TESTS OFF)
endif()

option(DP_BUILD_TESTS "Build the unit tests" ${DP_DEFAULT_BUILD_TESTS})
option(DP_BUILD_EXAMPLES "Build the example programs (examples/)" ${DP_DEFAULT_BUILD_TESTS})
option(DP_BUILD_BENCHMARKS "Build the Google Benchmark suite (bench/)" OFF)
option(DP_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(DP_ENABLE_SANITIZERS "Enable the sanitizers selected by DP_SANITIZER" OFF)
set(DP_SANITIZER "address,undefined"
    CACHE STRING "Sanitizers for DP_ENABLE_SANITIZERS: address,undefined or thread"
)
set_property(CACHE DP_SANITIZER PROPERTY STRINGS "address,undefined" "thread")
option(DP_ENABLE_COVERAGE "Instrument the library and tests for line coverage (gcov)" OFF)
option(DP_ENABLE_CLANG_TIDY "Run clang-tidy as part of the build" OFF)
option(DP_ENABLE_LTO "Enable link-time optimisation for Release builds" OFF)
option(DP_INSTALL "Generate install rules and the CMake package config" ${PROJECT_IS_TOP_LEVEL})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)
include(Coverage)
include(StaticAnalysis)
include(Dependencies)
include(GNUInstallDirs)

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
            src/engines/barrier_analytical.cpp
            src/engines/mc.cpp
            src/engines/pde.cpp
            src/engines/pde_barrier.cpp
            src/engines/pde_numerics.cpp
            src/enums.cpp
            src/forward.cpp
            src/implied_volatility.cpp
            src/market_data.cpp
            src/math/normal.cpp
            src/params.cpp
            src/simulation.cpp
            src/underlying.cpp
            src/valuation.cpp
)

target_include_directories(
    derivatives_pricing
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Eigen is in the public headers. The header-only private dependencies are compiled into the
# library, so they are build-only (BUILD_INTERFACE) and never reach an installed consumer.
target_link_libraries(
    derivatives_pricing PUBLIC Eigen3::Eigen
    PRIVATE $<BUILD_INTERFACE:Boost::math> $<BUILD_INTERFACE:pcg::pcg>
            $<BUILD_INTERFACE:dp_spdlog_headers> Threads::Threads
)

target_compile_features(derivatives_pricing PUBLIC cxx_std_20)
dp_set_project_warnings(derivatives_pricing)
dp_enable_sanitizers(derivatives_pricing)
dp_enable_coverage(derivatives_pricing)
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
# Install and package config (find_package(derivatives_pricing))
# ---------------------------------------------------------------------------
if(DP_INSTALL)
    include(Install)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(DP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(DP_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(DP_BUILD_BENCHMARKS)
    add_subdirectory(bench)
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
message(STATUS "  Eigen:             ${DP_EIGEN_SOURCE}")
message(STATUS "  Install rules:     ${DP_INSTALL}")
message(STATUS "  Compiler:          ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests:             ${DP_BUILD_TESTS}")
message(STATUS "  Examples:          ${DP_BUILD_EXAMPLES}")
message(STATUS "  Benchmarks:        ${DP_BUILD_BENCHMARKS}")
if(DP_ENABLE_SANITIZERS)
    message(STATUS "  Sanitizers:        ${DP_SANITIZER}")
else()
    message(STATUS "  Sanitizers:        OFF")
endif()
message(STATUS "  Coverage:          ${DP_ENABLE_COVERAGE}")
message(STATUS "  clang-tidy:        ${DP_ENABLE_CLANG_TIDY}")
message(STATUS "  Warnings as errors ${DP_WARNINGS_AS_ERRORS}")
message(STATUS "")
