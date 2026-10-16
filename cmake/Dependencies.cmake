# Third-party dependencies, fetched at configure time and pinned by tag.
#
# All dependencies are declared SYSTEM so their headers do not trigger our warning set or
# clang-tidy, and EXCLUDE_FROM_ALL so their own targets (docs, examples, tests) are not built.
#
# Eigen is the only public dependency (it appears in the public headers): an installed Eigen 3.4 is
# used when find_package() finds one, so a consumer that already uses Eigen links one copy.
# Boost.Math, PCG and spdlog are private and header-only; they are compiled into the library and
# consumers never see them.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- Eigen ------------------------------------------------------------------
# find_package(Eigen3 3.4) first; otherwise fetch. Eigen's own CMakeLists is slow to configure and
# probes for Fortran/BLAS. We only need the headers, so point SOURCE_SUBDIR at a non-existent
# directory to skip add_subdirectory() and define the target ourselves. The fallback is an IMPORTED
# target so the installed package refers to it by name, Eigen3::Eigen, and the package config finds
# it again with find_dependency(Eigen3).
fetchcontent_declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake-skip-subdir
    SYSTEM
    FIND_PACKAGE_ARGS 3.4 NAMES Eigen3
)
fetchcontent_makeavailable(eigen)
if(TARGET Eigen3::Eigen)
    set(DP_EIGEN_SOURCE "installed (${Eigen3_DIR})")
else()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(
        Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
    set(DP_EIGEN_SOURCE "fetched (3.4.0)")
endif()

# --- Boost.Math (standalone) -----------------------------------------------
# Header-only, usable without the rest of Boost. Provides the normal distribution, erf/erfc, root
# finders (TOMS748, Newton) and quadrature.
fetchcontent_declare(
    boost_math
    GIT_REPOSITORY https://github.com/boostorg/math.git
    GIT_TAG boost-1.86.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake-skip-subdir
    SYSTEM
)
fetchcontent_makeavailable(boost_math)
if(NOT TARGET Boost::math)
    add_library(boost_math INTERFACE)
    add_library(Boost::math ALIAS boost_math)
    target_include_directories(boost_math SYSTEM INTERFACE "${boost_math_SOURCE_DIR}/include")
    target_compile_definitions(boost_math INTERFACE BOOST_MATH_STANDALONE=1)
endif()

# --- PCG (random number engine) ----------------------------------------------
# Header-only. Same generator family as numpy's default BitGenerator; streams are NOT
# numpy-compatible (by design — MC agreement with the Python reference is statistical, never
# bitwise).
fetchcontent_declare(
    pcg_cpp
    GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
    GIT_TAG 428802d1a5634f96bcd0705fab379ff0113bcf13
    SOURCE_SUBDIR cmake-skip-subdir
    SYSTEM
)
fetchcontent_makeavailable(pcg_cpp)
if(NOT TARGET pcg::pcg)
    add_library(pcg_cpp_headers INTERFACE)
    add_library(pcg::pcg ALIAS pcg_cpp_headers)
    target_include_directories(pcg_cpp_headers SYSTEM INTERFACE "${pcg_cpp_SOURCE_DIR}/include")
endif()

# --- spdlog -----------------------------------------------------------------
# Header-only (with its bundled fmt), so the static library carries everything it needs and an
# installed package has no spdlog link dependency. Only the default warning handler uses it.
fetchcontent_declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake-skip-subdir
    SYSTEM
)
fetchcontent_makeavailable(spdlog)
if(NOT TARGET dp_spdlog_headers)
    add_library(dp_spdlog_headers INTERFACE)
    target_include_directories(dp_spdlog_headers SYSTEM INTERFACE "${spdlog_SOURCE_DIR}/include")
endif()
find_package(Threads REQUIRED)

# --- Catch2 (tests only) ----------------------------------------------------
if(DP_BUILD_TESTS)
    set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
    set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)
    fetchcontent_declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1
        GIT_SHALLOW TRUE
        SYSTEM EXCLUDE_FROM_ALL
    )
    fetchcontent_makeavailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()

# --- Google Benchmark (benchmarks only) --------------------------------------
if(DP_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
    fetchcontent_declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
        GIT_SHALLOW TRUE
        SYSTEM EXCLUDE_FROM_ALL
    )
    fetchcontent_makeavailable(benchmark)
endif()
