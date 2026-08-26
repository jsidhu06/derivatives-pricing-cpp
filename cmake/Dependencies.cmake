# Third-party dependencies, fetched at configure time and pinned by tag.
#
# All dependencies are declared SYSTEM so their headers do not trigger our warning set or
# clang-tidy, and EXCLUDE_FROM_ALL so their own targets (docs, examples, tests) are not built.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# --- Eigen ------------------------------------------------------------------
# Eigen's own CMakeLists is slow to configure and probes for Fortran/BLAS. We only need the headers,
# so point SOURCE_SUBDIR at a non-existent directory to skip add_subdirectory() and define the
# interface target ourselves.
fetchcontent_declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake-skip-subdir SYSTEM
)
fetchcontent_makeavailable(eigen)
if(NOT TARGET Eigen3::Eigen)
    add_library(eigen INTERFACE)
    add_library(Eigen3::Eigen ALIAS eigen)
    target_include_directories(eigen SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
endif()

# --- Boost.Math (standalone) -----------------------------------------------
# Header-only, usable without the rest of Boost. Provides the normal distribution, erf/erfc, root
# finders (TOMS748, Newton) and quadrature.
fetchcontent_declare(
    boost_math
    GIT_REPOSITORY https://github.com/boostorg/math.git
    GIT_TAG boost-1.86.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR cmake-skip-subdir SYSTEM
)
fetchcontent_makeavailable(boost_math)
if(NOT TARGET Boost::math)
    add_library(boost_math INTERFACE)
    add_library(Boost::math ALIAS boost_math)
    target_include_directories(boost_math SYSTEM INTERFACE "${boost_math_SOURCE_DIR}/include")
    target_compile_definitions(boost_math INTERFACE BOOST_MATH_STANDALONE=1)
endif()

# --- spdlog -----------------------------------------------------------------
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
fetchcontent_declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.3
    GIT_SHALLOW TRUE
    SYSTEM EXCLUDE_FROM_ALL
)
fetchcontent_makeavailable(spdlog)

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
