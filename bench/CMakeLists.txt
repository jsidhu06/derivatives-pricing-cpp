add_executable(dp_bench bench_pricing.cpp)
target_link_libraries(dp_bench PRIVATE dp::derivatives_pricing benchmark::benchmark)
dp_set_project_warnings(dp_bench)
dp_enable_clang_tidy(dp_bench)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Benchmarks built with CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}: "
                    "timings are only meaningful in Release (use the bench preset)."
    )
endif()
