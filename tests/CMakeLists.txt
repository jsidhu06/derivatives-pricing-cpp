include(Catch)

add_executable(
    dp_tests
    test_barrier_papers.cpp
    test_contracts.cpp
    test_dates.cpp
    test_diagnostics.cpp
    test_discount_curve.cpp
    test_dividends_forward.cpp
    test_enums.cpp
    test_implied_volatility.cpp
    test_market_data.cpp
    test_normal.cpp
    test_params.cpp
    test_simulation_processes.cpp
    test_underlying.cpp
    test_valuation_barrier.cpp
    test_valuation_barrier_binomial.cpp
    test_valuation_barrier_mc.cpp
    test_valuation_barrier_pde.cpp
    test_valuation_barrier_triggered.cpp
    test_valuation_binomial.cpp
    test_valuation_bsm.cpp
    test_valuation_conventions.cpp
    test_valuation_facade.cpp
    test_valuation_mc.cpp
    test_valuation_pde.cpp
)

target_link_libraries(dp_tests PRIVATE dp::derivatives_pricing Catch2::Catch2WithMain)
target_include_directories(dp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
dp_set_project_warnings(dp_tests)
dp_enable_clang_tidy(dp_tests)

# Tags become ctest labels so presets can filter on them (e.g. [slow]).
catch_discover_tests(dp_tests ADD_TAGS_AS_LABELS)
