# Example programs. Each is a single source file; the executable is named dp_example_<name>.
set(DP_EXAMPLES barrier_options implied_volatility monte_carlo vanilla_engines)

foreach(example IN LISTS DP_EXAMPLES)
    set(target dp_example_${example})
    add_executable(${target} ${example}.cpp)
    target_link_libraries(${target} PRIVATE dp::derivatives_pricing)
    dp_set_project_warnings(${target})
    dp_enable_clang_tidy(${target})
endforeach()
