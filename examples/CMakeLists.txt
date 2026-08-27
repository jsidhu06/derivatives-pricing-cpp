# Small, self-contained programs demonstrating the public API. Each .cpp file is one executable
# named after the file.
set(dp_examples quick_start)

foreach(example IN LISTS dp_examples)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE dp::derivatives_pricing)
    dp_set_project_warnings(${example})
    dp_enable_clang_tidy(${example})
endforeach()
