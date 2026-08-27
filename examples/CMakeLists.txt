# Small, self-contained programs demonstrating the public API. Every .cpp file in this directory
# becomes one executable named after the file, so scratch experiments can just be dropped here and
# built with cmake --build --preset debug --target <name> && build/debug/examples/<name>
file(GLOB dp_example_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(source IN LISTS dp_example_sources)
    get_filename_component(example "${source}" NAME_WE)
    add_executable(${example} "${source}")
    target_link_libraries(${example} PRIVATE dp::derivatives_pricing)
    dp_set_project_warnings(${example})
    dp_enable_clang_tidy(${example})
endforeach()
