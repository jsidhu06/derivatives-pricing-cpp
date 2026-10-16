# Install rules and the CMake package config, so an installed library is found with
# find_package(derivatives_pricing 0.1 REQUIRED) target_link_libraries(app PRIVATE
# dp::derivatives_pricing)
include(CMakePackageConfigHelpers)

set(DP_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/derivatives_pricing")

install(
    TARGETS derivatives_pricing
    EXPORT derivatives_pricing_targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/dp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
)
install(FILES "${PROJECT_BINARY_DIR}/generated/dp/version.hpp"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dp
)
install(
    EXPORT derivatives_pricing_targets
    NAMESPACE dp::
    FILE derivatives_pricingTargets.cmake
    DESTINATION ${DP_INSTALL_CMAKEDIR}
)

configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/derivatives_pricingConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/derivatives_pricingConfig.cmake"
    INSTALL_DESTINATION ${DP_INSTALL_CMAKEDIR}
)
# Pre-1.0, a minor release may break the API, so only the same minor version is compatible.
write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/derivatives_pricingConfigVersion.cmake" COMPATIBILITY SameMinorVersion
)
install(FILES "${PROJECT_BINARY_DIR}/derivatives_pricingConfig.cmake"
              "${PROJECT_BINARY_DIR}/derivatives_pricingConfigVersion.cmake"
        DESTINATION ${DP_INSTALL_CMAKEDIR}
)

if(DP_EIGEN_SOURCE MATCHES "^fetched")
    message(STATUS "Eigen was fetched: consumers of the installed package need Eigen 3.4 installed "
                   "where find_package(Eigen3) can find it"
    )
endif()
