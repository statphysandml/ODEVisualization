# Define the library
add_library(odevisualization STATIC
    collection/buffer.cpp
    collection/collection.cpp
    collection/collection_expander.cu
    collection/leaf.cpp
    recursive_search/dynamic_recursive_grid_computation.cu
    recursive_search/static_recursive_grid_computation.cu
    recursive_search/fixed_point_criterion.cu
    grid_computation/grid_computation.cu
    grid_computation/grid_computation_wrapper.cu
    evolution/evolution_observer.cu
    evolution/stepper.cu
    util/monitor.cu
    util/kmeans.cpp
    util/partial_ranges.cpp
    util/random.cu
    modes/evolution.cu
    modes/jacobians.cpp
    modes/jacobians.cu
    modes/kmeans_clustering.cu
    modes/mesh.cu
    modes/recursive_search.cu
    modes/separatrizes.cu
    boost/dev_dat_boost_integration.cu
)

# CUDA specific options including explicit JSON include path
target_compile_options(odevisualization PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
        $<$<BOOL:${JSON_INCLUDE_DIR}>:-I${JSON_INCLUDE_DIR}>
    >
)

# Include directories - make sure to include JSON paths for CUDA
target_include_directories(odevisualization
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${Boost_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        # Explicitly add JSON include directory for CUDA compilation
        $<$<BOOL:${JSON_INCLUDE_DIR}>:${JSON_INCLUDE_DIR}>
)

# Dependencies
target_link_libraries(odevisualization
    PUBLIC
        paramhelper::paramhelper
        flowequations::flowequations
        devdat::devdat
        ${Boost_LIBRARIES}
        #eigen_wrapper   # Updated to use our renamed Eigen target
)

# Add Eigen include path explicitly if provided
if(DEFINED EIGEN_INCLUDE_DIR)
    target_include_directories(odevisualization PUBLIC ${EIGEN_INCLUDE_DIR})
endif()

# Set properties for CUDA
set_target_properties(odevisualization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)