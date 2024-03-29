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

target_link_libraries(odevisualization PUBLIC stdc++fs paramhelper::paramhelper flowequations::flowequations devdat::devdat Eigen3::Eigen)

target_compile_options(odevisualization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

target_include_directories(odevisualization PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/Eigen/>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
