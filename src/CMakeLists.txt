add_library(odevisualization STATIC
    dev_dat.cu
    flow_equations/flow_equation.cu
    flow_equations/jacobian_equation.cu
    # executer.cu
    collection/buffer.cpp
    collection/collection.cpp
    collection/collection_expander.cu
    collection/leaf.cpp
    grid_computation/dynamic_recursive_grid_computation.cu
    grid_computation/grid_computation.cu
    grid_computation/grid_computation_wrapper.cu
    grid_computation/static_recursive_grid_computation.cu
    util/monitor.cu
    util/kmeans.cpp
    util/kmeans_clustering.cu
    util/partial_ranges.cpp
    util/python_integration.cpp
    util/random.cu
    # modes/coordinate_operator.cu
    modes/fixed_point_criterion.cu
    modes/recursive_search.cu
    # modes/visualization.cu
    modes/mesh.cu
    boost/dev_dat_boost_integration.cu
    eigen/jacobians.cpp
    # observers/evolution.cu
    # observers/conditional_range_observer.cu
    # observers/conditional_intersection_observer.cu
    # observers/tracking_observer.cu
)

if(RUN_WITH_PYTHON_BACKEND)
  target_compile_definitions(odevisualization PUBLIC PYTHON_BACKEND)
  target_link_libraries(odevisualization PRIVATE ${target_link_libraries_python_appendix})
endif()

SET(GPU ON CACHE BOOL "Running on GPU")

if (GPU)
    target_compile_definitions(odevisualization PUBLIC -D GPU)
endif()

target_link_libraries(odevisualization PUBLIC paramhelper::paramhelper Eigen3::Eigen)

target_compile_options(odevisualization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualization PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)

message("PATH To EIGEN")
message(${CMAKE_SOURCE_DIR}/ext/Eigen/)

target_include_directories(odevisualization PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/Eigen/>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
