add_library(odevisualisation STATIC
    flow_equation_interface/flow_equation.cu
    flow_equation_interface/jacobian_equation.cu
    # executer.cu
    hypercubes/hypercubes.cu
    hypercubes/node.cu
    hypercubes/buffer.cu
    hypercubes/gridcomputationwrapper.cu
    hypercubes/dynamic_recursive_grid_computation.cu
    hypercubes/nodesexpander.cu
    hypercubes/static_recursive_grid_computation.cu
    util/helper_functions.cu
    util/monitor.cu
    util/dev_dat.cu
    util/dev_dat_boost_integration.cu
    util/python_integration.cpp
    util/partial_ranges.cpp
    util/jacobians.cpp
    util/json_conversion.cpp
    # extern/parameters.cu
    # extern/pathfinder.cu
    coordinate_operator.cu
    visualization.cu
    # observers/evolution.cu
    # observers/conditional_range_observer.cu
    # observers/conditional_intersection_observer.cu
    # observers/tracking_observer.cu
    fixed_point_search.cu
)

if(RUN_WITH_PYTHON_BACKEND)
  target_compile_definitions(odevisualisation PUBLIC PYTHON_BACKEND)
  target_link_libraries(odevisualisation PRIVATE ${target_link_libraries_python_appendix})
endif()

SET(GPU ON CACHE BOOL "Running on GPU")

if (GPU)
    target_compile_definitions(odevisualisation PUBLIC -D GPU)
endif()

target_link_libraries(odevisualisation PUBLIC paramhelper::paramhelper Eigen3::Eigen)

target_compile_options(odevisualisation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualisation PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)

message("PATH To EIGEN")
message(${CMAKE_SOURCE_DIR}/ext/Eigen/)

target_include_directories(odevisualisation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ext/Eigen/>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
