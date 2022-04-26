add_library(odevisualisationlib STATIC
    flow_equation_interface/flow_equation.cu
    flow_equation_interface/jacobian_equation.cu
    flow_equation_interface/flow_equation_system.cu
    # executer.cu
    hypercubes/hypercubes.cu
    hypercubes/node.cu
    hypercubes/buffer.cu
    hypercubes/gridcomputationwrapper.cu
    hypercubes/nodesexpander.cu
    util/helper_functions.cu
    util/monitor.cu
    util/dev_dat.cu
    util/python_integration.cpp
    util/lambda_range_generator.cu
    # extern/parameters.cu
    # extern/pathfinder.cu
    # coordinate_operator.cu
    # visualization.cu
    # observers/evolution.cu
    # observers/conditional_range_observer.cu
    # observers/conditional_intersection_observer.cu
    # observers/tracking_observer.cu
    fixed_point_search.cu
)

if(RUN_WITH_PYTHON_BACKEND)
  target_compile_definitions(odevisualisationlib PUBLIC PYTHON_BACKEND)
  target_link_libraries(odevisualisationlib PRIVATE ${target_link_libraries_python_appendix})
endif()

SET(GPU ON CACHE BOOL "Running on GPU")

if (GPU)
    target_compile_definitions(odevisualisationlib PUBLIC -D GPU)
endif()

target_link_libraries(odevisualisationlib PUBLIC paramhelper::paramhelper)

target_compile_options(odevisualisationlib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda; --expt-relaxed-constexpr>) 

set_target_properties(odevisualisationlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)

target_include_directories(odevisualisationlib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
