# Look for JSON headers explicitly if not already done in main CMakeLists.txt
if(NOT DEFINED JSON_INCLUDE_DIR)
    find_path(JSON_INCLUDE_DIR "nlohmann/json.hpp"
        PATHS
            ${CMAKE_PREFIX_PATH}
            ${SUPERBUILD_INSTALL_DIR}
            ${CMAKE_INSTALL_PREFIX}
        PATH_SUFFIXES
            include
    )
    
    if(JSON_INCLUDE_DIR)
        message(STATUS "Found JSON headers for Python bindings at: ${JSON_INCLUDE_DIR}")
    else()
        message(WARNING "nlohmann/json.hpp not found directly, Python binding compilation might fail")
    endif()
endif()

pybind11_add_module(odevisualization_python
    src/dev_dat_python.cu
    src/evolution_python.cu
    src/flow_equations_python.cu
    src/modes_python.cu
    src/odevisualization_python.cu
    src/recursive_search_python.cu
)

# Add CUDA-specific compilation options with explicit JSON include path
target_compile_options(odevisualization_python PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda 
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
        $<$<BOOL:${JSON_INCLUDE_DIR}>:-I${JSON_INCLUDE_DIR}>
    >
)

target_include_directories(odevisualization_python
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${JSON_INCLUDE_DIR}  # Add JSON include path explicitly
)

target_link_libraries(odevisualization_python PUBLIC odevisualization stdc++fs paramhelper::paramhelper flowequations::flowequations devdat::devdat)

set_target_properties(odevisualization_python PROPERTIES OUTPUT_NAME odevisualizationlib)