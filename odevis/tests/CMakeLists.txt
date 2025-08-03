include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.8
)
FetchContent_MakeAvailable(Catch2)

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
        message(STATUS "Found JSON headers for tests at: ${JSON_INCLUDE_DIR}")
    else()
        message(WARNING "nlohmann/json.hpp not found directly, test compilation might fail")
    endif()
endif()

add_executable(odevisualization_tests tests.cpp util/dev_dat_t.cpp)

# Add CUDA-specific compilation options with explicit JSON include path
if(JSON_INCLUDE_DIR)
    target_compile_options(odevisualization_tests PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-I${JSON_INCLUDE_DIR}>
    )
endif()

target_include_directories(odevisualization_tests PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${JSON_INCLUDE_DIR}  # Add JSON include path explicitly
)

target_link_libraries(odevisualization_tests PRIVATE 
    odevisualization 
    paramhelper::paramhelper 
    flowequations::flowequations 
    devdat::devdat 
    Catch2::Catch2
)

# allow user to run tests with `make test` or `ctest`
include(${catch2_SOURCE_DIR}/contrib/Catch.cmake)
catch_discover_tests(odevisualization_tests)