add_executable(odevisualization_tests tests.cpp util/dev_dat_t.cpp)
target_link_libraries(odevisualization_tests PUBLIC odevisualization paramhelper::paramhelper flowequations::flowequations devdat::devdat Eigen3::Eigen Catch2::Catch2)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(odevisualization_tests)
