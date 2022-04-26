add_executable(odevisualisationlib_tests tests.cpp util/dev_dat_t.cpp)
target_link_libraries(odevisualisationlib_tests PUBLIC odevisualisationlib Catch2::Catch2)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(odevisualisationlib_tests)
