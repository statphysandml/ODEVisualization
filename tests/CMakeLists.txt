add_executable(odevisualisation_tests tests.cpp util/dev_dat_t.cpp)
target_link_libraries(odevisualisation_tests PUBLIC odevisualisation Catch2::Catch2)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(odevisualisation_tests)
