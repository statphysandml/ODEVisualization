// #include "testing/testing.hpp"
#include "catch2/catch.hpp"


#include <odesolver/dev_dat.hpp>
#include <odesolver/util/thrust_functors.hpp>

/* std::tuple< DevDatC > test_for_move_operators()
{
    DevDatC vec(1, 10, 0);
    *vec[0].begin() = 2;
    return std::make_tuple(vec);
} */

/* void devdat_basics_test()
{
    int dim = 3;
    std::vector< std::vector<cudaT> > data{std::vector<cudaT> {0, 1, 2}, std::vector<cudaT> {2, 3, 4}};
    // DevDatC dat = transpose_to_device_data(data);
    DevDatC dat;
    std::tie(dat) = test_for_move_operators();
    print_range("test", dat[0].begin(), dat[0].end());
    dat = DevDatC(dim, 10);
    DevDatC dat_long(dim, 10 * pow(2, dim));

    for(auto dim_index = 0; dim_index < dim; dim_index++) {
        print_range("Data", dat[dim_index].begin(), dat[dim_index].end());
        repeated_range<dev_iterator> rep_ref_vertex_iterator(dat[dim_index].begin(), dat[dim_index].end(), pow(2, dim));
        print_range("Data after rep", rep_ref_vertex_iterator.begin(), rep_ref_vertex_iterator.end());
        // Finalize computation of device vertex
        thrust::transform(rep_ref_vertex_iterator.begin(), rep_ref_vertex_iterator.end(), dat_long[dim_index].begin(),
                          []
                                  __host__ __device__(const cudaT &value) { return value * 2; });
        print_range("Data after transform", dat_long[dim_index].begin(), dat_long[dim_index].end());
    }
} */

TEST_CASE( "cuda_version", "[cuda]") {
    int driver_version , runtime_version;
    cudaDriverGetVersion( &driver_version );
    cudaRuntimeGetVersion ( &runtime_version );
    std::cout << driver_version << "\t" << runtime_version << std::endl;
}

TEST_CASE( "dev_dat", "[devdat]"){
    int dim = 3;
    std::vector< std::vector<cudaT> > data{std::vector<cudaT> {0, 1, 2}, std::vector<cudaT> {2, 3, 4}};
}

/* TEST_CASE( "add_and_get_entry", "[parameters]" ){
  Parameters params = Parameters::create_by_params(json {{"a", 0}, {"vec", std::vector<double> {0.0, 1.0}}});

  REQUIRE(params.get_entry<double>("a") == 0);

  params.add_entry("c", "test_c");

  REQUIRE(params.get_entry<std::string>("c") == "test_c");
} */