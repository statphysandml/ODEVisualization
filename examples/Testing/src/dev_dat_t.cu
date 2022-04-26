#include "../include/dev_dat_t.hpp"

void testing_devdat() {
    int driver_version , runtime_version;
    cudaDriverGetVersion( &driver_version );
    cudaRuntimeGetVersion ( &runtime_version );
    std::cout << driver_version << "\t" << runtime_version << std::endl;

    // generate 32M random numbers serially
    thrust::host_vector<cudaT> h_vec(2 << 3, 0);
    // std::generate(h_vec.begin(), h_vec.end(), rand);
    h_vec[4] = 2.0;
    h_vec[10] = -1.0;

    print_range("Host vector", h_vec.begin(), h_vec.end());
    
    // transfer data to the device
    dev_vec d_vec = h_vec;
    print_range("Device vector", d_vec.begin(), d_vec.end());

    odesolver::DevDatC sampled_coordinates(d_vec, 2);

    sampled_coordinates.print_dim_by_dim();
    sampled_coordinates.print_elem_by_elem();

    // Testing the copy constructor
    std::cout << "Testing the copy constructor" << std::endl;
    odesolver::DevDatC a = sampled_coordinates;
    std::cout << "a.size(): " << a.size() << "; sampled_coordinates.size(): " << sampled_coordinates.size() << std::endl;
    a.print_dim_by_dim();
    a.print_elem_by_elem();

    // Testing the assignment operator (copy-and-swap-idiom)
    std::cout << "Testing the assignment operator (copy + assign)" << std::endl;
    odesolver::DevDatC b;
    b = sampled_coordinates;
    std::cout << "b.size(): " << b.size() << "; sampled_coordinates.size(): " << sampled_coordinates.size() << std::endl;
    b.print_dim_by_dim();
    b.print_elem_by_elem();

    // Testing the move operator
    std::cout << "Testing the move operator" << std::endl;
    odesolver::DevDatC c = std::move(sampled_coordinates);
    std::cout << "c.size(): " << c.size() << "; sampled_coordinates.size(): " << sampled_coordinates.size() << std::endl;
    c.print_dim_by_dim();
    c.print_elem_by_elem();

    // Transpose data
    auto transposed_vec_vec = c.transpose_device_data();
    odesolver::DevDatC tranposed_dev_dat(transposed_vec_vec);
    tranposed_dev_dat.print_dim_by_dim();
    tranposed_dev_dat.print_elem_by_elem();

    // Fill vector
    odesolver::DevDatC d(2, 8); // dim, N
    d.print_dim_by_dim();
    d.print_elem_by_elem();
    d.fill_by_vec(d_vec);
    d.print_dim_by_dim();
    d.print_elem_by_elem();

    // ...
    std::cout << "d.dim_size(): " << d.dim_size() << "; d.n_elems(): " << d.n_elems() << std::endl;
}