// Todo!!
// Function for testing if project_coordinates_on_expanded_cube_and_depth_per_cube_indices works
void GridComputation::test_projection()
{
    /* // Testing projection of coordinates on expanded cube and depth per cube indices
    dev_ptrvec_vec_int * expanded_element_indices_ptr = nullptr;
    dev_vec_int * expanded_depth_per_element_ptr = nullptr;
    std::tie(expanded_element_indices_ptr, expanded_depth_per_element_ptr) = project_coordinates_on_expanded_cube_and_depth_per_cube_indices(vertices, true, 0);
    if(monitor)
    {
        auto i = 0;
        for(auto elem : *expanded_element_indices_ptr)
        {
            print_range("Recheck Expanded cube indices after filling with individual cube indices in depth " + std::to_string(i), elem->begin(), elem->end());
            i++;
        }
        print_range("Recheck Expanded depth per collection", expanded_depth_per_element_ptr->begin(), expanded_depth_per_element_ptr->end());
    }

    // Reduce on cube reference indices
    const uint8_t dim_ = dim;
    auto i = 0;
    for(auto depth_index = 0; depth_index < expanded_element_indices_ptr->size(); depth_index++)
    {
        auto last_expanded_cube_index_iterator = thrust::remove_copy_if(
                (*expanded_element_indices_ptr)[depth_index]->begin(),
                (*expanded_element_indices_ptr)[depth_index]->end(),
                thrust::make_counting_iterator(0), // Works as mask for values that should be copied (checked if identity is fulfilled)
                (*expanded_element_indices_ptr)[depth_index]->begin(),
        [dim_] __host__ __device__ (const int &val) { return val % int(pow(2,dim_)); });

        (*expanded_element_indices_ptr)[depth_index]->resize(last_expanded_cube_index_iterator - (*expanded_element_indices_ptr)[depth_index]->begin());
        if(monitor)
            print_range("Reduced set in depth " + std::to_string(i), (*expanded_element_indices_ptr)[depth_index]->begin(), (*expanded_element_indices_ptr)[depth_index]->end());
        i++;
    }

    auto last_expanded_depth_index_iterator = thrust::remove_copy_if(
            expanded_depth_per_element_ptr->begin(),
            expanded_depth_per_element_ptr->end(),
            thrust::make_counting_iterator(0), // Works as mask for values that should be copied (checked if identity is fulfilled)
            expanded_depth_per_element_ptr->begin(),
    [dim_] __host__ __device__ (const int &val) { return val % int(pow(2,dim_)); });

    expanded_depth_per_element_ptr->resize(last_expanded_depth_index_iterator - expanded_depth_per_element_ptr->begin());

    if(monitor)
        print_range("Reduced expanded depth", expanded_depth_per_element_ptr->begin(), expanded_depth_per_element_ptr->end());

    compute_vertices(*expanded_element_indices_ptr, *expanded_depth_per_element_ptr);

    // Free memory
    delete expanded_depth_per_element_ptr;
    //delete (*expanded_element_indices_ptr)[0];
    thrust::for_each(expanded_element_indices_ptr->begin(), expanded_element_indices_ptr->end(), [] (dev_vec_int *elem) { delete elem; }); */
}
