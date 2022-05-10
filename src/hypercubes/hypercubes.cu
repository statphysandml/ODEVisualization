#include "../../include/hypercubes/hypercubes.hpp"


struct accumulate_n_branches
{
    accumulate_n_branches(const int dim_index, int init) : dim_index_(dim_index), sum_(init)
    {}

    __host__
    int operator()(const std::vector<int> elem)
    {
        sum_ = sum_ * elem[dim_index_];
        return sum_;
    }

    int sum_;
    const int dim_index_;
};


struct compute_axis_index_of_coor
{
    compute_axis_index_of_coor(const cudaT lambda_offset, const cudaT delta_lambda, const int n_branch_per_depth_at_dim) :
        lambda_offset_(lambda_offset), delta_lambda_(delta_lambda), n_branch_per_depth_at_dim_(n_branch_per_depth_at_dim)
    {}

    __host__ __device__
    int operator()(const cudaT &coordinate)
    {
        return (int((coordinate + lambda_offset_) / delta_lambda_) % n_branch_per_depth_at_dim_);
    }

    const cudaT lambda_offset_;
    const cudaT delta_lambda_;
    const int n_branch_per_depth_at_dim_;
};


struct sum_axis_indices_to_cube_index
{
    sum_axis_indices_to_cube_index(const int accum_n_branch_per_depth) :
        accum_n_branch_per_depth_(accum_n_branch_per_depth)
    {}

    __host__ __device__
    int operator()(const int &axis_index, const int &cube_index)
    {
        return cube_index + axis_index * accum_n_branch_per_depth_;
    }

    const int accum_n_branch_per_depth_;
};


struct compute_depth_vertex_coor_weight
{
    compute_depth_vertex_coor_weight(const int n_branch_per_depth, const int accum_n_branch_per_depth,
            const int depth_weight_divisor, dev_vec_int const &accum_n_branches_per_dim) :
            n_branch_per_depth_(n_branch_per_depth), accum_n_branch_per_depth_(accum_n_branch_per_depth),
            depth_weight_divisor_(depth_weight_divisor),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int cube_index = thrust::get<0>(t);
        int maximum_cube_depth = thrust::get<1>(t) + 1; // = k
        int current_val = thrust::get<2>(t);

        thrust::get<2>(t) += ((cube_index / accum_n_branch_per_depth_) % n_branch_per_depth_) * accum_n_branches_per_dim_ptr_[maximum_cube_depth] / depth_weight_divisor_;
    }

    const int n_branch_per_depth_;
    const int accum_n_branch_per_depth_;
    const int depth_weight_divisor_;
    const int * accum_n_branches_per_dim_ptr_;
};


struct compute_inner_vertex_coor
{
    compute_inner_vertex_coor(const int dim_index) : dim_index_(dim_index)
    {}

    __host__ __device__
    int operator()(const int &ith_inner_index) const
    {
        return int(ith_inner_index/ pow(2, dim_index_)) % 2;
    }

    const int dim_index_;
};


struct finalize_vertex_computation
{
    finalize_vertex_computation(const cudaT lambda_range, const cudaT lambda_offset,
            dev_vec_int const& accum_n_branches_per_dim) :
            lambda_range_(lambda_range), lambda_offset_(lambda_offset),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int reference_vertex = thrust::get<0>(t);
        int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k
        int inner_vertex_coor = thrust::get<2>(t);

        thrust::get<3>(t) = (reference_vertex + inner_vertex_coor) * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth] + lambda_offset_;
    }

    const cudaT lambda_range_;
    const cudaT lambda_offset_;
    const int * accum_n_branches_per_dim_ptr_;
};


struct finalize_reference_vertex_computation
{
    finalize_reference_vertex_computation(const cudaT lambda_range, const cudaT lambda_offset,
                                dev_vec_int const& accum_n_branches_per_dim) :
            lambda_range_(lambda_range), lambda_offset_(lambda_offset),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {

        return  reference_vertex * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1] + lambda_offset_;
    }

    const cudaT lambda_range_;
    const cudaT lambda_offset_;
    const int * accum_n_branches_per_dim_ptr_;
};


struct finalize_center_vertex_computation
{
    finalize_center_vertex_computation(const cudaT lambda_range, const cudaT lambda_offset,
                                dev_vec_int const& accum_n_branches_per_dim) :
            lambda_range_(lambda_range), lambda_offset_(lambda_offset),
            accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {
        return ((reference_vertex) * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1] +
                (reference_vertex + 1) * lambda_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1]) / 2 + lambda_offset_;
    }

    const cudaT lambda_range_;
    const cudaT lambda_offset_;
    const int * accum_n_branches_per_dim_ptr_;
};


HyperCubes::HyperCubes(const std::vector<std::vector<int>> n_branches_per_depth,
    const std::vector<std::pair<cudaT, cudaT>> lambda_ranges) :
    dim_(lambda_ranges.size()),
    n_branches_per_depth_(n_branches_per_depth),
    accum_n_branches_per_dim_(HyperCubes::compute_accum_n_branches_per_dim(n_branches_per_depth, lambda_ranges.size())),
    accum_n_branches_per_depth_(HyperCubes::compute_accum_n_branches_per_depth(n_branches_per_depth, lambda_ranges.size())),
    lambda_ranges_(lambda_ranges)
{}

//[ Static functions

thrust::host_vector<thrust::host_vector<int>> HyperCubes::compute_accum_n_branches_per_dim(const std::vector< std::vector<int>> &n_branches_per_depth, const uint dim)
{
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_dim(dim);
    for(auto dim_index = 0; dim_index < dim; dim_index++)
    {
        thrust::host_vector<int> accum_n_branches(n_branches_per_depth.size() + 1);
        accum_n_branches[0] = 1;
        thrust::transform(n_branches_per_depth.begin(), n_branches_per_depth.end(), accum_n_branches.begin() + 1, accumulate_n_branches(dim_index, 1));
        accum_n_branches_per_dim[dim_index] = accum_n_branches;
        // Testing
        // print_range("Accum branches per dim in dim " + std::to_string(dim_index), accum_n_branches.begin(), accum_n_branches.end());
    }
    return accum_n_branches_per_dim;
}

thrust::host_vector<thrust::host_vector<int>> HyperCubes::compute_accum_n_branches_per_depth(const std::vector< std::vector<int>> &n_branches_per_depth, const uint dim)
{
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_depth(n_branches_per_depth.size());
    for(auto depth_index = 0; depth_index < n_branches_per_depth.size(); depth_index++)
    {
        thrust::host_vector<int> accum_n_branches(dim + 1);
        accum_n_branches[0] = 1;
        thrust::inclusive_scan(n_branches_per_depth[depth_index].begin(), n_branches_per_depth[depth_index].end(), accum_n_branches.begin() + 1, thrust::multiplies<int>());
        accum_n_branches_per_depth[depth_index] = accum_n_branches;
        // Testing
        // print_range("Accum branches per depth in depth " + std::to_string(depth_index), accum_n_branches.begin(), accum_n_branches.end());
    }
    return accum_n_branches_per_depth;
}

//]

GridComputationWrapper HyperCubes::project_coordinates_on_expanded_cube_and_depth_per_cube_indices(const odesolver::DevDatC &coordinates, bool coordinates_on_grid, int depth) const
{
    if(depth == -1)
        depth = n_branches_per_depth_.size() - 1;
    else if(depth > n_branches_per_depth_.size() - 1)
    {
        std::cout << "\nERROR: Maximum number of branches per depth " << n_branches_per_depth_.size() << " do not coincide with depth " << depth <<  std::endl;
        std::exit(EXIT_FAILURE);
    }

    int total_number_of_cubes = coordinates.n_elems();
    GridComputationWrapper grcompwrap(total_number_of_cubes, depth + 1, depth);

    for(auto dim_index = 0; dim_index < dim_; dim_index++)
    {
        cudaT lambda_dim_range = (lambda_ranges_[dim_index].second - lambda_ranges_[dim_index].first);
        cudaT lambda_range_left = lambda_ranges_[dim_index].first;

        cudaT lambda_offset;
        if(coordinates_on_grid) {
            lambda_offset = 0.5*lambda_dim_range/accum_n_branches_per_dim_[dim_index][depth + 1]; // For avoidance of rounding errors -> corresponds to half of the width of the smallest considered cube
        }
        else  {
            lambda_offset = 0.0;
        }

        dev_vec temp_coordinates(coordinates[dim_index].begin(), coordinates[dim_index].end());

        // Shift coordinates to reference system (most left coordinate == 0)
        thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), temp_coordinates.begin(),
        [lambda_range_left] __host__ __device__(const cudaT &coor) { return coor - lambda_range_left; });

        for(auto depth_index = 0; depth_index < grcompwrap.expanded_cube_indices_.dim_size(); depth_index++)
        {

            cudaT delta_lambda = lambda_dim_range / accum_n_branches_per_dim_[dim_index][depth_index + 1]; // corresponds to the width of the considered cube

            // Compute axis indices
            dev_vec_int axis_index(total_number_of_cubes, 0); // corresponds to the index of the considered axis
            thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), axis_index.begin(),
                              compute_axis_index_of_coor(lambda_offset, delta_lambda, n_branches_per_depth_[depth_index][dim_index]));

            // Shift coordinates to corresponding new reference system in considered depth
            thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), axis_index.begin(),
                              temp_coordinates.begin(),
            [delta_lambda] __host__ __device__ (const cudaT &coor, const cudaT &cube_index)
            {
                return coor - (cube_index * delta_lambda);
            });

            // Add axis index to expanded cube indices
            thrust::transform(axis_index.begin(), axis_index.end(),
                              grcompwrap.expanded_cube_indices_[depth_index].begin(),
                              grcompwrap.expanded_cube_indices_[depth_index].begin(),
                              sum_axis_indices_to_cube_index(accum_n_branches_per_depth_[depth_index][dim_index]));
        }
    }

    return grcompwrap;
}

GridComputationWrapper HyperCubes::project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<Leaf*> &leaves, int depth) const
{
    if(depth == -1)
        depth = n_branches_per_depth_.size() - 1;
    else if(depth > n_branches_per_depth_.size() - 1)
    {
        std::cout << "\nERROR: Maximum number of branches per depth " << n_branches_per_depth_.size() << " do not coincide with depth " << depth <<  std::endl;
        std::exit(EXIT_FAILURE);
    }

    int total_number_of_cubes = leaves.size();

    GridComputationWrapper grcompwrap(total_number_of_cubes, depth +1 , depth);

    thrust::host_vector<int> host_expanded_cube_indices ((depth + 1) * total_number_of_cubes, 0);
    for(auto depth_index = 0; depth_index < grcompwrap.expanded_cube_indices_.dim_size(); depth_index++)
    {
        thrust::transform(thrust::host, leaves.begin(), leaves.end(), host_expanded_cube_indices.begin() + total_number_of_cubes * depth_index, [depth_index] (const Leaf * leaf) { return leaf->get_ith_cube_depth_index(depth_index); });
    }
    grcompwrap.expanded_cube_indices_.fill_by_vec(host_expanded_cube_indices);
    return grcompwrap;
}

void HyperCubes::compute_reference_vertices(odesolver::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap, int maximum_depth)
{
    for(auto dim_index = 0; dim_index < dim_; dim_index++) {
        compute_reference_vertex_in_dim(reference_vertices[dim_index], grcompwrap, dim_index, maximum_depth);

        // Compute delta range
        cudaT lambda_dim_range = (lambda_ranges_[dim_index].second - lambda_ranges_[dim_index].first);
        cudaT lambda_offset = lambda_ranges_[dim_index].first;

        // Finalize computation of the device reference vertex
        thrust::transform(reference_vertices[dim_index].begin(), reference_vertices[dim_index].end(), grcompwrap.expanded_depth_per_cube_.begin(), reference_vertices[dim_index].begin(),
                          finalize_reference_vertex_computation(lambda_dim_range, lambda_offset, accum_n_branches_per_dim_[dim_index]));
    }
}

odesolver::DevDatC HyperCubes::compute_reference_vertices(GridComputationWrapper &grcompwrap)
{
    // Initialize reference_vertices
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    auto reference_vertices = odesolver::DevDatC(dim_, total_number_of_cubes);
    
    // Compute reference_vertices
    compute_reference_vertices(reference_vertices, grcompwrap);

    return std::move(reference_vertices);
}

void HyperCubes::compute_vertices(odesolver::DevDatC &vertices, GridComputationWrapper &grcompwrap, int maximum_depth)
{
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();

    for(auto dim_index = 0; dim_index < dim_; dim_index++)
    {
        // Generate device vector of reference vertices for each vector
        odesolver::DevDatC reference_vertices_wrapper(1, total_number_of_cubes, 0.0);
        odesolver::DimensionIteratorC& reference_vertices = reference_vertices_wrapper[0];
        compute_reference_vertex_in_dim(reference_vertices, grcompwrap, dim_index, maximum_depth);

        // Testing -> Can be used as test without regarding the correct reference vertices
        // print_range("Reference vertices in dimension " + std::to_string(dim_index + 1), reference_vertices.begin(), reference_vertices.end());
        /* // Compute delta ranges
         * cudaT lambda_dim_range = (lambda_ranges[dim_index].second - lambda_ranges[dim_index].first);
         * cudaT lambda_offset = lambda_ranges[dim_index].first;
         * thrust::transform(reference_vertices.begin(), reference_vertices.end(), grcompwrap.expanded_depth_per_cube_.begin(), reference_vertices.begin(),
                          finalize_reference_vertex_computation(lambda_dim_range, lambda_offset, accum_n_branches_per_dim[dim_index])) */

        // Preparations for the expansion to vertices

        // Repeat reference vertex according to the number of vertices per cube
        repeated_range<dev_iterator> rep_ref_vertex_iterator(
            reference_vertices.begin(),
            reference_vertices.end(),
            pow(2, dim_));

        // Repeat maximum depth values according to the number of vertices per cube
        repeated_range<dev_iterator_int> rep_ref_depth_per_cube_iterator(
            grcompwrap.expanded_depth_per_cube_.begin(),
            grcompwrap.expanded_depth_per_cube_.end(),
            pow(2, dim_)
        );

        // Compute inner cube offset
        dev_vec_bool inner_vertex_coors(pow(2, dim_));
        thrust::tabulate(inner_vertex_coors.begin(), inner_vertex_coors.end(), compute_inner_vertex_coor(dim_index));
        tiled_range<dev_iterator_bool> rep_inner_vertex_coors(inner_vertex_coors.begin(), inner_vertex_coors.end(), total_number_of_cubes);

        // Testing
        // print_range("Inner coors", inner_vertex_coors.begin(), inner_vertex_coors.end());

        // Finalize

        // Compute delta range
        cudaT lambda_dim_range = (lambda_ranges_[dim_index].second - lambda_ranges_[dim_index].first);
        cudaT lambda_offset = lambda_ranges_[dim_index].first;

        // Finalize computation of device vertex
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(rep_ref_vertex_iterator.begin(), rep_ref_depth_per_cube_iterator.begin(), rep_inner_vertex_coors.begin(), vertices[dim_index].begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(rep_ref_vertex_iterator.end(), rep_ref_depth_per_cube_iterator.end(), rep_inner_vertex_coors.end(), vertices[dim_index].end())),
                         finalize_vertex_computation(lambda_dim_range, lambda_offset, accum_n_branches_per_dim_[dim_index]));

        // Testing
         /* if(monitor)
            print_range("Vertices in dimension " + std::to_string(dim_index + 1), vertices[dim_index].begin(), vertices[dim_index].end()); */
    }
}

odesolver::DevDatC HyperCubes::compute_vertices(GridComputationWrapper &grcompwrap)
{
    // Initialize vertices
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    auto vertices = odesolver::DevDatC(dim_, total_number_of_cubes * pow(2, dim_));
    
    // Compute vertices
    compute_vertices(vertices, grcompwrap);

    return std::move(vertices);
}

void HyperCubes::compute_cube_center_vertices(odesolver::DevDatC &center_vertices, GridComputationWrapper &grcompwrap, int maximum_depth)
{
    for (auto dim_index = 0; dim_index < dim_; dim_index++) {
        // Generate device vector of reference vertices for each vector
        compute_reference_vertex_in_dim(center_vertices[dim_index], grcompwrap, dim_index, maximum_depth);

        // Finalize

        // Compute delta range
        cudaT lambda_dim_range = (lambda_ranges_[dim_index].second - lambda_ranges_[dim_index].first);
        cudaT lambda_range_left = lambda_ranges_[dim_index].first;

        // Finalize computation of device center vertex
        thrust::transform(center_vertices[dim_index].begin(), center_vertices[dim_index].end(),
                          grcompwrap.expanded_depth_per_cube_.begin(), center_vertices[dim_index].begin(),
                          finalize_center_vertex_computation(lambda_dim_range, lambda_range_left,
                                                             accum_n_branches_per_dim_[dim_index]));

        // Testing
        if (monitor)
            print_range("Cube center vertices in dimension " + std::to_string(dim_index + 1),
                        center_vertices[dim_index].begin(), center_vertices[dim_index].end());
    }
}

odesolver::DevDatC HyperCubes::compute_cube_center_vertices(GridComputationWrapper &grcompwrap)
{
    // Initialize center_vertices
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    auto center_vertices = odesolver::DevDatC(dim_, total_number_of_cubes);
        
    // Compute center_vertices
    compute_cube_center_vertices(center_vertices, grcompwrap);

    return std::move(center_vertices);
}

// Getter functions

const std::vector<std::vector<int>>& HyperCubes::get_n_branches_per_depth() const
{
    return n_branches_per_depth_;
}

const std::vector<std::pair<cudaT, cudaT>>& HyperCubes::get_lambda_ranges() const
{
    return lambda_ranges_;
}

size_t HyperCubes::dim() const
{
    return dim_;
}

// Protected functions

void HyperCubes::compute_reference_vertex_in_dim(odesolver::DimensionIteratorC &reference_vertices, GridComputationWrapper &grcompwrap, int dim_index, int maximum_depth) const
{
    if(maximum_depth == 0)
        maximum_depth = grcompwrap.expanded_cube_indices_.dim_size();

    for(auto depth_index = 0; depth_index < maximum_depth; depth_index++)
    {
        int accum_n_branch_per_depth = accum_n_branches_per_depth_[depth_index][dim_index];
        int n_branch_per_depth = n_branches_per_depth_[depth_index][dim_index];
        int depth_weight_divisor = accum_n_branches_per_dim_[dim_index][depth_index + 1];
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(grcompwrap.expanded_cube_indices_[depth_index].begin(), grcompwrap.expanded_depth_per_cube_.begin(), reference_vertices.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(grcompwrap.expanded_cube_indices_[depth_index].end(), grcompwrap.expanded_depth_per_cube_.end(), reference_vertices.end())),
                         compute_depth_vertex_coor_weight(n_branch_per_depth, accum_n_branch_per_depth, depth_weight_divisor, accum_n_branches_per_dim_[dim_index]));
    }
}