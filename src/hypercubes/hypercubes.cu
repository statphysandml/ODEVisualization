#include "../../include/hypercubes/hypercubes.hpp"



struct greater_than_zero
{
    template< typename T>
    __host__ __device__
    T operator()(const T &val) const
    {
        return val > 0;
    }
};


struct accumulate_n_branches
{
    accumulate_n_branches(const int dim_index_, int init_) : dim_index(dim_index_), sum(init_)
    {}

    __host__
    int operator()(const std::vector<int> elem)
    {
        sum = sum * elem[dim_index];
        return sum;
    }

    int sum;
    const int dim_index;
};


struct compute_axis_index_of_coor
{
    compute_axis_index_of_coor(const cudaT lambda_offset_, const cudaT delta_lambda_, const int n_branch_per_depth_at_dim_) :
        lambda_offset(lambda_offset_), delta_lambda(delta_lambda_), n_branch_per_depth_at_dim(n_branch_per_depth_at_dim_)
    {}

    __host__ __device__
    int operator()(const cudaT &coordinate)
    {
        return (int((coordinate+lambda_offset)/delta_lambda) % n_branch_per_depth_at_dim);
    }

    const cudaT lambda_offset;
    const cudaT delta_lambda;
    const int n_branch_per_depth_at_dim;
};


struct sum_axis_indices_to_cube_index
{
    sum_axis_indices_to_cube_index(const int accum_n_branch_per_depth_) :
        accum_n_branch_per_depth(accum_n_branch_per_depth_)
    {}

    __host__ __device__
    int operator()(const int &axis_index, const int &cube_index)
    {
        return cube_index + axis_index * accum_n_branch_per_depth;
    }

    const int accum_n_branch_per_depth;
};


/* Remark: Generating an array for accum_n_branches_per_dim in advance as it is done for the other expanded vectors
 * makes no sense since the values do not differ from node to node (only the depth values) (reconsider this argument)
 * -> this is probably not true, only valid argument would be redundancy */

struct compute_depth_vertex_coor_weight
{
    compute_depth_vertex_coor_weight(const int n_branch_per_depth_, const int accum_n_branch_per_depth_,
            const int depth_weight_divisor_, dev_vec_int const& accum_n_branches_per_dim_) :
            n_branch_per_depth(n_branch_per_depth_), accum_n_branch_per_depth(accum_n_branch_per_depth_),
            depth_weight_divisor(depth_weight_divisor_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int cube_index = thrust::get<0>(t);
        int maximum_cube_depth = thrust::get<1>(t) + 1; // = k
        int current_val = thrust::get<2>(t);

        thrust::get<2>(t) += ((cube_index/ accum_n_branch_per_depth) % n_branch_per_depth) * accum_n_branches_per_dim_ptr[maximum_cube_depth] / depth_weight_divisor;
    }

    const int n_branch_per_depth;
    const int accum_n_branch_per_depth;
    const int depth_weight_divisor;
    const int * accum_n_branches_per_dim_ptr;
};


struct compute_inner_vertex_coor
{
    compute_inner_vertex_coor(const int dim_index_) : dim_index(dim_index_)
    {}

    __host__ __device__
    int operator()(const int &ith_inner_index) const
    {
        return int(ith_inner_index/ pow(2, dim_index)) % 2;
    }

    const int dim_index;
};


struct finalize_vertex_computation
{
    finalize_vertex_computation(const cudaT lambda_range_, const cudaT lambda_offset_,
            dev_vec_int const& accum_n_branches_per_dim_) :
            lambda_range(lambda_range_), lambda_offset(lambda_offset_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        int reference_vertex = thrust::get<0>(t);
        int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k
        int inner_vertex_coor = thrust::get<2>(t);

        thrust::get<3>(t) = (reference_vertex + inner_vertex_coor) * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth] + lambda_offset;
    }

    const cudaT lambda_range;
    const cudaT lambda_offset;
    const int * accum_n_branches_per_dim_ptr;
};


struct finalize_reference_vertex_computation
{
    finalize_reference_vertex_computation(const cudaT lambda_range_, const cudaT lambda_offset_,
                                dev_vec_int const& accum_n_branches_per_dim_) :
            lambda_range(lambda_range_), lambda_offset(lambda_offset_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {

        return  reference_vertex * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth + 1] + lambda_offset;
    }

    const cudaT lambda_range;
    const cudaT lambda_offset;
    const int * accum_n_branches_per_dim_ptr;
};


struct finalize_center_vertex_computation
{
    finalize_center_vertex_computation(const cudaT lambda_range_, const cudaT lambda_offset_,
                                dev_vec_int const& accum_n_branches_per_dim_) :
            lambda_range(lambda_range_), lambda_offset(lambda_offset_),
            accum_n_branches_per_dim_ptr(thrust::raw_pointer_cast(&accum_n_branches_per_dim_[0]))
    {}

    __host__ __device__
    cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
    {
        // int reference_vertex = thrust::get<0>(t);
        // int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k

        return ((reference_vertex) * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth + 1] +
                (reference_vertex + 1) * lambda_range / accum_n_branches_per_dim_ptr[maximum_cube_depth + 1]) / 2 + lambda_offset;
    }

    const cudaT lambda_range;
    const cudaT lambda_offset;
    const int * accum_n_branches_per_dim_ptr;
};


// Checks if the given number of positive signs is equal to 0 or to upper bound.
// If this is not the case, the given cube contains definitly no fixed point.
// With status, the previous status is taken into account (if it has been recognized already as no fixed point)
struct check_for_no_fixed_point
{
    check_for_no_fixed_point(const int upper_bound_): upper_bound(upper_bound_)
    {}

    __host__ __device__
    bool operator()(const int &val, const bool& status) const
    {
        return ((val == upper_bound) or (val == 0)) or status;
    }

    const int upper_bound;
};


HyperCubes::HyperCubes(const std::vector< std::vector<int>> n_branches_per_depth_,
    const std::vector <std::pair<cudaT, cudaT>> lambda_ranges_) :
    dim(lambda_ranges_.size()),
    n_branches_per_depth(n_branches_per_depth_),
    accum_n_branches_per_dim(HyperCubes::compute_accum_n_branches_per_dim(n_branches_per_depth_, lambda_ranges_.size())),
    accum_n_branches_per_depth(HyperCubes::compute_accum_n_branches_per_depth(n_branches_per_depth_, lambda_ranges_.size())),
    lambda_ranges(lambda_ranges_)
{}

//[ Static functions

thrust::host_vector<thrust::host_vector<int>> HyperCubes::compute_accum_n_branches_per_dim(const std::vector< std::vector<int>> &n_branches_per_depth_, const uint dim_)
{
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_dim_(dim_);
    for(auto dim_index = 0; dim_index < dim_; dim_index++)
    {
        thrust::host_vector<int> accum_n_branches(n_branches_per_depth_.size() + 1);
        accum_n_branches[0] = 1;
        thrust::transform(n_branches_per_depth_.begin(), n_branches_per_depth_.end(), accum_n_branches.begin() + 1, accumulate_n_branches(dim_index, 1));
        accum_n_branches_per_dim_[dim_index] = accum_n_branches;
        // Testing
        // print_range("Accum branches per dim in dim " + std::to_string(dim_index), accum_n_branches.begin(), accum_n_branches.end());
    }
    return accum_n_branches_per_dim_;
}

thrust::host_vector<thrust::host_vector<int>> HyperCubes::compute_accum_n_branches_per_depth(const std::vector< std::vector<int>> &n_branches_per_depth_, const uint dim_)
{
    thrust::host_vector<thrust::host_vector<int>> accum_n_branches_per_depth_(n_branches_per_depth_.size());
    for(auto depth_index = 0; depth_index < n_branches_per_depth_.size(); depth_index++)
    {
        thrust::host_vector<int> accum_n_branches(dim_ + 1);
        accum_n_branches[0] = 1;
        thrust::inclusive_scan(n_branches_per_depth_[depth_index].begin(), n_branches_per_depth_[depth_index].end(), accum_n_branches.begin() + 1, thrust::multiplies<int>());
        accum_n_branches_per_depth_[depth_index] = accum_n_branches;
        // Testing
        // print_range("Accum branches per depth in depth " + std::to_string(depth_index), accum_n_branches.begin(), accum_n_branches.end());
    }
    return accum_n_branches_per_depth_;
}

void HyperCubes::compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs)
{
    // Initialize a vectors for sign checks
    auto total_number_of_cubes = summed_positive_signs.size();
    auto total_number_of_vertices = velocity_sign_properties_in_dim.size();
    if(total_number_of_cubes != 0)
    {
        auto number_of_vertices_per_cube = int(total_number_of_vertices / total_number_of_cubes); // = pow(2, dim)

        dev_vec_int indices_of_summed_positive_signs(total_number_of_vertices);

        // Necessary that reduce by key works (cannot handle mixture of bool and integer), ToDo: Alternative solution??
        dev_vec_int int_velocity_sign_properties_in_dim(velocity_sign_properties_in_dim.begin(), velocity_sign_properties_in_dim.end());

        /*Use iterators to transform the linear index into a row index -> the final iterator repeats the
        * row indices (0 to pow(2, dim)-1) total_number_of_cubes times, i.e.: 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1
        * These are then used as a mask to define which signs in vertex_velocity should be summed up.
        * indices_of_summed_positive_signs contains the keys for the mask, i.e. (0, 1, 2, etc.) and
        * summed_positive_signs the corresponding sum per key. */
        // Sum positive signs
        thrust::reduce_by_key
                (thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                 linear_index_to_row_index<int>(number_of_vertices_per_cube)),
                 thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                 linear_index_to_row_index<int>(number_of_vertices_per_cube)) +
                 (number_of_vertices_per_cube * total_number_of_cubes),
                 int_velocity_sign_properties_in_dim.begin(),
                 indices_of_summed_positive_signs.begin(),
                 summed_positive_signs.begin(),
                 thrust::equal_to<int>(),
                 thrust::plus<int>());
    }
}

//]

GridComputationWrapper HyperCubes::project_coordinates_on_expanded_cube_and_depth_per_cube_indices(odesolver::DevDatC coordinates, bool coordinates_on_grid, int depth) const // no reference since coordinates is changed within this function
{
    if(depth == -1)
        depth = n_branches_per_depth.size() - 1;
    else if(depth > n_branches_per_depth.size() - 1)
    {
        std::cout << "\nERROR: Maximum number of branches per depth " << n_branches_per_depth.size() << " do not coincide with depth " << depth <<  std::endl;
        std::exit(EXIT_FAILURE);
    }

    int tnoc = coordinates.n_elems(); // total number of cubes
    GridComputationWrapper grcompwrap(tnoc, depth + 1, depth);

    for(auto dim_index = 0; dim_index < dim; dim_index++)
    {
        cudaT lambda_dim_range = (lambda_ranges[dim_index].second - lambda_ranges[dim_index].first);
        cudaT lambda_range_left = lambda_ranges[dim_index].first;

        cudaT lambda_offset;
        if(coordinates_on_grid) {
            lambda_offset = 0.5*lambda_dim_range/accum_n_branches_per_dim[dim_index][depth + 1]; // For avoidance of rounding errors -> corresponds to half of the width of the smallest considered cube
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

            cudaT delta_lambda = lambda_dim_range / accum_n_branches_per_dim[dim_index][depth_index + 1]; // corresponds to the width of the considered cube

            // Compute axis indices
            dev_vec_int axis_index(tnoc, 0); // corresponds to the index of the considered axis
            thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), axis_index.begin(),
                              compute_axis_index_of_coor(lambda_offset, delta_lambda, n_branches_per_depth[depth_index][dim_index]));

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
                              sum_axis_indices_to_cube_index(accum_n_branches_per_depth[depth_index][dim_index]));
        }
    }

    return grcompwrap;
}

GridComputationWrapper HyperCubes::project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<Leaf*> &leaves, int depth) const
{
    if(depth == -1)
        depth = n_branches_per_depth.size() - 1;
    else if(depth > n_branches_per_depth.size() - 1)
    {
        std::cout << "\nERROR: Maximum number of branches per depth " << n_branches_per_depth.size() << " do not coincide with depth " << depth <<  std::endl;
        std::exit(EXIT_FAILURE);
    }

    int tnoc = leaves.size();

    GridComputationWrapper grcompwrap(tnoc, depth +1 , depth);

    thrust::host_vector<int> host_expanded_cube_indices ((depth + 1) * tnoc, 0);
    for(auto depth_index = 0; depth_index < grcompwrap.expanded_cube_indices_.dim_size(); depth_index++)
    {
        thrust::transform(thrust::host, leaves.begin(), leaves.end(), host_expanded_cube_indices.begin() + tnoc * depth_index, [depth_index] (const Leaf * leaf) { return leaf->get_ith_cube_depth_index(depth_index); });
    }
    grcompwrap.expanded_cube_indices_.fill_by_vec(host_expanded_cube_indices);
    return grcompwrap;
}

void HyperCubes::compute_reference_vertices(odesolver::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap)
{
    for(auto dim_index = 0; dim_index < dim; dim_index++) {
        compute_reference_vertex_in_dim(reference_vertices[dim_index], grcompwrap, dim_index);

        // Compute delta range
        cudaT lambda_dim_range = (lambda_ranges[dim_index].second - lambda_ranges[dim_index].first);
        cudaT lambda_offset = lambda_ranges[dim_index].first;

        // Finalize computation of the device reference vertex
        thrust::transform(reference_vertices[dim_index].begin(), reference_vertices[dim_index].end(), grcompwrap.expanded_depth_per_cube_.begin(), reference_vertices[dim_index].begin(),
                          finalize_reference_vertex_computation(lambda_dim_range, lambda_offset, accum_n_branches_per_dim[dim_index]));
    }
    vertex_mode = ReferenceVertices;;
}

odesolver::DevDatC HyperCubes::compute_reference_vertices(GridComputationWrapper &grcompwrap)
{
    // Initialize reference_vertices
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    auto reference_vertices = odesolver::DevDatC (dim, total_number_of_cubes);
    
    // Compute reference_vertices
    compute_reference_vertices(reference_vertices, grcompwrap);

    return std::move(reference_vertices);
}

void HyperCubes::compute_vertices(odesolver::DevDatC &vertices, GridComputationWrapper &grcompwrap, int total_number_of_cubes)
{
    if(total_number_of_cubes == 0)
        total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();

    for(auto dim_index = 0; dim_index < dim; dim_index++)
    {
        // Generate device vector of reference vertices for each vector
        odesolver::DevDatC reference_vertices_wrapper(1, total_number_of_cubes, 0.0);
        odesolver::DimensionIteratorC& reference_vertices_ = reference_vertices_wrapper[0];
        compute_reference_vertex_in_dim(reference_vertices_, grcompwrap, dim_index);

        // Testing -> Can be used as test without regarding the correct reference vertices
        // print_range("Reference vertices in dimension " + std::to_string(dim_index + 1), reference_vertices_.begin(), reference_vertices_.end());
        /* // Compute delta ranges
         * cudaT lambda_dim_range = (lambda_ranges[dim_index].second - lambda_ranges[dim_index].first);
         * cudaT lambda_offset = lambda_ranges[dim_index].first;
         * thrust::transform(reference_vertices_.begin(), reference_vertices_.end(), grcompwrap.expanded_depth_per_cube_.begin(), reference_vertices_.begin(),
                          finalize_reference_vertex_computation(lambda_dim_range, lambda_offset, accum_n_branches_per_dim[dim_index])) */

        // Preparations for the expansion to vertices

        // Repeat reference vertex according to the number of vertices per cube
        repeated_range<dev_iterator> rep_ref_vertex_iterator(
            reference_vertices_.begin(),
            reference_vertices_.end(),
            pow(2, dim));

        // Repeat maximum depth values according to the number of vertices per cube
        repeated_range<dev_iterator_int> rep_ref_depth_per_cube_iterator(
            grcompwrap.expanded_depth_per_cube_.begin(),
            grcompwrap.expanded_depth_per_cube_.begin() + total_number_of_cubes,
            pow(2, dim)
        );

        // Compute inner cube offset
        dev_vec_bool inner_vertex_coors(pow(2, dim)); // Computation can be shifted to global function (-> apropriate??)
        thrust::tabulate(inner_vertex_coors.begin(), inner_vertex_coors.end(), compute_inner_vertex_coor(dim_index));
        tiled_range<dev_iterator_bool> rep_inner_vertex_coors(inner_vertex_coors.begin(), inner_vertex_coors.end(), total_number_of_cubes);

        // Testing
        // print_range("Inner coors", inner_vertex_coors.begin(), inner_vertex_coors.end());

        // Finalize

        // Compute delta range
        cudaT lambda_dim_range = (lambda_ranges[dim_index].second - lambda_ranges[dim_index].first);
        cudaT lambda_offset = lambda_ranges[dim_index].first;

        // Finalize computation of device vertex
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(rep_ref_vertex_iterator.begin(), rep_ref_depth_per_cube_iterator.begin(), rep_inner_vertex_coors.begin(), vertices[dim_index].begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(rep_ref_vertex_iterator.end(), rep_ref_depth_per_cube_iterator.end(), rep_inner_vertex_coors.end(), vertices[dim_index].begin() + total_number_of_cubes * pow(2, dim))),
                         finalize_vertex_computation(lambda_dim_range, lambda_offset, accum_n_branches_per_dim[dim_index]));

        // Testing
         /* if(monitor)
            print_range("Vertices in dimension " + std::to_string(dim_index + 1), vertices[dim_index].begin(), vertices[dim_index].end()); */
    }

    vertex_mode = CubeVertices;
}

odesolver::DevDatC HyperCubes::compute_vertices(GridComputationWrapper &grcompwrap)
{
    // Initialize vertices
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    auto vertices = odesolver::DevDatC(dim, total_number_of_cubes * pow(2, dim));
    
    // Compute vertices
    compute_vertices(vertices, grcompwrap);

    return std::move(vertices);
}

void HyperCubes::compute_cube_center_vertices(odesolver::DevDatC &center_vertices, GridComputationWrapper &grcompwrap)
{
    for (auto dim_index = 0; dim_index < dim; dim_index++) {
        // Generate device vector of reference vertices for each vector
        compute_reference_vertex_in_dim(center_vertices[dim_index], grcompwrap, dim_index);

        // Finalize

        // Compute delta range
        cudaT lambda_dim_range = (lambda_ranges[dim_index].second - lambda_ranges[dim_index].first);
        cudaT lambda_range_left = lambda_ranges[dim_index].first;

        // Finalize computation of device center vertex
        thrust::transform(center_vertices[dim_index].begin(), center_vertices[dim_index].end(),
                          grcompwrap.expanded_depth_per_cube_.begin(), center_vertices[dim_index].begin(),
                          finalize_center_vertex_computation(lambda_dim_range, lambda_range_left,
                                                             accum_n_branches_per_dim[dim_index]));

        // Testing
        if (monitor)
            print_range("Cube center vertices in dimension " + std::to_string(dim_index + 1),
                        center_vertices[dim_index].begin(), center_vertices[dim_index].end());
    }

    vertex_mode = CenterVertices;
}

odesolver::DevDatC HyperCubes::compute_cube_center_vertices(GridComputationWrapper &grcompwrap)
{
    // Initialize center_vertices
    auto total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    auto center_vertices = odesolver::DevDatC(dim, total_number_of_cubes);
        
    // Compute center_vertices
    compute_cube_center_vertices(center_vertices, grcompwrap);

    return std::move(center_vertices);
}

/* void HyperCubes::determine_vertex_velocities(FlowEquationsWrapper * flow_equations)
{
    vertex_velocities = compute_vertex_velocities(vertices, flow_equations); */
    // Testing
    /* if(monitor)
        for(auto dim_index = 0; dim_index < dim; dim_index++)
            print_range("Vertex velocities in dimension " + std::to_string(dim_index + 1), vertex_velocities[dim_index].begin(), vertex_velocities[dim_index].end()); */
// }

thrust::host_vector<int> HyperCubes::determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities, int total_number_of_cubes)
{
    if (vertex_mode != CubeVertices and vertex_mode != CenterVertices)
    {
        std::cout << "\nERROR: Wrong vertex mode for computation of potential fix points" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if(total_number_of_cubes == 0)
        total_number_of_cubes = int(vertex_velocities.n_elems() / pow(2, dim));

    auto number_of_vertices_ = vertex_velocities.n_elems(); // to avoid a pass of this within the lambda capture
    thrust::host_vector<dev_vec_bool> velocity_sign_properties(dim);
    thrust::generate(velocity_sign_properties.begin(), velocity_sign_properties.end(), [number_of_vertices_]() { return dev_vec_bool (number_of_vertices_, false); });

    // Initial potential fixed points -> at the beginning all cubes contain potential fixed points ( false = potential fixed point )
    dev_vec_bool pot_fixed_points(total_number_of_cubes, false);
    for(auto dim_index = 0; dim_index < dim; dim_index ++)
    {
        // Turn vertex_velocities into an array with 1.0 and 0.0 for change in sign
        thrust::transform(vertex_velocities[dim_index].begin(), vertex_velocities[dim_index].end(), velocity_sign_properties[dim_index].begin(), greater_than_zero());

        // Initialize a vector for sign checks
        dev_vec_int summed_positive_signs(total_number_of_cubes, 0); // Contains the sum of positive signs within each cube
        HyperCubes::compute_summed_positive_signs_per_cube(velocity_sign_properties[dim_index], summed_positive_signs);

        // Testing
        if(monitor)
            print_range("Summed positive signs in dim " + std::to_string(dim_index), summed_positive_signs.begin(), summed_positive_signs.end());

        // Check if the sign has changed in this component (dimension), takes the previous status into account
        thrust::transform(summed_positive_signs.begin(), summed_positive_signs.end(), pot_fixed_points.begin(), pot_fixed_points.begin(), check_for_no_fixed_point(pow(2, dim)));
    }

    // Genereate mock fixed points
    //srand(13);
    //thrust::generate(thrust::host, pot_fixed_points.begin(), pot_fixed_points.end(), []() { return 0; } ); // rand() % 8

    // Test output
    /* std::cout << "Potential fixed points in linearized vertex velocities: " << std::endl;
    int i = 0;
    for(const auto &elem : pot_fixed_points) {
        std::cout << i << ": " << elem << " - ";
        i++;
    }
    std::cout << std::endl; */

    // Reduce on indices with potential fixed points (filter the value with pot_fixed_points==True) // (offset iterator + 1)  -> not used anymore (why initially used??)
    dev_vec_int indices_of_pot_fixed_points(total_number_of_cubes);
    auto last_potential_fixed_point_iterator = thrust::remove_copy_if(
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(total_number_of_cubes),
            pot_fixed_points.begin(), // Works as mask for values that should be copied (checked if identity is fulfilled)
            indices_of_pot_fixed_points.begin(),
            thrust::identity<int>());

    // Store valid indices of potential fixed points in host_indices_of_pot_fixed_points
    thrust::host_vector<int> host_indices_of_pot_fixed_points(indices_of_pot_fixed_points.begin(), last_potential_fixed_point_iterator);
    // indices_of_pot_fixed_points.resize(last_potential_fixed_point_iterator - indices_of_pot_fixed_points.begin());  -> alternative way to do this
    // host_indices_of_pot_fixed_points = indices_of_pot_fixed_points;

    // Test output
    /* std::cout << "Indices of potential fixed points: " << std::endl;
    for(auto &elem : host_indices_of_pot_fixed_points)
        std::cout << elem << " ";
    std::cout << std::endl; */

    return host_indices_of_pot_fixed_points;
}

// Getter functions

/* const odesolver::DevDatC& HyperCubes::get_vertex_velocities() const
{
    return vertex_velocities;
} */

const std::vector<std::vector<int>>& HyperCubes::get_n_branches_per_depth() const
{
    return n_branches_per_depth;
}

const std::vector<std::pair<cudaT, cudaT>>& HyperCubes::get_lambda_ranges() const
{
    return lambda_ranges;
}

// Test function

// Todo!!
// Function for testing if project_coordinates_on_expanded_cube_and_depth_per_cube_indices works
void HyperCubes::test_projection()
{
    /* // Testing projection of coordinates on expanded cube and depth per cube indices
    dev_ptrvec_vec_int * expanded_cube_indices_ptr = nullptr;
    dev_vec_int * expanded_depth_per_cube_ptr = nullptr;
    std::tie(expanded_cube_indices_ptr, expanded_depth_per_cube_ptr) = project_coordinates_on_expanded_cube_and_depth_per_cube_indices(vertices, true, 0);
    if(monitor)
    {
        auto i = 0;
        for(auto elem : *expanded_cube_indices_ptr)
        {
            print_range("Recheck Expanded cube indices after filling with individual cube indices in depth " + std::to_string(i), elem->begin(), elem->end());
            i++;
        }
        print_range("Recheck Expanded depth per node", expanded_depth_per_cube_ptr->begin(), expanded_depth_per_cube_ptr->end());
    }

    // Reduce on cube reference indices
    const uint8_t dim_ = dim;
    auto i = 0;
    for(auto depth_index = 0; depth_index < expanded_cube_indices_ptr->size(); depth_index++)
    {
        auto last_expanded_cube_index_iterator = thrust::remove_copy_if(
                (*expanded_cube_indices_ptr)[depth_index]->begin(),
                (*expanded_cube_indices_ptr)[depth_index]->end(),
                thrust::make_counting_iterator(0), // Works as mask for values that should be copied (checked if identity is fulfilled)
                (*expanded_cube_indices_ptr)[depth_index]->begin(),
        [dim_] __host__ __device__ (const int &val) { return val % int(pow(2,dim_)); });

        (*expanded_cube_indices_ptr)[depth_index]->resize(last_expanded_cube_index_iterator - (*expanded_cube_indices_ptr)[depth_index]->begin());
        if(monitor)
            print_range("Reduced set in depth " + std::to_string(i), (*expanded_cube_indices_ptr)[depth_index]->begin(), (*expanded_cube_indices_ptr)[depth_index]->end());
        i++;
    }

    auto last_expanded_depth_index_iterator = thrust::remove_copy_if(
            expanded_depth_per_cube_ptr->begin(),
            expanded_depth_per_cube_ptr->end(),
            thrust::make_counting_iterator(0), // Works as mask for values that should be copied (checked if identity is fulfilled)
            expanded_depth_per_cube_ptr->begin(),
    [dim_] __host__ __device__ (const int &val) { return val % int(pow(2,dim_)); });

    expanded_depth_per_cube_ptr->resize(last_expanded_depth_index_iterator - expanded_depth_per_cube_ptr->begin());

    if(monitor)
        print_range("Reduced expanded depth", expanded_depth_per_cube_ptr->begin(), expanded_depth_per_cube_ptr->end());

    compute_vertices(*expanded_cube_indices_ptr, *expanded_depth_per_cube_ptr);

    // Free memory
    delete expanded_depth_per_cube_ptr;
    //delete (*expanded_cube_indices_ptr)[0];
    thrust::for_each(expanded_cube_indices_ptr->begin(), expanded_cube_indices_ptr->end(), [] (dev_vec_int *elem) { delete elem; }); */
}

// Include this function into hypercubes?? # Currently not used!!
/* thrust::host_vector< dev_vec_bool* > * get_velocity_sign_properties(std::vector< dev_vec* > &vertex_velocities)
{
    const uint dim = vertex_velocities.size();
    auto number_of_coordinates_ = vertex_velocities[0]->size(); // to avoid a pass of this within the lambda capture
    auto * velocity_sign_properties_ptr = new thrust::host_vector< dev_vec_bool* > (dim);
    thrust::generate(velocity_sign_properties_ptr->begin(), velocity_sign_properties_ptr->end(), [number_of_coordinates_]() { return new dev_vec_bool (number_of_coordinates_, false); });
    for(auto dim_index = 0; dim_index < dim; dim_index ++)
    {
        // Turn vertex_velocities into an array with 1.0 and 0.0 for change in sign
        thrust::transform(vertex_velocities[dim_index]->begin(), vertex_velocities[dim_index]->end(), (*velocity_sign_properties_ptr)[dim_index]->begin(), greater_than_zero());
    }
    return velocity_sign_properties_ptr;
} */


// Private functions

void HyperCubes::compute_reference_vertex_in_dim(odesolver::DimensionIteratorC &reference_vertices_, GridComputationWrapper &grcompwrap, int dim_index, int total_number_of_cubes, int maximum_depth) const
{
    if(total_number_of_cubes == 0)
        total_number_of_cubes = grcompwrap.expanded_depth_per_cube_.size();
    if(maximum_depth == 0)
        maximum_depth = grcompwrap.expanded_cube_indices_.dim_size();

    for(auto depth_index = 0; depth_index < maximum_depth; depth_index++)
    {
        int accum_n_branch_per_depth = accum_n_branches_per_depth[depth_index][dim_index];
        int n_branch_per_depth = n_branches_per_depth[depth_index][dim_index];
        int depth_weight_divisor = accum_n_branches_per_dim[dim_index][depth_index + 1];
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(grcompwrap.expanded_cube_indices_[depth_index].begin(), grcompwrap.expanded_depth_per_cube_.begin(), reference_vertices_.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(grcompwrap.expanded_cube_indices_[depth_index].begin() + total_number_of_cubes, grcompwrap.expanded_depth_per_cube_.begin() + total_number_of_cubes, reference_vertices_.end())),
                         compute_depth_vertex_coor_weight(n_branch_per_depth, accum_n_branch_per_depth, depth_weight_divisor, accum_n_branches_per_dim[dim_index]));
    }
}