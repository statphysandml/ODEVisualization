#include <odesolver/grid_computation/grid_computation.hpp>


namespace odesolver {
    namespace gridcomputation {
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
            compute_axis_index_of_coor(const cudaT variable_offset, const cudaT delta_variable, const int n_branch_per_depth_at_dim) :
                variable_offset_(variable_offset), delta_variable_(delta_variable), n_branch_per_depth_at_dim_(n_branch_per_depth_at_dim)
            {}

            __host__ __device__
            int operator()(const cudaT &coordinate)
            {
                return int((coordinate + variable_offset_) / delta_variable_) % n_branch_per_depth_at_dim_;
            }

            const cudaT variable_offset_;
            const cudaT delta_variable_;
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
            finalize_vertex_computation(const cudaT variable_range, const cudaT variable_offset,
                    dev_vec_int const& accum_n_branches_per_dim) :
                    variable_range_(variable_range), variable_offset_(variable_offset),
                    accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
            {}

            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                int reference_vertex = thrust::get<0>(t);
                int maximum_cube_depth  = thrust::get<1>(t) + 1; // = k
                int inner_vertex_coor = thrust::get<2>(t);

                thrust::get<3>(t) = (reference_vertex + inner_vertex_coor) * variable_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth] + variable_offset_;
            }

            const cudaT variable_range_;
            const cudaT variable_offset_;
            const int * accum_n_branches_per_dim_ptr_;
        };


        struct finalize_reference_vertex_computation
        {
            finalize_reference_vertex_computation(const cudaT variable_range, const cudaT variable_offset,
                                        dev_vec_int const& accum_n_branches_per_dim) :
                    variable_range_(variable_range), variable_offset_(variable_offset),
                    accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
            {}

            __host__ __device__
            cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
            {

                return  reference_vertex * variable_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1] + variable_offset_;
            }

            const cudaT variable_range_;
            const cudaT variable_offset_;
            const int * accum_n_branches_per_dim_ptr_;
        };


        struct finalize_center_vertex_computation
        {
            finalize_center_vertex_computation(const cudaT variable_range, const cudaT variable_offset,
                                        dev_vec_int const& accum_n_branches_per_dim) :
                    variable_range_(variable_range), variable_offset_(variable_offset),
                    accum_n_branches_per_dim_ptr_(thrust::raw_pointer_cast(&accum_n_branches_per_dim[0]))
            {}

            __host__ __device__
            cudaT operator()(const int &reference_vertex, const int &maximum_cube_depth)
            {
                return ((reference_vertex) * variable_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1] +
                        (reference_vertex + 1) * variable_range_ / accum_n_branches_per_dim_ptr_[maximum_cube_depth + 1]) / 2 + variable_offset_;
            }

            const cudaT variable_range_;
            const cudaT variable_offset_;
            const int * accum_n_branches_per_dim_ptr_;
        };


        GridComputation::GridComputation(const std::vector<std::vector<int>> n_branches_per_depth,
            const std::vector<std::pair<cudaT, cudaT>> variable_ranges) :
            dim_(variable_ranges.size()),
            n_branches_per_depth_(n_branches_per_depth),
            accum_n_branches_per_dim_(GridComputation::compute_accum_n_branches_per_dim(n_branches_per_depth, variable_ranges.size())),
            accum_n_branches_per_depth_(GridComputation::compute_accum_n_branches_per_depth(n_branches_per_depth, variable_ranges.size())),
            variable_ranges_(variable_ranges)
        {}

        //[ Static functions

        thrust::host_vector<thrust::host_vector<int>> GridComputation::compute_accum_n_branches_per_dim(const std::vector< std::vector<int>> &n_branches_per_depth, const uint dim)
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

        thrust::host_vector<thrust::host_vector<int>> GridComputation::compute_accum_n_branches_per_depth(const std::vector< std::vector<int>> &n_branches_per_depth, const uint dim)
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

        GridComputationWrapper GridComputation::project_coordinates_on_expanded_cube_and_depth_per_cube_indices(const devdat::DevDatC &coordinates, bool coordinates_on_grid, int depth) const
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
                cudaT variable_dim_range = (variable_ranges_[dim_index].second - variable_ranges_[dim_index].first);
                cudaT variable_range_left = variable_ranges_[dim_index].first;

                cudaT variable_offset;
                if(coordinates_on_grid) {
                    variable_offset = 0.5*variable_dim_range/accum_n_branches_per_dim_[dim_index][depth + 1]; // For avoidance of rounding errors -> corresponds to half of the width of the smallest considered cube
                }
                else  {
                    variable_offset = 0.0;
                }

                dev_vec temp_coordinates(coordinates[dim_index].begin(), coordinates[dim_index].end());

                // Shift coordinates to reference system (most left coordinate == 0)
                thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), temp_coordinates.begin(),
                [variable_range_left] __host__ __device__(const cudaT &coor) { return coor - variable_range_left; });

                for(auto depth_index = 0; depth_index < grcompwrap.expanded_element_indices_.dim_size(); depth_index++)
                {

                    cudaT delta_variable = variable_dim_range / accum_n_branches_per_dim_[dim_index][depth_index + 1]; // corresponds to the width of the considered cube

                    // Compute axis indices
                    dev_vec_int axis_index(total_number_of_cubes, 0); // corresponds to the index of the considered axis
                    thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), axis_index.begin(),
                                    compute_axis_index_of_coor(variable_offset, delta_variable, n_branches_per_depth_[depth_index][dim_index]));

                    // Shift coordinates to corresponding new reference system in considered depth
                    thrust::transform(temp_coordinates.begin(), temp_coordinates.end(), axis_index.begin(),
                                    temp_coordinates.begin(),
                    [delta_variable] __host__ __device__ (const cudaT &coor, const cudaT &cube_index)
                    {
                        return coor - (cube_index * delta_variable);
                    });

                    // Add axis index to expanded cube indices
                    thrust::transform(axis_index.begin(), axis_index.end(),
                                    grcompwrap.expanded_element_indices_[depth_index].begin(),
                                    grcompwrap.expanded_element_indices_[depth_index].begin(),
                                    sum_axis_indices_to_cube_index(accum_n_branches_per_depth_[depth_index][dim_index]));
                }
            }

            return grcompwrap;
        }

        GridComputationWrapper GridComputation::project_leaves_on_expanded_cube_and_depth_per_cube_indices(std::vector<odesolver::collections::Leaf*> &leaves, int depth) const
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

            thrust::host_vector<int> host_expanded_element_indices ((depth + 1) * total_number_of_cubes, 0);
            for(auto depth_index = 0; depth_index < grcompwrap.expanded_element_indices_.dim_size(); depth_index++)
            {
                thrust::transform(thrust::host, leaves.begin(), leaves.end(), host_expanded_element_indices.begin() + total_number_of_cubes * depth_index, [depth_index] (const odesolver::collections::Leaf* leaf) { return leaf->get_ith_depth_index(depth_index); });
            }
            grcompwrap.expanded_element_indices_.fill_by_vec(host_expanded_element_indices);
            return grcompwrap;
        }

        GridComputationWrapper GridComputation::project_collection_package_on_expanded_cube_and_depth_per_cube_indices(std::vector<odesolver::collections::Collection*> &collection_package, int expected_number_of_cubes, int expected_maximum_depth) const
        {
            // Expand collections
            GridComputationWrapper grid_computation_wrapper(expected_number_of_cubes, expected_maximum_depth);
            
            odesolver::collections::CollectionExpander collectionsexpander(expected_maximum_depth, collection_package.size());

            // Fill vectors of length number of collections
            collectionsexpander.extract_collection_information(collection_package);
            
            // Fill vectors of length total number of cubes
            collectionsexpander.expand_collection_information(
                collection_package,
                grid_computation_wrapper.expanded_element_indices_,
                grid_computation_wrapper.expanded_depth_per_element_
            );

            // Testing
            if(monitor)
                grid_computation_wrapper.print_expanded_vectors();
            return grid_computation_wrapper;
        }

        void GridComputation::compute_reference_vertices(devdat::DevDatC &reference_vertices, GridComputationWrapper &grcompwrap, int maximum_depth)
        {
            for(auto dim_index = 0; dim_index < dim_; dim_index++) {
                compute_reference_vertex_in_dim(reference_vertices[dim_index], grcompwrap, dim_index, maximum_depth);

                // Compute delta range
                cudaT variable_dim_range = (variable_ranges_[dim_index].second - variable_ranges_[dim_index].first);
                cudaT variable_offset = variable_ranges_[dim_index].first;

                // Finalize computation of the device reference vertex
                thrust::transform(reference_vertices[dim_index].begin(), reference_vertices[dim_index].end(), grcompwrap.expanded_depth_per_element_.begin(), reference_vertices[dim_index].begin(),
                                finalize_reference_vertex_computation(variable_dim_range, variable_offset, accum_n_branches_per_dim_[dim_index]));
            }
        }

        devdat::DevDatC GridComputation::compute_reference_vertices(GridComputationWrapper &grcompwrap)
        {
            // Initialize reference_vertices
            auto total_number_of_cubes = grcompwrap.expanded_depth_per_element_.size();
            auto reference_vertices = devdat::DevDatC(dim_, total_number_of_cubes);
            
            // Compute reference_vertices
            compute_reference_vertices(reference_vertices, grcompwrap);

            return std::move(reference_vertices);
        }

        void GridComputation::compute_cube_vertices(devdat::DevDatC &cube_vertices, GridComputationWrapper &grcompwrap, int maximum_depth)
        {
            auto total_number_of_cubes = grcompwrap.expanded_depth_per_element_.size();

            for(auto dim_index = 0; dim_index < dim_; dim_index++)
            {
                // Generate device vector of reference vertices for each vector
                devdat::DevDatC reference_vertices_wrapper(1, total_number_of_cubes, 0.0);
                devdat::DimensionIteratorC& reference_vertices = reference_vertices_wrapper[0];
                compute_reference_vertex_in_dim(reference_vertices, grcompwrap, dim_index, maximum_depth);

                // Testing -> Can be used as test without regarding the correct reference vertices
                // print_range("Reference vertices in dimension " + std::to_string(dim_index + 1), reference_vertices.begin(), reference_vertices.end());
                /* // Compute delta ranges
                * cudaT variable_dim_range = (variable_ranges[dim_index].second - variable_ranges[dim_index].first);
                * cudaT variable_offset = variable_ranges[dim_index].first;
                * thrust::transform(reference_vertices.begin(), reference_vertices.end(), grcompwrap.expanded_depth_per_element_.begin(), reference_vertices.begin(),
                                finalize_reference_vertex_computation(variable_dim_range, variable_offset, accum_n_branches_per_dim[dim_index])) */

                // Preparations for the expansion to vertices

                // Repeat reference vertex according to the number of vertices per cube
                repeated_range<dev_iterator> rep_ref_vertex_iterator(
                    reference_vertices.begin(),
                    reference_vertices.end(),
                    pow(2, dim_));

                // Repeat maximum depth values according to the number of vertices per cube
                repeated_range<dev_iterator_int> rep_ref_depth_per_cube_iterator(
                    grcompwrap.expanded_depth_per_element_.begin(),
                    grcompwrap.expanded_depth_per_element_.end(),
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
                cudaT variable_dim_range = (variable_ranges_[dim_index].second - variable_ranges_[dim_index].first);
                cudaT variable_offset = variable_ranges_[dim_index].first;

                // Finalize computation of device vertex
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(rep_ref_vertex_iterator.begin(), rep_ref_depth_per_cube_iterator.begin(), rep_inner_vertex_coors.begin(), cube_vertices[dim_index].begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(rep_ref_vertex_iterator.end(), rep_ref_depth_per_cube_iterator.end(), rep_inner_vertex_coors.end(), cube_vertices[dim_index].end())),
                                finalize_vertex_computation(variable_dim_range, variable_offset, accum_n_branches_per_dim_[dim_index]));

                // Testing
                /* if(monitor)
                    print_range("Vertices in dimension " + std::to_string(dim_index + 1), cube_vertices[dim_index].begin(), cube_vertices[dim_index].end()); */
            }
        }

        devdat::DevDatC GridComputation::compute_cube_vertices(GridComputationWrapper &grcompwrap)
        {
            // Initialize vertices
            auto total_number_of_cubes = grcompwrap.expanded_depth_per_element_.size();
            auto cube_vertices = devdat::DevDatC(dim_, total_number_of_cubes * pow(2, dim_));
            
            // Compute vertices
            compute_cube_vertices(cube_vertices, grcompwrap);

            return std::move(cube_vertices);
        }

        void GridComputation::compute_cube_center_vertices(devdat::DevDatC &center_vertices, GridComputationWrapper &grcompwrap, int maximum_depth)
        {
            for(auto dim_index = 0; dim_index < dim_; dim_index++) {
                // Generate device vector of reference vertices for each vector
                compute_reference_vertex_in_dim(center_vertices[dim_index], grcompwrap, dim_index, maximum_depth);

                // Finalize

                // Compute delta range
                cudaT variable_dim_range = (variable_ranges_[dim_index].second - variable_ranges_[dim_index].first);
                cudaT variable_range_left = variable_ranges_[dim_index].first;

                // Finalize computation of device center vertex
                thrust::transform(center_vertices[dim_index].begin(), center_vertices[dim_index].end(),
                                grcompwrap.expanded_depth_per_element_.begin(), center_vertices[dim_index].begin(),
                                finalize_center_vertex_computation(variable_dim_range, variable_range_left,
                                                                    accum_n_branches_per_dim_[dim_index]));

                // Testing
                if(monitor)
                    print_range("Cube center vertices in dimension " + std::to_string(dim_index + 1),
                                center_vertices[dim_index].begin(), center_vertices[dim_index].end());
            }
        }

        devdat::DevDatC GridComputation::compute_cube_center_vertices(GridComputationWrapper &grcompwrap)
        {
            // Initialize center_vertices
            auto total_number_of_cubes = grcompwrap.expanded_depth_per_element_.size();
            auto center_vertices = devdat::DevDatC(dim_, total_number_of_cubes);
                
            // Compute center_vertices
            compute_cube_center_vertices(center_vertices, grcompwrap);

            return std::move(center_vertices);
        }

        // Getter functions

        const std::vector<std::vector<int>> GridComputation::n_branches_per_depth() const
        {
            return n_branches_per_depth_;
        }

        const std::vector<std::pair<cudaT, cudaT>> GridComputation::variable_ranges() const
        {
            return variable_ranges_;
        }

        size_t GridComputation::dim() const
        {
            return dim_;
        }

        // Protected functions

        void GridComputation::compute_reference_vertex_in_dim(devdat::DimensionIteratorC &reference_vertices, GridComputationWrapper &grcompwrap, int dim_index, int maximum_depth) const
        {
            if(maximum_depth == 0)
                maximum_depth = grcompwrap.expanded_element_indices_.dim_size();

            for(auto depth_index = 0; depth_index < maximum_depth; depth_index++)
            {
                int accum_n_branch_per_depth = accum_n_branches_per_depth_[depth_index][dim_index];
                int n_branch_per_depth = n_branches_per_depth_[depth_index][dim_index];
                int depth_weight_divisor = accum_n_branches_per_dim_[dim_index][depth_index + 1];
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(grcompwrap.expanded_element_indices_[depth_index].begin(), grcompwrap.expanded_depth_per_element_.begin(), reference_vertices.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(grcompwrap.expanded_element_indices_[depth_index].end(), grcompwrap.expanded_depth_per_element_.end(), reference_vertices.end())),
                                compute_depth_vertex_coor_weight(n_branch_per_depth, accum_n_branch_per_depth, depth_weight_divisor, accum_n_branches_per_dim_[dim_index]));
            }
        }
    }
}
