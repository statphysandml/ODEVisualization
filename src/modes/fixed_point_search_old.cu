#include <odesolver/modes/fixed_point_search.hpp>


namespace odesolver {
    namespace modes {
        struct greater_than_zero
        {
            template< typename T>
            __host__ __device__
            T operator()(const T &val) const
            {
                return val > 0;
            }
        };


        // Checks if the given number of positive signs is equal to 0 or to upper bound.
        // If this is not the case, the given cube contains definitely no fixed point.
        // With status, the previous status is taken into account (if it has been recognized already as no fixed point)
        struct check_for_no_fixed_point
        {
            check_for_no_fixed_point(const int upper_bound): upper_bound_(upper_bound)
            {}

            __host__ __device__
            bool operator()(const int &val, const bool& status) const
            {
                return ((val == upper_bound_) or (val == 0)) or status;
            }

            const int upper_bound_;
        };


        // FixedPointSearch Constructors

        FixedPointSearch::FixedPointSearch(
            const json params,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        ) : ODEVisualization(params, flow_equations_ptr, jacobians_ptr, computation_parameters_path),
            dim_(get_entry<json>("flow_equation")["dim"].get<cudaT>()),
            maximum_recursion_depth_(get_entry<int>("maximum_recursion_depth"))
        {
            n_branches_per_depth_ = odesolver::util::json_to_vec_vec<int>(get_entry<json>("n_branches_per_depth"));
            variable_ranges_ = odesolver::util::json_to_vec_pair<double>(get_entry<json>("variable_ranges"));

            // Tests
            if (n_branches_per_depth_.size() < maximum_recursion_depth_)
            {
                std::cout << "\nERROR: Maximum recursion depth " << maximum_recursion_depth_
                        << " is higher than the available number of branches per depth " << n_branches_per_depth_.size()
                        << std::endl;
                std::exit(EXIT_FAILURE);
            }

            for (const auto &n_branches : n_branches_per_depth_)
            {
                if (n_branches.size() != dim_) {
                    std::cout << "\nERROR: Number of branches per depth " << n_branches.size() << " do not coincide with dimension " << dim_ <<  std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }

            if(variable_ranges_.size() != dim_)
            {
                std::cout << "\nERROR: Number of variable ranges " << variable_ranges_.size() << " do not coincide with dimension" << dim_ << std::endl;
                std::exit(EXIT_FAILURE);
            }

            if(flow_equations_ptr_->get_dim() != dim_)
            {
                std::cout << "\nERROR: Dimensions and number of flow equation do not coincide" << dim_ << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

        FixedPointSearch FixedPointSearch::generate(
            const int maximum_recursion_depth,
            const std::vector< std::vector<int> > n_branches_per_depth,
            const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        )
        {
            return FixedPointSearch(
                json {{"maximum_recursion_depth", maximum_recursion_depth},
                    {"n_branches_per_depth", n_branches_per_depth},
                    {"variable_ranges", variable_ranges}},
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path
            );
        }

        FixedPointSearch FixedPointSearch::from_file(
            const std::string rel_config_dir,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        )
        {
            return FixedPointSearch(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path
            );
        }
        

        void FixedPointSearch::compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs)
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

        // Main functions
        
        void FixedPointSearch::eval(std::string memory_usage)
        {
            if(memory_usage == "dynamic")
                evaluate_with_dynamic_memory();
            else
                evaluate_with_preallocated_memory();
            if(leaves_.size() == 0)
            {
                std::cout << "No fixed points have been found within in the given range and based on the provided variable ranges" << std::endl;
            }
            else
            {
                project_leaves_on_cube_centers();
            }
        }

        void FixedPointSearch::evaluate_with_dynamic_memory()
        {
            odesolver::recursivesearch::DynamicRecursiveGridComputation dynamic_recursive_grid_computation(
                computation_parameters_.number_of_cubes_per_gpu_call_,
                computation_parameters_.maximum_number_of_gpu_calls_
            );

            odesolver::DevDatC vertices;
            odesolver::DevDatC vertex_velocities;

            thrust::host_vector<int> host_indices_of_pot_fixed_points;
            
            dynamic_recursive_grid_computation.initialize(
                odesolver::util::json_to_vec_vec<int>(get_entry<json>("n_branches_per_depth")),
                odesolver::util::json_to_vec_pair<double>(get_entry<json>("variable_ranges")), odesolver::recursivesearch::DynamicRecursiveGridComputation::CubeVertices
            );

            while(!dynamic_recursive_grid_computation.finished())
            {
                // Compute vertices
                dynamic_recursive_grid_computation.next(vertices);
                
                // Compute vertex velocities
                vertex_velocities = odesolver::DevDatC(vertices.dim_size(), vertices.n_elems());
                compute_flow(vertices, vertex_velocities, flow_equations_ptr_.get());
                // hypercubes.determine_vertex_velocities(flow_equations_ptr_));
            
                // Determine potential fixed points
                host_indices_of_pot_fixed_points = determine_potential_fixed_points(vertex_velocities);
            
                // Generate new collections and derive leaves based on collections and indices of potential fixed points
                generate_new_collections_and_leaves(
                    host_indices_of_pot_fixed_points,
                    dynamic_recursive_grid_computation.get_collection_package(),
                    dynamic_recursive_grid_computation.get_buffer()
                );
            }
        }

        void FixedPointSearch::evaluate_with_preallocated_memory()
        {
            // Initialize grid computation wrapper
            odesolver::gridcomputation::GridComputationWrapper grid_computation_wrapper(computation_parameters_.number_of_cubes_per_gpu_call_, maximum_recursion_depth_);

            // Initialize vertices
            odesolver::DevDatC vertices(dim_, computation_parameters_.number_of_cubes_per_gpu_call_ * pow(2, dim_));

            // Initialize vertex velocities
            odesolver::DevDatC vertex_velocities(dim_, computation_parameters_.number_of_cubes_per_gpu_call_ * pow(2, dim_));

            // Initialize vector for storing indices of potential fixed points
            thrust::host_vector<int> host_indices_of_pot_fixed_points;

            // Initialize recursive grid computation
            odesolver::recursivesearch::StaticRecursiveGridComputation static_recursive_grid_computation(
                maximum_recursion_depth_,
                computation_parameters_.number_of_cubes_per_gpu_call_,
                computation_parameters_.maximum_number_of_gpu_calls_
            );
            
            static_recursive_grid_computation.initialize(
                odesolver::util::json_to_vec_vec<int>(get_entry<json>("n_branches_per_depth")),
                odesolver::util::json_to_vec_pair<double>(get_entry<json>("variable_ranges")), odesolver::recursivesearch::DynamicRecursiveGridComputation::CubeVertices
            );

            while(!static_recursive_grid_computation.finished())
            {
                // Compute vertices
                static_recursive_grid_computation.next(vertices);

                // Compute vertex velocities
                vertex_velocities.set_N(vertices.n_elems());
                compute_flow(vertices, vertex_velocities, flow_equations_ptr_.get());
                // hypercubes.determine_vertex_velocities(flow_equations_ptr_));
            
                // Determine potential fixed points
                host_indices_of_pot_fixed_points = determine_potential_fixed_points(vertex_velocities);
            
                // Generate new collections and derive leaves based on collections and indices of potential fixed points
                generate_new_collections_and_leaves(
                    host_indices_of_pot_fixed_points,
                    static_recursive_grid_computation.get_collection_package(),
                    static_recursive_grid_computation.get_buffer()
                );
            }
        }
        
        void FixedPointSearch::project_leaves_on_cube_centers()
        {
            // Compute vertices of leaves
            std::vector<odesolver::collections::Leaf*> leaves;
            std::transform(leaves_.begin(), leaves_.end(), std::back_inserter(leaves), [](const std::shared_ptr<odesolver::collections::Leaf>& leaf_ptr) {
                return leaf_ptr.get();
            });
            
            odesolver::gridcomputation::GridComputation hypercubes(n_branches_per_depth_, variable_ranges_);
            
            odesolver::gridcomputation::GridComputationWrapper grcompwrap = hypercubes.project_leaves_on_expanded_cube_and_depth_per_cube_indices(leaves);
                        
            // Get center vertices
            fixed_points_ = hypercubes.compute_cube_center_vertices(grcompwrap);
        }

        const std::vector<std::shared_ptr<odesolver::collections::Leaf>> FixedPointSearch::leaves() const
        {
            return leaves_;
        }

        const odesolver::DevDatC FixedPointSearch::fixed_points() const
        {
            return fixed_points_;
        }

        // Iterate over collections and generate new collections based on the indices of pot fixed points
        void FixedPointSearch::generate_new_collections_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_fixed_points, const std::vector<odesolver::collections::Collection*> &collections, odesolver::collections::Buffer &buffer)
        {
            int n_new_collections = 0;
            int n_new_leaves = 0;

            // No potential fixed points have been found
            if(host_indices_of_pot_fixed_points.size() > 0)
            {
                // Initial conditions
                auto pot_fixed_point_iterator = host_indices_of_pot_fixed_points.begin();
                int index_offset = 0;

                // Iterate over collections
                for(const auto &collection : collections)
                {
                    // Get first potential fix point -> is defined with respect to 0
                    int index_of_pot_fixed_point = *pot_fixed_point_iterator - index_offset; // (-1 to undo offset) -> not used anymore (why initially used??)

                    // Fix points have been found in collection
                    if(index_of_pot_fixed_point < collection->size())
                    {
                        // Inspect fixed points
                        while(index_of_pot_fixed_point < collection->size() and pot_fixed_point_iterator != host_indices_of_pot_fixed_points.end())
                        {
                            // Compute parent collection indices
                            std::vector<int> parent_cube_indices(collection->get_parent_indices());
                            parent_cube_indices.push_back(index_of_pot_fixed_point + collection->get_internal_start_index());

                            // Generate new collections
                            if(collection->get_depth() + 1 < maximum_recursion_depth_) {
                                buffer.append_collection(0, odesolver::collections::compute_internal_end_index(n_branches_per_depth_[collection->get_depth() + 1]), parent_cube_indices);
                                n_new_collections++;
                            }
                            else // Found solution -> Generate new leaf
                            {
                                leaves_.push_back(std::make_shared<odesolver::collections::Leaf>(parent_cube_indices));
                                n_new_leaves++;
                            }
                            // Update
                            pot_fixed_point_iterator++;
                            index_of_pot_fixed_point = *pot_fixed_point_iterator - index_offset; // (-1 to undo offset) -> not used anymore (why initially used??)
                        }
                    }

                    // Update index offset
                    index_offset += collection->size();
                }

                assert(host_indices_of_pot_fixed_points.size() ==  n_new_collections + n_new_leaves && "Number of new collections and number of potential fixed points do not coincide");
            }

            // buffer_.add_collections(new_collections);
            // leaves_.insert(leaves_.end(), new_leaves.begin(), new_leaves.end());

            /* if(monitor) {
                std::cout << "\n### New collections" << std::endl;
                buffer_.get_collections_info(new_collections);
            } */
        }

        thrust::host_vector<int> FixedPointSearch::determine_potential_fixed_points(odesolver::DevDatC& vertex_velocities)
        {
            auto total_number_of_cubes = int(vertex_velocities.n_elems() / pow(2, dim_));

            auto number_of_vertices = vertex_velocities.n_elems(); // to avoid a pass of this within the lambda capture
            thrust::host_vector<dev_vec_bool> velocity_sign_properties(dim_);
            thrust::generate(velocity_sign_properties.begin(), velocity_sign_properties.end(), [number_of_vertices]() { return dev_vec_bool (number_of_vertices, false); });

            // Initial potential fixed points -> at the beginning all cubes contain potential fixed points ( false = potential fixed point )
            dev_vec_bool pot_fixed_points(total_number_of_cubes, false);
            for(auto dim_index = 0; dim_index < dim_; dim_index ++)
            {
                // Turn vertex_velocities into an array with 1.0 and 0.0 for change in sign
                thrust::transform(vertex_velocities[dim_index].begin(), vertex_velocities[dim_index].end(), velocity_sign_properties[dim_index].begin(), greater_than_zero());

                // Initialize a vector for sign checks
                dev_vec_int summed_positive_signs(total_number_of_cubes, 0); // Contains the sum of positive signs within each cube
                FixedPointSearch::compute_summed_positive_signs_per_cube(velocity_sign_properties[dim_index], summed_positive_signs);

                // Testing
                if(monitor)
                    print_range("Summed positive signs in dim " + std::to_string(dim_index), summed_positive_signs.begin(), summed_positive_signs.end());

                // Check if the sign has changed in this component (dimension), takes the previous status into account
                thrust::transform(summed_positive_signs.begin(), summed_positive_signs.end(), pot_fixed_points.begin(), pot_fixed_points.begin(), check_for_no_fixed_point(pow(2, dim_)));
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
    }
}