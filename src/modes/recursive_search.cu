#include <odesolver/modes/recursive_search.hpp>


namespace odesolver {
    namespace modes {
        // RecursiveSearch Constructors

        RecursiveSearch::RecursiveSearch(
            const json params,
            std::shared_ptr<odesolver::modes::RecursiveSearchCriterion> criterion_ptr,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
        ) : ODEVisualization(params, flow_equations_ptr, jacobians_ptr),
            criterion_ptr_(criterion_ptr),
            dim_(get_entry<json>("flow_equation")["dim"].get<cudaT>()),
            maximum_recursion_depth_(get_entry<int>("maximum_recursion_depth")),
            number_of_cubes_per_gpu_call_(get_entry<int>("number_of_cubes_per_gpu_call")),
            maximum_number_of_gpu_calls_(get_entry<int>("maximum_number_of_gpu_calls"))
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

        RecursiveSearch RecursiveSearch::generate(
            const int maximum_recursion_depth,
            const std::vector<std::vector<int>> n_branches_per_depth,
            const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
            std::shared_ptr<odesolver::modes::RecursiveSearchCriterion> criterion_ptr,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const int number_of_cubes_per_gpu_call,
            const int maximum_number_of_gpu_calls
        )
        {
            return RecursiveSearch(
                json {{"maximum_recursion_depth", maximum_recursion_depth},
                    {"n_branches_per_depth", n_branches_per_depth},
                    {"variable_ranges", variable_ranges},
                    {"number_of_cubes_per_gpu_call", number_of_cubes_per_gpu_call},
                    {"maximum_number_of_gpu_calls", maximum_number_of_gpu_calls}},
                criterion_ptr,
                flow_equations_ptr,
                jacobians_ptr
            );
        }

        RecursiveSearch RecursiveSearch::from_file(
            const std::string rel_config_dir,
            std::shared_ptr<odesolver::modes::RecursiveSearchCriterion> criterion_ptr,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
        )
        {
            return RecursiveSearch(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
                criterion_ptr,
                flow_equations_ptr,
                jacobians_ptr
            );
        }
        
        // Main functions
        
        void RecursiveSearch::eval(std::string memory_usage)
        {
            if(memory_usage == "dynamic")
                evaluate_with_dynamic_memory();
            else
                evaluate_with_preallocated_memory();
            if(leaves_.size() == 0)
            {
                std::cout << "No fixed points have been found within in the given range and based on the provided variable ranges." << std::endl;
            }
            else
            {
                project_leaves_on_cube_centers();
            }
        }

        void RecursiveSearch::evaluate_with_dynamic_memory()
        {
            odesolver::gridcomputation::DynamicRecursiveGridComputation dynamic_recursive_grid_computation(number_of_cubes_per_gpu_call_, maximum_number_of_gpu_calls_
            );

            odesolver::DevDatC vertices;
            odesolver::DevDatC vertex_velocities;

            thrust::host_vector<int> host_indices_of_pot_solutions;
            
            dynamic_recursive_grid_computation.initialize(
                odesolver::util::json_to_vec_vec<int>(get_entry<json>("n_branches_per_depth")),
                odesolver::util::json_to_vec_pair<double>(get_entry<json>("variable_ranges")), odesolver::gridcomputation::DynamicRecursiveGridComputation::CubeVertices
            );

            while(!dynamic_recursive_grid_computation.finished())
            {
                // Compute vertices
                dynamic_recursive_grid_computation.next(vertices);
                
                // Compute vertex velocities
                vertex_velocities = odesolver::DevDatC(vertices.dim_size(), vertices.n_elems());
                compute_flow(vertices, vertex_velocities, flow_equations_ptr_.get());
                
                // Determine potential solutions
                host_indices_of_pot_solutions = criterion_ptr_->determine_potential_solutions(vertices, vertex_velocities);

                // Generate new collections and derive leaves based on collections and indices of potential fixed points
                generate_new_collections_and_leaves(
                    host_indices_of_pot_solutions,
                    dynamic_recursive_grid_computation.get_collection_package(),
                    dynamic_recursive_grid_computation.get_buffer()
                );
            }
        }

        void RecursiveSearch::evaluate_with_preallocated_memory()
        {
            // Initialize vertices
            odesolver::DevDatC vertices(dim_, number_of_cubes_per_gpu_call_ * pow(2, dim_));

            // Initialize vertex velocities
            odesolver::DevDatC vertex_velocities(dim_, number_of_cubes_per_gpu_call_ * pow(2, dim_));

            // Initialize vector for storing indices of potential fixed points
            thrust::host_vector<int> host_indices_of_pot_solutions;

            // Initialize recursive grid computation
            odesolver::gridcomputation::StaticRecursiveGridComputation static_recursive_grid_computation(
                maximum_recursion_depth_,
                number_of_cubes_per_gpu_call_,
                maximum_number_of_gpu_calls_
            );
            
            static_recursive_grid_computation.initialize(
                odesolver::util::json_to_vec_vec<int>(get_entry<json>("n_branches_per_depth")),
                odesolver::util::json_to_vec_pair<double>(get_entry<json>("variable_ranges")), odesolver::gridcomputation::DynamicRecursiveGridComputation::CubeVertices
            );

            while(!static_recursive_grid_computation.finished())
            {
                // Compute vertices
                static_recursive_grid_computation.next(vertices);

                // Compute vertex velocities
                vertex_velocities.set_N(vertices.n_elems());
                compute_flow(vertices, vertex_velocities, flow_equations_ptr_.get());
                
                // Determine potential solutions
                host_indices_of_pot_solutions = criterion_ptr_->determine_potential_solutions(vertices, vertex_velocities);

                // Generate new collections and derive leaves based on collections and indices of potential fixed points
                generate_new_collections_and_leaves(
                    host_indices_of_pot_solutions,
                    static_recursive_grid_computation.get_collection_package(),
                    static_recursive_grid_computation.get_buffer()
                );
            }
        }
        
        void RecursiveSearch::project_leaves_on_cube_centers()
        {
            // Compute vertices of leaves
            std::vector<odesolver::collections::Leaf*> leaves;
            std::transform(leaves_.begin(), leaves_.end(), std::back_inserter(leaves), [](const std::shared_ptr<odesolver::collections::Leaf>& leaf_ptr) {
                return leaf_ptr.get();
            });
            
            odesolver::gridcomputation::GridComputation hypercubes(n_branches_per_depth_, variable_ranges_);
            
            odesolver::gridcomputation::GridComputationWrapper grcompwrap = hypercubes.project_leaves_on_expanded_cube_and_depth_per_cube_indices(leaves);
                        
            // Get center vertices
            solutions_ = hypercubes.compute_cube_center_vertices(grcompwrap);
        }

        const std::vector<std::shared_ptr<odesolver::collections::Leaf>> RecursiveSearch::leaves() const
        {
            return leaves_;
        }

        const odesolver::DevDatC RecursiveSearch::solutions() const
        {
            return solutions_;
        }

        // Iterate over collections and generate new collections based on the indices of pot fixed points
        void RecursiveSearch::generate_new_collections_and_leaves(const thrust::host_vector<int> &host_indices_of_pot_solutions, const std::vector<odesolver::collections::Collection*> &collections, odesolver::collections::Buffer &buffer)
        {
            int n_new_collections = 0;
            int n_new_leaves = 0;

            // No potential fixed points have been found
            if(host_indices_of_pot_solutions.size() > 0)
            {
                // Initial conditions
                auto pot_solution_iterator = host_indices_of_pot_solutions.begin();
                int index_offset = 0;

                // Iterate over collections
                for(const auto &collection : collections)
                {
                    // Get first potential fix point -> is defined with respect to 0
                    int indices_of_pot_solution = *pot_solution_iterator - index_offset; // (-1 to undo offset) -> not used anymore (why initially used??)

                    // Fix points have been found in collection
                    if(indices_of_pot_solution < collection->size())
                    {
                        // Inspect fixed points
                        while(indices_of_pot_solution < collection->size() and pot_solution_iterator != host_indices_of_pot_solutions.end())
                        {
                            // Compute parent collection indices
                            std::vector<int> parent_cube_indices(collection->get_parent_indices());
                            parent_cube_indices.push_back(indices_of_pot_solution + collection->get_internal_start_index());

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
                            pot_solution_iterator++;
                            indices_of_pot_solution = *pot_solution_iterator - index_offset; // (-1 to undo offset) -> not used anymore (why initially used??)
                        }
                    }

                    // Update index offset
                    index_offset += collection->size();
                }

                assert(host_indices_of_pot_solutions.size() ==  n_new_collections + n_new_leaves && "Number of new collections and number of potential fixed points do not coincide");
            }

            // buffer_.add_collections(new_collections);
            // leaves_.insert(leaves_.end(), new_leaves.begin(), new_leaves.end());

            /* if(monitor) {
                std::cout << "\n### New collections" << std::endl;
                buffer_.get_collections_info(new_collections);
            } */
        }
    }
}