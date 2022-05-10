#include <odesolver/modes/visualization.hpp>


namespace odesolver {
    namespace modes {
        struct finalize_sample_around_saddle_point
        {
            finalize_sample_around_saddle_point(const cudaT coordinate_val, const cudaT shift) :
                coordinate_val_(coordinate_val), shift_(shift)
            {}
            __host__ __device__
            cudaT operator()(const cudaT &sampled_val)
            {

                return coordinate_val_ +  shift_ * sampled_val;
            }

            const cudaT coordinate_val_;
            const cudaT shift_;
        };


        struct normalize_by_square_root
        {
            __host__ __device__
            cudaT operator() (const cudaT &val1, const cudaT &val2) {
                return val1 / std::sqrt(val2);
            }
        };


        struct sum_square
        {
            __host__ __device__
            cudaT operator() (const cudaT &val1, const cudaT &val2) {
                return val1 + val2 * val2;
            }
        };

        struct sum_manifold_eigenvector
        {
            sum_manifold_eigenvector(const cudaT vector_elem) : vector_elem_(vector_elem)
            {}

            __host__ __device__
            cudaT operator() (const cudaT &previous_val, const cudaT random_number) {
                return previous_val + random_number * vector_elem_;
            }

            const cudaT vector_elem_;
        };


        Visualization::Visualization(
            const json params,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        ) : ODEVisualization(params, flow_equations_ptr, jacobians_ptr, computation_parameters_path),
            dim_(get_entry<json>("flow_equation")["dim"].get<cudaT>()),
            n_branches_(get_entry<std::vector<int>>("n_branches")),
            partial_lambda_ranges_(odesolver::util::json_to_vec_pair<double>(get_entry<json>("lambda_ranges"))),
            fixed_lambdas_(odesolver::util::json_to_vec_vec<double>(get_entry<json>("fixed_lambdas")))
        {
            if (n_branches_.size() != dim_) {
                std::cout << "\nERROR: Number of branches per depth " << n_branches_.size() << " do not coincide with dimension " << dim_ <<  std::endl;
                std::exit(EXIT_FAILURE);
            }

            if(partial_lambda_ranges_.size() + fixed_lambdas_.size() != dim_)
            {
                std::cout << "\nERROR: Number of lambda ranges and fix lambdas " << partial_lambda_ranges_.size() << ", " << fixed_lambdas_.size() << " do not coincide with dimension " << dim_ << std::endl;
                std::exit(EXIT_FAILURE);
            }

            if(flow_equations_ptr_->get_dim() != dim_)
            {
                std::cout << "\nERROR: Dimensions and number of flow equation do not coincide" << dim_ << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // Check consistent definition of n_branches, lambda_ranges and fixed_lambdas
            auto number_of_ones = 0;
            for(auto &n_branch: n_branches_)
            {
                if(n_branch == 1)
                    number_of_ones += 1;
            }
            if(number_of_ones != fixed_lambdas_.size())
            {
                std::cout << "\nERROR: Inconsistent definition of n_branches and fixed_lambdas -> cannot expand lambda range for n_branch=1" << dim_ << std::endl;
                std::exit(EXIT_FAILURE);
            }

            for(auto n_branch_index = 0; n_branch_index < n_branches_.size(); n_branch_index++)
            {
                if(n_branches_[n_branch_index] == 1)
                    indices_of_fixed_lambdas_.push_back(n_branch_index);
            }
        }

        Visualization Visualization::from_file(
            const std::string rel_config_dir,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        )
        {
            return Visualization(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path
            );
        }

        Visualization Visualization::from_parameters(
            const std::vector<int> n_branches,
            const std::vector<std::pair<cudaT, cudaT>> lambda_ranges,
            const std::vector<std::vector<cudaT>> fixed_lambdas,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        )
        {
            return Visualization(
                json {{"n_branches", n_branches},
                    {"partial_lambda_ranges", lambda_ranges},
                    {"fixed_lambdas", fixed_lambdas}},
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path
            );
        }

        /* void Visualization::append_explicit_points_parameters(
                std::vector< std::vector<cudaT> > explicit_points,
                std::string source_root_dir,
                bool source_relative_path,
                std::string source_file_dir)
        {
            // Will be imported from file
            if(explicit_points.size() == 0)
            {
                // ToDo: Correct?? -> No
                std::string path = get_absolute_path(path_parameters.get_base_path() + source_file_dir + "/fixed_points.json", source_relative_path);
                params["explicit_points"]["from_file"] = path;
            }
                // Explicit
            else
            {
                json j;
                for(auto &fixed_point : explicit_points)
                    j.push_back(fixed_point);
                params["explicit_points"] = json {{"number_of_explicit_points", explicit_points.size()}, {"explicit_points", j}};
            }
        } */


        Visualization::ComputeVertexVelocitiesParameters::ComputeVertexVelocitiesParameters(const json params) : Parameters(params),
            skip_fixed_lambdas_(get_entry<bool>("skip_fixed_lambdas")),
            with_vertices_(get_entry<bool>("with_vertices"))
        {}

        Visualization::ComputeVertexVelocitiesParameters::ComputeVertexVelocitiesParameters(
            const bool skip_fixed_lambdas, const bool with_vertices
        ) : ComputeVertexVelocitiesParameters(
                json {{"skip_fixed_lambdas", skip_fixed_lambdas},
                    {"with_vertices", with_vertices}}
        )
        {}


        Visualization::ComputeSeparatrizesParameters::ComputeSeparatrizesParameters(const json params) : Parameters(params),
            N_per_eigen_dim_(get_entry<uint>("N_per_eigen_dim")),
            shift_per_dim_(get_entry<std::vector<double>>("shift_per_dim"))
        {}

        Visualization::ComputeSeparatrizesParameters::ComputeSeparatrizesParameters(
                const uint N_per_eigen_dim,
                const std::vector<double> shift_per_dim
        ) : ComputeSeparatrizesParameters(
                json {{"N_per_eigen_dim", N_per_eigen_dim},
                    {"shift_per_dim", shift_per_dim}}
        )
        {}

        std::vector<std::vector<cudaT>> Visualization::get_fixed_points() const
        {
            return odesolver::util::json_to_vec_vec<double>(get_entry<json>("fixed_points")["fixed_points"]);
        }

        void Visualization::evaluate_vertices(std::string rel_dir, bool skip_fixed_lambdas, bool with_vertices)
        {
            std::ofstream os, os_vertices;
            os.open(param_helper::proj::project_root() + rel_dir + "/" + "velocities" + ".dat");

            if(with_vertices)
                os_vertices.open(param_helper::proj::project_root() + rel_dir + "/" + "vertices" + ".dat");

            std::vector<int> skip_iterators_in_dimensions {};
            for(auto i = 0; i < n_branches_.size(); i++)
            {
                if(n_branches_[i] == 1)
                    skip_iterators_in_dimensions.push_back(i);
            }

            odesolver::grid_computation::DynamicRecursiveGridComputation dynamic_recursive_grid_computation(
                computation_parameters_.number_of_cubes_per_gpu_call_,
                computation_parameters_.maximum_number_of_gpu_calls_
            );

            odesolver::DevDatC reference_vertices;
            odesolver::DevDatC reference_vertex_velocities;

            odesolver::util::PartialRanges partial_ranges(n_branches_, partial_lambda_ranges_, fixed_lambdas_);

            for(auto i = 0; i < partial_ranges.size(); i++)
            {
                auto lambda_ranges = partial_ranges[i];

                dynamic_recursive_grid_computation.initialize(
                    std::vector<std::vector<int>> {n_branches_},
                    lambda_ranges,
                    odesolver::grid_computation::DynamicRecursiveGridComputation::ReferenceVertices
                );
            
                while(!dynamic_recursive_grid_computation.finished())
                {
                    // Compute vertices
                    dynamic_recursive_grid_computation.next(reference_vertices);
                
                    // Compute vertex velocities
                    reference_vertex_velocities = odesolver::DevDatC(reference_vertices.dim_size(), reference_vertices.n_elems());
                    compute_vertex_velocities(reference_vertices, reference_vertex_velocities, flow_equations_ptr_.get());
            
                    if (!skip_fixed_lambdas) {
                        write_data_to_ofstream(reference_vertex_velocities, os);
                        if (with_vertices)
                            write_data_to_ofstream(reference_vertices, os_vertices);
                    } else {
                        write_data_to_ofstream(reference_vertex_velocities, os, skip_iterators_in_dimensions);
                        if (with_vertices)
                            write_data_to_ofstream(reference_vertices, os_vertices, skip_iterators_in_dimensions);
                    }
                }
            }
            os.close();
            if(with_vertices)
                os_vertices.close();
        }

        /* void Visualization::compute_vertex_velocities_from_parameters(std::string rel_dir)
        {
            auto compute_vertex_velocities_parameters = vp.get_entry<json>("compute_vertex_velocities");
            auto params1 = Visualization::ComputeVertexVelocitiesParameters(compute_vertex_velocities_parameters);
            compute_vertex_velocities(dir, params1.skip_fixed_lambdas, params1.with_vertices);
        } */

        /* void Visualization::compute_separatrizes(const std::string rel_dir,
                                const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges,
                                const std::vector <cudaT> minimum_change_of_state,
                                const cudaT minimum_delta_t, const cudaT maximum_flow_val,
                                const std::vector <cudaT> vicinity_distances,
                                const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
                                const uint N_per_eigen_dim,
                                const std::vector<double> shift_per_dim)
        {
            cudaT delta_t = 0.001;

            // Get saddle fixed points (in coordinates)
            // Sample around fix points
            // Evolve with corresponding particularities

            // odesolver::DevDatC saddle_coordinates = get_saddle_fix_points();

            std::vector < std::vector<cudaT> > potential_saddle_points = vp.get_fixed_points();
            // std::vector < std::vector< std::vector <int> > > eigen_compositions = vp.get_eigen_compositions();

            CoordinateOperatorParameters coordinate_operator_parameters = CoordinateOperatorParameters::from_parameters(vp.path_parameters.theory, potential_saddle_points);
            CoordinateOperator saddle_point_evaluator(coordinate_operator_parameters);
            saddle_point_evaluator.compute_jacobians_and_eigendata(); */
            /* auto eigenvectors_real_part = saddle_point_evaluator.get_real_parts_of_eigenvectors();
            auto eigenvectors_imag_part = saddle_point_evaluator.get_imag_parts_of_eigenvectors();
            auto eigenvalues = saddle_point_evaluator.get_real_parts_of_eigenvalues(); */

        /*     const std::vector<int> saddle_point_indices = saddle_point_evaluator.get_indices_with_saddle_point_characteristics();

            // Iterator through emerging lambda ranges from given fixed lambdas
            odesolver::util::PartialRanges lambda_range_generator(n_branches_, partial_lambda_ranges_, fixed_lambdas_);
            auto c = 0;
            while(!lambda_range_generator.finished())
            {
                std::ofstream os;
                std::string path = vp.get_absolute_path(vp.path_parameters.get_base_path() + "/" + dir + "/", vp.path_parameters.relative_path);
                os.open(path + "separatrices_" + std::to_string(c) + ".dat");

                auto lambda_ranges = lambda_range_generator.next();

                // Retrieve currently considered fixed lambdas
                std::vector< cudaT > fixed_lambdas {};
                std::transform(indices_of_fixed_lambdas.begin(), indices_of_fixed_lambdas.end(), std::back_inserter(fixed_lambdas),
                            [lambda_ranges] (const int& index) { return lambda_ranges[index].first; });
                // Iterate over all saddle points
                for(auto saddle_point_index = 0; saddle_point_index < saddle_point_indices.size(); saddle_point_index++)
                {
                    std::cout << "Performing for saddle point with x = " << potential_saddle_points[saddle_point_index][0] << std::endl; */
                    /* std::vector<std::vector<cudaT>> eigenvector_real_part = eigenvectors_real_part[saddle_point_index];
                    std::vector<std::vector<cudaT>> eigenvector_imag_part = eigenvectors_imag_part[saddle_point_index];
                    std::vector<cudaT> eigenvalue = eigenvalues[saddle_point_index]; */

        /*             auto eigenvector = saddle_point_evaluator.get_eigenvector(saddle_point_index);
                    auto eigenvalue = saddle_point_evaluator.get_eigenvalue(saddle_point_index);

                    std::vector<int> stable_manifold_indices {};
                    std::vector<int> unstable_manifold_indices {};
                    std::vector<std::vector<cudaT>> manifold_eigenvectors(eigenvalue.size());
                    extract_stable_and_unstable_manifolds(eigenvalue, eigenvector, stable_manifold_indices, unstable_manifold_indices, manifold_eigenvectors);

                    compute_separatrizes_of_manifold(
                            potential_saddle_points[saddle_point_index],
                            stable_manifold_indices,
                            manifold_eigenvectors,
                            -1.0*delta_t,
                            boundary_lambda_ranges,
                            minimum_change_of_state,
                            minimum_delta_t, maximum_flow_val,
                            vicinity_distances,
                            observe_every_nth_step, maximum_total_number_of_steps,
                            N_per_eigen_dim,
                            shift_per_dim,
                            os,
                            fixed_lambdas
                    );

                    compute_separatrizes_of_manifold(
                            potential_saddle_points[saddle_point_index],
                            unstable_manifold_indices,
                            manifold_eigenvectors,
                            delta_t,
                            boundary_lambda_ranges,
                            minimum_change_of_state,
                            minimum_delta_t, maximum_flow_val,
                            vicinity_distances,
                            observe_every_nth_step, maximum_total_number_of_steps,
                            N_per_eigen_dim,
                            shift_per_dim,
                            os,
                            fixed_lambdas
                    );
                }

                os.close();
                c++;
            }

            // Consistency check
            auto total_number_of_generated_lambdas = 1;
            for(auto &fix_lambd : fixed_lambdas_)
                total_number_of_generated_lambdas *= fix_lambd.size();
            if(total_number_of_generated_lambdas != c) {
                std::cout
                        << "\nERROR: Something went wrong during the generation of the lambda ranges (possibility for duplicates, etc.)"
                        << dim_ << std::endl;
                std::exit(EXIT_FAILURE);
            }
        } */

        /* void Visualization::compute_separatrizes_from_parameters(const std::string rel_dir)
        { */
            // ToDo: Reactivate
            /* auto evolve_on_condition_parameters = vp.get_entry<json>("evolve_on_condition");
            auto compute_separatrizes_parameters = vp.get_entry<json>("compute_separatrizes");
            auto params3 = Visualization::ComputeSeparatrizesParameters(compute_separatrizes_parameters);
            if(fixed_lambdas_.size() > 0)
            {
                auto conditional_observer_parameters = vp.get_entry<json>("conditional_intersection_observer");
                auto params1 = ConditionalIntersectionObserverParameters(conditional_observer_parameters);
                auto params2 = CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalIntersectionObserverParameters>(evolve_on_condition_parameters);
                compute_separatrizes(dir, params1.boundary_lambda_ranges, params1.minimum_change_of_state,
                                    params1.minimum_delta_t, params1.maximum_flow_val, params1.vicinity_distances,
                                    params2.observe_every_nth_step, params2.maximum_total_number_of_steps,
                                    params3.N_per_eigen_dim, params3.shift_per_dim);
            }
            else
            {
                auto conditional_observer_parameters = vp.get_entry<json>("conditional_range_observer");
                auto params1 = ConditionalRangeObserverParameters(conditional_observer_parameters);
                auto params2 = CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalRangeObserverParameters>(evolve_on_condition_parameters);
                compute_separatrizes(dir, params1.boundary_lambda_ranges, params1.minimum_change_of_state,
                                    params1.minimum_delta_t, params1.maximum_flow_val, std::vector <cudaT> {},
                                    params2.observe_every_nth_step, params2.maximum_total_number_of_steps,
                                    params3.N_per_eigen_dim, params3.shift_per_dim);
            } */
        // }

        /* odesolver::DevDatC Visualization::sample_around_saddle_point(const std::vector<double> coordinate, const std::vector<int> manifold_indices,
                                                        const std::vector<std::vector<cudaT>> manifold_eigenvectors, const std::vector<double> shift_per_dim, const uint N_per_eigen_dim)
        {
            const uint eigen_dim = manifold_indices.size();
            const int N = pow(N_per_eigen_dim, eigen_dim);
            const uint8_t dim = coordinate.size();
            odesolver::DevDatC sampled_points(dim, N, 0);

            // Generate (eigen_dim x N) random numbers
            int discard = 0;
            odesolver::DevDatC random_numbers(eigen_dim, N, 0);
            for(auto eigen_dim_index = 0; eigen_dim_index < eigen_dim; eigen_dim_index++) {
                thrust::transform(
                        thrust::make_counting_iterator(0 + discard),
                        thrust::make_counting_iterator(N + discard),
                        random_numbers[eigen_dim_index].begin(),
                        RandomNormalGenerator());
                discard += N;
            }

            dev_vec sum(N, 0);
            for(auto dim_index=0; dim_index < dim; dim_index++)
            {
                // Iteration over the random_numbers per eigen_dimension
                for(auto eigen_dim_index = 0; eigen_dim_index < eigen_dim; eigen_dim_index++)
                {
                    thrust::transform(sampled_points[dim_index].begin(), sampled_points[dim_index].end(), random_numbers[eigen_dim_index].begin(), sampled_points[dim_index].begin(),
                            sum_manifold_eigenvector(manifold_eigenvectors[manifold_indices[eigen_dim_index]][dim_index]));
                }

                // For latter normalization
                thrust::transform(sum.begin(), sum.end(), sampled_points[dim_index].begin(), sum.begin(), sum_square());
            }

            for(auto dim_index=0; dim_index < dim; dim_index++)
                thrust::transform(sampled_points[dim_index].begin(), sampled_points[dim_index].end(), sum.begin(), sampled_points[dim_index].begin(), normalize_by_square_root());
            // Shift coordinates by random numbers
            for(auto dim_index=0; dim_index < dim; dim_index++)
            {
                // if(std::find(manifold_indices.begin(), manifold_indices.end(), dim_index) != manifold_indices.end())
                // print_range("Sampled points ", sampled_points[eigen_dim_index].begin(), sampled_points[eigen_dim_index].end());
                thrust::transform(
                        sampled_points[dim_index].begin(),
                        sampled_points[dim_index].end(),
                        sampled_points[dim_index].begin(),
                        finalize_sample_around_saddle_point(coordinate[dim_index], shift_per_dim[dim_index]));
                // print_range("Sampled points2 ", sampled_points[eigen_dim_index].begin(), sampled_points[eigen_dim_index].end());
            }
            return sampled_points;
        }

        odesolver::DevDatC Visualization::get_initial_values_to_eigenvector(const std::vector<double> saddle_point, const std::vector<cudaT> eigenvector, const std::vector<double> shift_per_dim)
        {
            const uint8_t dim = saddle_point.size();
            odesolver::DevDatC points(dim, 2, 0);

            for(auto dim_index=0; dim_index < dim; dim_index++)
            {
                auto it = points[dim_index].begin();
                *it = saddle_point[dim_index] + shift_per_dim[dim_index] * eigenvector[dim_index];
                it++;
                *it = saddle_point[dim_index] - shift_per_dim[dim_index] * eigenvector[dim_index];
                // print_range("Points ", points[dim_index].begin(), points[dim_index].end());
            }
            return points;
        }


        void Visualization::extract_stable_and_unstable_manifolds(
                Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenvalue,
                Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenvector,
                std::vector<int> &stable_manifold_indices,
                std::vector<int> &unstable_manifold_indices,
                std::vector<std::vector<cudaT>> &manifold_eigenvectors
        )
        {
            std::vector< std::complex<double> > complex_eigenvals_buffer{};
            std::vector< int > complex_eigenvals_buffer_index{};

            for(auto i = 0; i < eigenvalue.size(); i++)
            {
                if(eigenvalue[i].real() < 0)
                    stable_manifold_indices.push_back(i);
                else
                    unstable_manifold_indices.push_back(i);

                auto eigen_vec = eigenvector.col(i);
                if(eigenvalue[i].imag() != 0)
                {
                    auto it_complex_eigenvals_buffer = std::find(complex_eigenvals_buffer.begin(), complex_eigenvals_buffer.end(), std::conj(eigenvalue[i]));
                    if (it_complex_eigenvals_buffer != complex_eigenvals_buffer.end())
                    {
                        for(auto j = 0; j < eigen_vec.size(); j++)
                        {
                            manifold_eigenvectors[complex_eigenvals_buffer_index[it_complex_eigenvals_buffer - complex_eigenvals_buffer.begin()]].push_back(eigen_vec[j].imag());
                            manifold_eigenvectors[i].push_back(eigen_vec[j].real());
                        }
                    }
                    else
                    {
                        complex_eigenvals_buffer.push_back(eigenvalue[i]);
                        complex_eigenvals_buffer_index.push_back(i);
                    }
                }
                else
                {
                    for(auto j = 0; j < eigen_vec.size(); j++)
                        manifold_eigenvectors[i].push_back(eigen_vec[j].real());
                }
            }
        } */


        /* void Visualization::compute_separatrizes_of_manifold(
                const std::vector<double> saddle_point,
                const std::vector<int> manifold_indices,
                const std::vector<std::vector<cudaT>> manifold_eigenvectors,
                const cudaT delta_t,
                const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges,
                const std::vector <cudaT> minimum_change_of_state,
                const cudaT minimum_delta_t, const cudaT maximum_flow_val,
                const std::vector <cudaT> vicinity_distances,
                const uint observe_every_nth_step, const uint maximum_total_number_of_steps,
                const uint N_per_eigen_dim,
                const std::vector<double> shift_per_dim,
                std::ofstream &os,
                std::vector< cudaT > fixed_lambdas
                ) { */
            // ToDo: Reactivate
            // Perform computation of separatrix for stable manifold
            /* odesolver::DevDatC sampled_coordinates;
            // Single line
            if(manifold_indices.size() == 1)
            {
                sampled_coordinates = get_initial_values_to_eigenvector(saddle_point,
                                                                        manifold_eigenvectors[manifold_indices[0]],
                                                                        shift_per_dim);
            }
            else
            {
                sampled_coordinates = sample_around_saddle_point(saddle_point,
                                                                manifold_indices,
                                                                manifold_eigenvectors,
                                                                shift_per_dim, N_per_eigen_dim);
            }

            if(fixed_lambdas_.size() > 0)
            {
                auto observer = new ConditionalIntersectionObserver(flow_equations_ptr_, sampled_coordinates.size(), os,
                                                                    boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val, vicinity_distances,
                                                                    fixed_lambdas, indices_of_fixed_lambdas);
                observer->initalize_side_counter(sampled_coordinates);

                // for(auto dim_index = 0; dim_index < sampled_coordinates.dim_size(); dim_index++)
                //     print_range("Vertex in dim " + std::to_string(dim_index) + ": ", sampled_coordinates[dim_index].begin(), sampled_coordinates[dim_index].end());
                Evolution<ConditionalIntersectionObserver> evaluator(flow_equations_ptr_, observer, observe_every_nth_step, maximum_total_number_of_steps);

                print_range("Initial point", sampled_coordinates.begin(), sampled_coordinates.end());
                evaluator.evolve_observer_based(sampled_coordinates, delta_t);
                print_range("End point", sampled_coordinates.begin(), sampled_coordinates.end());
            }
            else
            {
                auto observer = new ConditionalRangeObserver(flow_equations_ptr_, sampled_coordinates.size(), os,
                                                            boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);
                // for(auto dim_index = 0; dim_index < sampled_coordinates.dim_size(); dim_index++)
                //     print_range("Vertex in dim " + std::to_string(dim_index) + ": ", sampled_coordinates[dim_index].begin(), sampled_coordinates[dim_index].end());
                Evolution<ConditionalRangeObserver> evaluator(flow_equations_ptr_, observer, observe_every_nth_step, maximum_total_number_of_steps);

                print_range("Initial point", sampled_coordinates.begin(), sampled_coordinates.end());
                evaluator.evolve_observer_based(sampled_coordinates, delta_t);
                print_range("End point", sampled_coordinates.begin(), sampled_coordinates.end());
            } */
        // }
    }
}