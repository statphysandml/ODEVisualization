#include <odesolver/modes/separatrizes.hpp>


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

        // Separatrizes Constructors

        Separatrizes::Separatrizes(
            const json params,
            std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobians_ptr
        ) : ODEVisualization(params, flow_equations_ptr, jacobians_ptr),
            N_per_eigen_dim_(get_entry<uint>("N_per_eigen_dim")),
            shift_per_dim_(get_entry<std::vector<double>>("shift_per_dim")),
            n_max_steps_(get_entry<uint>("n_max_steps"))
        {}

        Separatrizes Separatrizes::generate(
            const uint N_per_eigen_dim,
            const std::vector<double> shift_per_dim,
            const uint n_max_steps,
            std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobians_ptr
        )
        {
            return Separatrizes(
                json {{"N_per_eigen_dim", N_per_eigen_dim},
                      {"shift_per_dim", shift_per_dim},
                      {"n_max_steps", n_max_steps}},
                flow_equations_ptr,
                jacobians_ptr
            );
        }

        Separatrizes Separatrizes::from_file(
            const std::string rel_config_dir,
            std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<flowequations::JacobianEquationsWrapper> jacobians_ptr
        )
        {
            return Separatrizes(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
                flow_equations_ptr,
                jacobians_ptr
            );
        }

        void Separatrizes::extract_stable_and_unstable_manifolds(odesolver::modes::Jacobians &jacobians, int saddle_point_index, std::vector<int> &stable_manifold_indices, std::vector<int> &unstable_manifold_indices, std::vector<std::vector<cudaT>> &manifold_eigenvectors)
        {
            auto eigenvector = jacobians.get_eigenvector(saddle_point_index);
            auto eigenvalue = jacobians.get_eigenvalue(saddle_point_index);
            
            manifold_eigenvectors.resize(eigenvalue.size());

            std::vector<std::complex<double>> complex_eigenvals_buffer;
            std::vector<int> complex_eigenvals_buffer_index;

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
                    // it_complex_eigenvals_buffer points to the previously complex conjugated eigenvalue
                    // it_complex_eigenvals_buffer - complex_eigenvals_buffer.begin() returns respectively the index ues below to assign the respective eigenvector
                    if(it_complex_eigenvals_buffer != complex_eigenvals_buffer.end())
                    {
                        for(auto j = 0; j < eigen_vec.size(); j++)
                        {
                            manifold_eigenvectors[complex_eigenvals_buffer_index[it_complex_eigenvals_buffer - complex_eigenvals_buffer.begin()]].push_back(eigen_vec[j].real());
                            manifold_eigenvectors[i].push_back(eigen_vec[j].imag());
                        }
                    }
                    else
                    {
                        // Note that in this case no manifold vector has been added so far - this will haben as soon as the conjugate EV has been found
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
        }

        devdat::DevDatC Separatrizes::get_initial_values_to_eigenvector(const std::vector<double> &saddle_point, const std::vector<cudaT> &manifold_eigenvector)
        {
            const size_t dim = saddle_point.size();
            devdat::DevDatC points(dim, 2, 0);

            for(auto dim_index=0; dim_index < dim; dim_index++)
            {
                auto it = points[dim_index].begin();
                *it = saddle_point[dim_index] + shift_per_dim_[dim_index] * manifold_eigenvector[dim_index];
                it++;
                *it = saddle_point[dim_index] - shift_per_dim_[dim_index] * manifold_eigenvector[dim_index];
            }
            return points;
        }

        devdat::DevDatC Separatrizes::sample_around_saddle_point(const std::vector<double> &saddle_point, const std::vector<std::vector<cudaT>> &manifold_eigenvectors, const std::vector<int> &manifold_indices)
        {
            const uint eigen_dim = manifold_indices.size();
            const int N = pow(N_per_eigen_dim_, eigen_dim);
            const size_t dim = saddle_point.size();
            devdat::DevDatC sampled_points(dim, N, 0);

            // Generate (eigen_dim x N) random numbers
            int discard = 0;
            devdat::DevDatC random_numbers(eigen_dim, N, 0);
            for(auto eigen_dim_index = 0; eigen_dim_index < eigen_dim; eigen_dim_index++) {
                thrust::transform(
                        thrust::make_counting_iterator(0 + discard),
                        thrust::make_counting_iterator(N + discard),
                        random_numbers[eigen_dim_index].begin(),
                        odesolver::util::RandomNormalGenerator());
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
                    finalize_sample_around_saddle_point(saddle_point[dim_index], shift_per_dim_[dim_index]));
                // print_range("Sampled points2 ", sampled_points[eigen_dim_index].begin(), sampled_points[eigen_dim_index].end());
            }
            return sampled_points;
        }
    }
}