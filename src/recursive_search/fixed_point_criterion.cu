#include <odesolver/recursive_search/fixed_point_criterion.hpp>


namespace odesolver {
    namespace recursivesearch {
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


        void FixedPointCriterion::compute_summed_positive_signs_per_cube(dev_vec_bool &velocity_sign_properties_in_dim, dev_vec_int &summed_positive_signs)
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

        thrust::host_vector<int> FixedPointCriterion::determine_potential_solutions(odesolver::DevDatC& vertices, odesolver::DevDatC& vertex_velocities)
        {
            auto dim = vertex_velocities.dim_size();
            auto total_number_of_cubes = int(vertex_velocities.n_elems() / pow(2, dim));

            auto number_of_vertices = vertex_velocities.n_elems(); // to avoid a pass of this within the lambda capture
            thrust::host_vector<dev_vec_bool> velocity_sign_properties(dim);
            thrust::generate(velocity_sign_properties.begin(), velocity_sign_properties.end(), [number_of_vertices]() { return dev_vec_bool (number_of_vertices, false); });

            // Initial potential fixed points -> at the beginning all cubes contain potential fixed points ( false = potential fixed point )
            dev_vec_bool pot_fixed_points(total_number_of_cubes, false);
            for(auto dim_index = 0; dim_index < dim; dim_index ++)
            {
                // Turn vertex_velocities into an array with 1.0 and 0.0 for change in sign
                thrust::transform(vertex_velocities[dim_index].begin(), vertex_velocities[dim_index].end(), velocity_sign_properties[dim_index].begin(), greater_than_zero());

                // Initialize a vector for sign checks
                dev_vec_int summed_positive_signs(total_number_of_cubes, 0); // Contains the sum of positive signs within each cube
                FixedPointCriterion::compute_summed_positive_signs_per_cube(velocity_sign_properties[dim_index], summed_positive_signs);

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
    }
}