#include <odesolver/evolution/conditional_range_observer.hpp>

namespace odesolver {
    namespace evolution {

        struct compare_to_previous_change
        {
            compare_to_previous_change(const cudaT minimum_change_of_state_in_dim) :
                minimum_change_of_state_in_dim_(minimum_change_of_state_in_dim)
            {}

            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                cudaT current_coor = thrust::get<0>(t);
                cudaT previous_coor = thrust::get<1>(t);
                bool status  = thrust::get<2>(t);

                if(abs(current_coor - previous_coor) < minimum_change_of_state_in_dim_ and current_coor != previous_coor)
                    thrust::get<3>(t) = true;
                else // Could be omitted
                    thrust::get<3>(t) = status;
            }

            const cudaT minimum_change_of_state_in_dim_;
        };


        ConditionalRangeObserver::ConditionalRangeObserver(
            const json params, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr
        ) : Parameters(params),
            flow_equations_ptr_(flow_equations_ptr),
            dim_(flow_equations_ptr_->get_dim()),
            N_(get_entry<size_t>("N")),
            minimum_delta_t_(get_entry<cudaT>("minimum_delta_t")),
            maximum_flow_val_(get_entry<cudaT>("maximum_flow_val")),
            minimum_change_of_state_(get_entry<std::vector<cudaT>>("minimum_change_of_state"))
        {
            auto variable_ranges = odesolver::util::json_to_vec_pair(get_entry<json>("variable_ranges");
            upper_variable_ranges_.reserve(variable_ranges.size());
            lower_variable_ranges_.reserve(variable_ranges.size());
            for (auto it = std::make_move_iterator(variable_ranges.begin()),
                        end = std::make_move_iterator(variable_ranges.end()); it != end; ++it)
            {
                upper_variable_ranges_.push_back(std::move(it->second));
                lower_variable_ranges_.push_back(std::move(it->first));
            }
            indices_of_boundary_variables_ = std::vector<int>(dim_);
            std::iota(indices_of_boundary_variables_.begin(), indices_of_boundary_variables_.end(), 0);

            previous_coordinates_ = odesolver::DevDatC(dim_, N_, 0.0);
            out_of_system_ = dev_vec_bool(N_, false);
        }

        ConditionalRangeObserver ConditionalRangeObserver::generate(
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                const size_t N,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
                const std::vector<cudaT> minimum_change_of_state,
                const cudaT minimum_delta_t,
                const cudaT maximum_flow_val
        )
        {
            return ConditionalRangeObserver(
                json {
                      {"dim", flow_equations_ptr->get_dim()},
                      {"N", N},
                      {"variable_ranges", variable_ranges},
                      {"minimum_change_of_state", minimum_change_of_state},
                      {"minimum_delta_t", minimum_delta_t},
                      {"maximum_flow_val", maximum_flow_val}},
                flow_equations_ptr
            );
        }

        void ConditionalRangeObserver::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            if(monitor)
            {
                std::cout << "t: " << t << std::endl;
                for(auto dim_index = 0; dim_index < dim_; dim_index++)
                    print_range("Dim " + std::to_string(dim_index), coordinates[dim_index].begin(), coordinates[dim_index].end());
            }

            update_out_of_system(coordinates);
            previous_coordinates_ = coordinates;
        }


        void ConditionalRangeObserver::update_out_of_system(const odesolver::DevDatC &coordinates)
        {
            // Check if trajectories are still inside the provided variable ranges - makes use of variable_ranges
            auto upper_out_of_range = compute_side_counter(coordinates, upper_variable_ranges_, indices_of_boundary_variables_);
            auto lower_out_of_range = compute_side_counter(coordinates, lower_variable_ranges_, indices_of_boundary_variables_);
            for(auto dim_index = 0; dim_index < dim_; dim_index++)
            {
                thrust::transform_if(upper_out_of_range[dim_index].begin(), upper_out_of_range[dim_index].end(), out_of_system_.begin(),
                                    [] __host__ __device__ (const bool&status) { return true; }, [] __host__ __device__ (const bool& status) { return status; });
                thrust::transform_if(lower_out_of_range[dim_index].begin(), lower_out_of_range[dim_index].end(), out_of_system_.begin(),
                                    [] __host__ __device__ (const bool&status) { return true; }, [] __host__ __device__ (const bool& status) { return !status; });
            }

            // Check if change in variable is large enough - makes use of minimum_change_in_state
            for(auto dim_index = 0; dim_index < dim_; dim_index++) {
                thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                        coordinates[dim_index].begin(),
                        previous_coordinates_[dim_index].begin(),
                        out_of_system_.begin(),
                        out_of_system_.begin())),
                                thrust::make_zip_iterator(thrust::make_tuple(
                                        coordinates[dim_index].end(),
                                        previous_coordinates_[dim_index].end(),
                                        out_of_system_.end(),
                                        out_of_system_.end())),
                                compare_to_previous_change(minimum_change_of_state_[dim_index])
                );
            }

            // Check if evaluated flow is too large - makes use of maximum_flow_val
            auto flow = compute_flow(coordinates, flow_equations_ptr_.get());
            auto maximum_flow_val = maximum_flow_val_;
            for(auto dim_index = 0; dim_index < dim_; dim_index++) {
                thrust::transform_if(flow[dim_index].begin(), flow[dim_index].end(), out_of_system_.begin(),
                                    [] __host__ __device__(const cudaT &val) { return true; },
                [maximum_flow_val] __host__ __device__(const cudaT &val) { return val > maximum_flow_val; });
            }

            if(monitor)
            {
                print_range("Out of system", out_of_system_.begin(), out_of_system_.end());
                std::cout << "Check for out of system results in " << std::to_string(check_for_out_of_system()) << std::endl;
                std::cout << std::endl;
            }
        }
        
        int n_out_of_system()
        {
            return thrust::count(out_of_system_.begin(), out_of_system_.end(), true);
        }

        dev_vec_bool coordinate_indices_mask()
        {
            return out_of_system;
        }

        bool ConditionalRangeObserver::check_for_out_of_system() const
        {
            // Returns true if all coordinates are out of system
            auto n_out_of_system = thrust::count(out_of_system_.begin(), out_of_system_.end(), true);
            return out_of_system_.size() == n_out_of_system;
        }
        
        void ConditionalRangeObserver::valid_coordinate_incides() {
            dev_vec_bool potential_valid_coordinates(out_of_system.size(), true);
            thrust::transform_if(out_of_system_.begin(), out_of_system_.end(), potential_valid_coordinates.begin(),
                                [] __host__ __device__ (const bool&status) { return false; }, [] __host__ __device__ (const bool& status) { return status; });
        }

        DevDatBool ConditionalRangeObserver::compute_side_counter(const odesolver::DevDatC &coordinates, const std::vector<cudaT>& variables, const std::vector<int>& indices_of_variables)
        {
            DevDatBool side_counter_(variables.size(), coordinates.size()/coordinates.dim_size(), false);
            for(auto dim_index = 0; dim_index < variables.size(); dim_index++)
            {
                auto index_of_variables = indices_of_variables[dim_index];
                auto variable_in_dim = variables[dim_index];
                thrust::transform(coordinates[index_of_variables].begin(), coordinates[index_of_variables].end(), side_counter_[dim_index].begin(), [variable_in_dim] __host__ __device__ (const cudaT &val) {
                    return val > variable_in_dim;
                });
            }
            return side_counter_;
        }

    }
}