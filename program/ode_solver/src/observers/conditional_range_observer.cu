#include "../../include/observers/conditional_range_observer.hpp"

struct compare_to_previous_change
{
    compare_to_previous_change(const cudaT minimum_change_of_state_in_dim_) :
            minimum_change_of_state_in_dim(minimum_change_of_state_in_dim_)
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        cudaT current_coor = thrust::get<0>(t);
        cudaT previous_coor = thrust::get<1>(t);
        bool status  = thrust::get<2>(t);

        if(abs(current_coor - previous_coor) < minimum_change_of_state_in_dim and current_coor != previous_coor)
            thrust::get<3>(t) = true;
        else // Could be omitted
            thrust::get<3>(t) = status;
    }

    const cudaT minimum_change_of_state_in_dim;
};


ConditionalRangeObserverParameters::ConditionalRangeObserverParameters(const json params_) : Parameters(params_),
                                                         minimum_delta_t(get_value_by_key<cudaT>("minimum_delta_t")),
                                                         maximum_flow_val(get_value_by_key<cudaT>("maximum_flow_val"))
{
    auto boundary_lambda_ranges_ = get_value_by_key<json>("boundary_lambda_ranges");
    auto minimum_change_of_state_ = get_value_by_key<json>("minimum_change_of_state");

    std::transform(boundary_lambda_ranges_.begin(), boundary_lambda_ranges_.end(), std::back_inserter(boundary_lambda_ranges),
                   [] (json &dat) { return dat.get< std::pair<cudaT, cudaT> >(); });
    minimum_change_of_state = minimum_change_of_state_.get< std::vector<cudaT> >();
}

ConditionalRangeObserverParameters::ConditionalRangeObserverParameters(
        const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_,
        const std::vector < cudaT > minimum_change_of_state_,
        const cudaT minimum_delta_t_,
        const cudaT maximum_flow_val_
) : ConditionalRangeObserverParameters(
        json {{"boundary_lambda_ranges", boundary_lambda_ranges_},
              {"minimum_change_of_state", minimum_change_of_state_},
              {"minimum_delta_t", minimum_delta_t_},
              {"maximum_flow_val", maximum_flow_val_}}
)
{}



ConditionalRangeObserver::ConditionalRangeObserver(FlowEquationsWrapper * const flow_equations_, const uint N_total_,
        std::ofstream &os_,
        const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_,
        const std::vector < cudaT > minimum_change_of_state_,
        const cudaT minimum_delta_t_,
        const cudaT maximum_flow_val_
    ) : flow_equations(flow_equations_), dim(flow_equations_->get_dim()), N_total(N_total_), N(N_total_/flow_equations_->get_dim()), os(os_),
    minimum_change_of_state(minimum_change_of_state_), minimum_delta_t(minimum_delta_t_), maximum_flow_val(maximum_flow_val_)
{
    upper_lambda_ranges.reserve(boundary_lambda_ranges_.size());
    lower_lambda_ranges.reserve(boundary_lambda_ranges_.size());
    for (auto it = std::make_move_iterator(boundary_lambda_ranges_.begin()),
                 end = std::make_move_iterator(boundary_lambda_ranges_.end()); it != end; ++it)
    {
        upper_lambda_ranges.push_back(std::move(it->second));
        lower_lambda_ranges.push_back(std::move(it->first));
    }
    indices_of_boundary_lambdas = std::vector<int>(dim);
    std::iota(indices_of_boundary_lambdas.begin(), indices_of_boundary_lambdas.end(), 0);

    previous_coordinates = DevDatC(dim, N, 100001.02342);
    out_of_system = dev_vec_bool(N, false);
}

ConditionalRangeObserver::ConditionalRangeObserver(FlowEquationsWrapper * const flow_equations_, const uint N_total_,
        std::ofstream &os_, const ConditionalRangeObserverParameters &params) :
        ConditionalRangeObserver(flow_equations_, N_total_, os_, params.boundary_lambda_ranges, params.minimum_change_of_state,
        params.minimum_delta_t, params.maximum_flow_val)
{}


void ConditionalRangeObserver::operator() (const DevDatC &coordinates, cudaT t)
{
    if(monitor)
    {
        std::cout << "t: " << t << std::endl;
        for(auto dim_index = 0; dim_index < dim; dim_index++)
            print_range("Dim " + std::to_string(dim_index), coordinates[dim_index].begin(), coordinates[dim_index].end());
    }

    auto n_out_of_system = thrust::count(out_of_system.begin(), out_of_system.end(), true);
    if(n_out_of_system == 0)
        write_data_to_ofstream(coordinates, os);
    else
    {
        dev_vec_bool potential_in_system_coordinates(N, true);
        // Write only the in system coordinates to file
        update_for_valid_coordinates(potential_in_system_coordinates);
        DevDatC in_system_coordinates(dim, coordinates.size() / coordinates.dim_size(), 0);
        std::vector< dev_iterator > end_iterators {};
        for (auto dim_index = 0; dim_index < dim; dim_index++) {
            auto end_iterator = thrust::copy_if(coordinates[dim_index].begin(), coordinates[dim_index].end(),
                                                potential_in_system_coordinates.begin(),
                                                in_system_coordinates[dim_index].begin(), []
                                                __host__ __device__(const bool &status) { return status; });
            end_iterators.push_back(end_iterator);

        }
        write_data_to_ofstream(in_system_coordinates, os, std::vector<int> {}, end_iterators);
    }

    update_out_of_system(coordinates);
    previous_coordinates = coordinates;
}


void ConditionalRangeObserver::update_out_of_system(const DevDatC &coordinates)
{
    // Check if trajectories are still inside the provided lambda ranges - makes use of boundary_lambda_ranges
    auto upper_out_of_range = compute_side_counter(coordinates, upper_lambda_ranges, indices_of_boundary_lambdas);
    auto lower_out_of_range = compute_side_counter(coordinates, lower_lambda_ranges, indices_of_boundary_lambdas);
    for(auto dim_index = 0; dim_index < dim; dim_index++)
    {
        thrust::transform_if(upper_out_of_range[dim_index].begin(), upper_out_of_range[dim_index].end(), out_of_system.begin(),
                             [] __host__ __device__ (const bool&status) { return true; }, [] __host__ __device__ (const bool& status) { return status; });
        thrust::transform_if(lower_out_of_range[dim_index].begin(), lower_out_of_range[dim_index].end(), out_of_system.begin(),
                             [] __host__ __device__ (const bool&status) { return true; }, [] __host__ __device__ (const bool& status) { return !status; });
    }

    // Check if change in variable is large enough - makes use of minimum_change_in_state
    for(auto dim_index = 0; dim_index < dim; dim_index++) {
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                coordinates[dim_index].begin(),
                previous_coordinates[dim_index].begin(),
                out_of_system.begin(),
                out_of_system.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(
                                 coordinates[dim_index].end(),
                                 previous_coordinates[dim_index].end(),
                                 out_of_system.end(),
                                 out_of_system.end())),
                         compare_to_previous_change(minimum_change_of_state[dim_index])
        );
    }

    // Check if evaluated flow is too large - makes use of maximum_flow_val
    auto velocities = compute_vertex_velocities(coordinates, flow_equations);
    auto maximum_flow_val_ = maximum_flow_val;
    for(auto dim_index = 0; dim_index < dim; dim_index++) {
        thrust::transform_if(velocities[dim_index].begin(), velocities[dim_index].end(), out_of_system.begin(),
                             [] __host__ __device__(const cudaT &val) { return true; },
        [maximum_flow_val_] __host__ __device__(const cudaT &val) { return val > maximum_flow_val_; });
    }

    if(monitor)
    {
        print_range("Out of system", out_of_system.begin(), out_of_system.end());
        std::cout << "Check for out of system results in " << std::to_string(check_for_out_of_system()) << std::endl;
        std::cout << std::endl;
    }
}

bool ConditionalRangeObserver::check_for_out_of_system() const
{
    // Returns true if all coordinates are out of system
    auto n_out_of_system = thrust::count(out_of_system.begin(), out_of_system.end(), true);
    return out_of_system.size() == n_out_of_system;
}

void ConditionalRangeObserver::update_for_valid_coordinates(dev_vec_bool& potential_valid_coordinates) {
    // Set elem of potential valid coordinates to false if it is out of system
    thrust::transform_if(out_of_system.begin(), out_of_system.end(), potential_valid_coordinates.begin(),
                         [] __host__ __device__ (const bool&status) { return false; }, [] __host__ __device__ (const bool& status) { return status; });
}

DevDatBool ConditionalRangeObserver::compute_side_counter(const DevDatC &coordinates, const std::vector <cudaT>& lambdas, const std::vector<int>& indices_of_lambdas)
{
    DevDatBool side_counter_(lambdas.size(), coordinates.size()/coordinates.dim_size(), false);
    for(auto dim_index = 0; dim_index < lambdas.size(); dim_index++)
    {
        auto index_of_lambdas = indices_of_lambdas[dim_index];
        auto lambda_in_dim = lambdas[dim_index];
        thrust::transform(coordinates[index_of_lambdas].begin(), coordinates[index_of_lambdas].end(), side_counter_[dim_index].begin(), [lambda_in_dim] __host__ __device__ (const cudaT &val) {
            return val > lambda_in_dim;
        });
    }
    return side_counter_;
}