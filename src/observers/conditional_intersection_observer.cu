#include "../../include/observers/conditional_intersection_observer.hpp"


struct check_potential_intersection_in_dim
{
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        bool current_side = thrust::get<0>(t);
        bool previous_side = thrust::get<1>(t);
        bool status  = thrust::get<2>(t);

        if(current_side == previous_side)
            thrust::get<3>(t) = false;
        else // Could be omitted
            thrust::get<3>(t) = status;
    }
};


struct check_for_vicinity_in_dim
{
    check_for_vicinity_in_dim(const cudaT fixed_lambda_, const cudaT vicinity_distance_) :
            fixed_lambda(fixed_lambda_), vicinity_distance(vicinity_distance_)
    {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        cudaT current_coor = thrust::get<0>(t);
        bool status  = thrust::get<1>(t);

        if(abs(current_coor - fixed_lambda) > vicinity_distance)
            thrust::get<2>(t) = false;
        else // Could be omitted
            thrust::get<2>(t) = status;
    }

    const cudaT fixed_lambda;
    const cudaT vicinity_distance;
};


ConditionalIntersectionObserverParameters::ConditionalIntersectionObserverParameters(const json params_) : ConditionalRangeObserverParameters(params_)
{
    auto vicinity_distances_ = get_value_by_key<json>("vicinity_distances");

    // Serves as an alternative to: vicinity_distances = vicinity_distances_.get< std::vector<cudaT> >();
    std::transform(vicinity_distances_.begin(), vicinity_distances_.end(), std::back_inserter(vicinity_distances), [] (json &dat) { return dat.get< cudaT >(); });
}

ConditionalIntersectionObserverParameters::ConditionalIntersectionObserverParameters(
        const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_,
        const std::vector < cudaT > minimum_change_of_state_,
        const cudaT minimum_delta_t_,
        const cudaT maximum_flow_val_,
        const std::vector < cudaT > vicinity_distances_
) : ConditionalIntersectionObserverParameters(
        json {{"boundary_lambda_ranges", boundary_lambda_ranges_},
              {"minimum_change_of_state", minimum_change_of_state_},
              {"minimum_delta_t", minimum_delta_t_},
              {"maximum_flow_val", maximum_flow_val_},
              {"vicinity_distances", vicinity_distances_}}
)
{}


ConditionalIntersectionObserver::ConditionalIntersectionObserver(FlowEquationsWrapper * flow_equations_, const uint N_total_, std::ofstream &os_,
    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_,
    const std::vector < cudaT > minimum_change_of_state_,
    const cudaT minimum_delta_t_,
    const cudaT maximum_flow_val_,
    const std::vector < cudaT > vicinity_distances_,
    const std::vector < cudaT > fixed_lambdas_,
    const std::vector<int> indices_of_fixed_lambdas_
    ) : ConditionalRangeObserver(flow_equations_, N_total_, os_, boundary_lambda_ranges_, minimum_change_of_state_, minimum_delta_t_, maximum_flow_val_),
    vicinity_distances(vicinity_distances_), fixed_lambdas(fixed_lambdas_), indices_of_fixed_lambdas(indices_of_fixed_lambdas_)
{
    intersection_counter = dev_vec_int(N, 0);
    side_counter = DevDatBool(fixed_lambdas.size(), N, false);
}

ConditionalIntersectionObserver::ConditionalIntersectionObserver(FlowEquationsWrapper * flow_equations_,
        const uint N_total_, std::ofstream &os_, ConditionalIntersectionObserverParameters &params,
        const std::vector<cudaT> fixed_lambdas_, const std::vector<int> indices_of_fixed_lambdas_) :
    ConditionalIntersectionObserver(flow_equations_, N_total_, os_, params.boundary_lambda_ranges, params.minimum_change_of_state,
    params.minimum_delta_t, params.maximum_flow_val, params.vicinity_distances, fixed_lambdas_, indices_of_fixed_lambdas_)
{}


void ConditionalIntersectionObserver::initalize_side_counter(const odesolver::DevDatC &coordinates)
{
    side_counter = compute_side_counter(coordinates, fixed_lambdas, indices_of_fixed_lambdas);
}


void ConditionalIntersectionObserver::operator() (const odesolver::DevDatC &coordinates, cudaT t)
{
    if(monitor)
    {
        std::cout << "t: " << t << std::endl;
        for(auto dim_index = 0; dim_index < dim; dim_index++)
            print_range("Dim " + std::to_string(dim_index), coordinates[dim_index].begin(), coordinates[dim_index].end());
    }

    // Update intersection_counter
    auto updated_side_counter = compute_side_counter(coordinates, fixed_lambdas, indices_of_fixed_lambdas);

    auto intersections = check_for_intersection(updated_side_counter);
    update_for_valid_coordinates(intersections);
    write_intersecting_separatrizes_to_file(coordinates, intersections);
    auto vicinity_to_fixed_dimensions = check_for_vicinity_to_fixed_dimensions(coordinates);
    update_for_valid_coordinates(vicinity_to_fixed_dimensions);
    write_vicinity_separatrizes_to_file(coordinates, vicinity_to_fixed_dimensions);
    update_out_of_system(coordinates);

    previous_coordinates = coordinates;
    side_counter = updated_side_counter;
}

void ConditionalIntersectionObserver::write_intersecting_separatrizes_to_file(const odesolver::DevDatC &coordinates, const dev_vec_bool& intersections)
{
    // Function to average over previous and current coordinates and write into separatrizes file
    auto n_intersections = thrust::count(intersections.begin(), intersections.end(), true);
    if(n_intersections > 0)
    {
        auto c = 0;
        odesolver::DevDatC separatrizes(dim - fixed_lambdas.size(), coordinates.size() / coordinates.dim_size(), 0);
        std::vector< dev_iterator > end_iterators {};
        for (auto dim_index = 0; dim_index < dim; dim_index++) {
            // Fixed indices are fix -> only other coordinates are needed
            if (std::find(indices_of_fixed_lambdas.begin(), indices_of_fixed_lambdas.end(), dim_index) ==
                indices_of_fixed_lambdas.end()) {

                // Write coordinates into separatrizes
                auto end_iterator = thrust::copy_if(coordinates[dim_index].begin(), coordinates[dim_index].end(),
                                                    intersections.begin(),
                                                    separatrizes[c].begin(), []
                __host__ __device__(const bool &status) { return status; });

                // Extract intersecting previous coordinates
                dev_vec intersecting_previous_coordinates(end_iterator - separatrizes[c].begin(), 0);
                thrust::copy_if(previous_coordinates[dim_index].begin(), previous_coordinates[dim_index].end(),
                                intersections.begin(),
                                intersecting_previous_coordinates.begin(), []
                __host__ __device__(const bool &status) { return status; });

                thrust::transform(separatrizes[c].begin(), separatrizes[c].end(), intersecting_previous_coordinates.begin(),
                                  separatrizes[c].begin(), [] __host__ __device__ (const cudaT& val1, const cudaT& val2) { return (val1 + val2)/2; });
                end_iterators.push_back(end_iterator);
                c++;
            }
        }
        write_data_to_ofstream(separatrizes, os, std::vector<int> {}, end_iterators);
    }
}

void ConditionalIntersectionObserver::write_vicinity_separatrizes_to_file(const odesolver::DevDatC &coordinates, const dev_vec_bool& vicinity_to_fixed_dimensions)
{
    // Function to project on fixed dimensions
    auto n_vicinity = thrust::count(vicinity_to_fixed_dimensions.begin(), vicinity_to_fixed_dimensions.end(), true);
    if(n_vicinity > 0)
    {
        auto c = 0;
        odesolver::DevDatC separatrizes(dim - fixed_lambdas.size(), coordinates.size() / coordinates.dim_size(), 0);
        std::vector< dev_iterator > end_iterators {};
        for (auto dim_index = 0; dim_index < dim; dim_index++) {
            // Fixed indices are fix -> only other coordinates are needed
            if (std::find(indices_of_fixed_lambdas.begin(), indices_of_fixed_lambdas.end(), dim_index) ==
                indices_of_fixed_lambdas.end()) {
                // dev_vec vicinity_c(vicinity_to_fixed_dimensions.begin(), vicinity_to_fixed_dimensions.end());
                auto end_iterator = thrust::copy_if(coordinates[dim_index].begin(), coordinates[dim_index].end(),
                                                    vicinity_to_fixed_dimensions.begin(),
                                                    separatrizes[c].begin(), []
                __host__ __device__(const bool &status) { return status; });
                end_iterators.push_back(end_iterator);

                c++;
            }
        }
        write_data_to_ofstream(separatrizes, os, std::vector<int> {}, end_iterators);
    }
}

dev_vec_bool ConditionalIntersectionObserver::check_for_intersection(const DevDatBool& updated_side_counter)
{
    // Initialize true and set to false if in some dimension no side change took place
    dev_vec_bool potential_intersections(N, true);
    for(auto fixed_dim_index = 0; fixed_dim_index < fixed_lambdas.size(); fixed_dim_index++)
    {
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                updated_side_counter[fixed_dim_index].begin(),
                side_counter[fixed_dim_index].begin(),
                potential_intersections.begin(),
                potential_intersections.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(
                                 updated_side_counter[fixed_dim_index].end(),
                                 side_counter[fixed_dim_index].end(),
                                 potential_intersections.end(),
                                 potential_intersections.end())),
                         check_potential_intersection_in_dim()
        );
    }
    if(monitor)
        print_range("Intersection check", potential_intersections.begin(), potential_intersections.end());

    // ToDo: May not work on gpu!!
    thrust::transform(intersection_counter.begin(), intersection_counter.end(), potential_intersections.begin(),
                      intersection_counter.begin(), [] __host__ __device__ (const int& counter, const bool& potential_intersection) {
        return counter + int(potential_intersection);
    });
    if(monitor)
        print_range("Intersection counter", intersection_counter.begin(), intersection_counter.end());
    return potential_intersections;
}

dev_vec_bool ConditionalIntersectionObserver::check_for_vicinity_to_fixed_dimensions(const odesolver::DevDatC &coordinates)
{
    // Initialize true and set to false if in some dimension the coordinate is not in the vicinity
    dev_vec_bool potential_vicinities(N, true);
    for(auto fixed_dim_index = 0; fixed_dim_index < fixed_lambdas.size(); fixed_dim_index++)
    {
        auto index_of_fixed_lambdas = indices_of_fixed_lambdas[fixed_dim_index];
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                coordinates[index_of_fixed_lambdas].begin(),
                potential_vicinities.begin(),
                potential_vicinities.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(
                                 coordinates[index_of_fixed_lambdas].end(),
                                 potential_vicinities.end(),
                                 potential_vicinities.end())),
                         check_for_vicinity_in_dim(fixed_lambdas[fixed_dim_index], vicinity_distances[fixed_dim_index])
        );
    }
    if(monitor)
        print_range("Vicinity check", potential_vicinities.begin(), potential_vicinities.end());
    return potential_vicinities;
}