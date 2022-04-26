//
// Created by kades on 8/12/19.
//

#ifndef PROJECT_CONDITIONAL_INTERSECTION_OBSERVER_HPP
#define PROJECT_CONDITIONAL_INTERSECTION_OBSERVER_HPP

#include "conditional_range_observer.hpp"

struct ConditionalIntersectionObserver;

struct ConditionalIntersectionObserverParameters : public ConditionalRangeObserverParameters
{
    ConditionalIntersectionObserverParameters() : ConditionalRangeObserverParameters()
    {
        std::cout << "Called default constructor of ConditionalIntersectionObserverParameters" << std::endl;
    }

    ConditionalIntersectionObserverParameters(const json params_);

    ConditionalIntersectionObserverParameters(
            const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_,
            const std::vector < cudaT > minimum_change_of_state_ = std::vector<cudaT> {},
            const cudaT minimum_delta_t_ = 0.0000001,
            const cudaT maximum_flow_val_ = 1e10,
            const std::vector < cudaT > vicinity_distances_ = std::vector<cudaT> {}
    );

    static std::string name() {  return "conditional_intersection_observer";  }

    std::vector <cudaT> vicinity_distances;

    typedef ConditionalIntersectionObserver ConditionalObserver;
};


struct ConditionalIntersectionObserver : public ConditionalRangeObserver
{
public:
    explicit ConditionalIntersectionObserver(FlowEquationsWrapper * flow_equations_, const uint N_total_, std::ofstream &os_,
            const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_ = std::vector <std::pair<cudaT, cudaT> > {},
            const std::vector < cudaT > minimum_change_of_state_ = std::vector<cudaT> {},
            const cudaT minimum_delta_t_ = 0.0000001,
            const cudaT maximum_flow_val_ = 1e10,
            const std::vector < cudaT > vicinity_distances_ = std::vector<cudaT> {},
            const std::vector < cudaT > fixed_lambdas_ = std::vector<cudaT> {},
            const std::vector<int> indices_of_fixed_lambdas_ = std::vector<int> {}
    );

    ConditionalIntersectionObserver(FlowEquationsWrapper * flow_equations_, const uint N_total_, std::ofstream &os_, ConditionalIntersectionObserverParameters &params,
                        const std::vector<cudaT> fixed_lambdas_ = std::vector<cudaT> {}, const std::vector<int> indices_of_fixed_lambdas_ = std::vector<int> {});

    void initalize_side_counter(const odesolver::DevDatC &coordinates);

    void operator() (const odesolver::DevDatC &coordinates, cudaT t) override;

    void write_intersecting_separatrizes_to_file(const odesolver::DevDatC &coordinates, const dev_vec_bool& intersections);

    void write_vicinity_separatrizes_to_file(const odesolver::DevDatC &coordinates, const dev_vec_bool& vicinity_to_fixed_dimensions);

    dev_vec_bool check_for_intersection(const DevDatBool& updated_side_counter);

    dev_vec_bool check_for_vicinity_to_fixed_dimensions(const odesolver::DevDatC &coordinates);

    static std::string name() {  return "conditional_intersection_observer";  }

    const std::vector <cudaT> fixed_lambdas;
    const std::vector<int> indices_of_fixed_lambdas;
    const std::vector <cudaT> vicinity_distances;

    dev_vec_int intersection_counter;
    DevDatBool side_counter;
};

#endif //PROJECT_CONDITIONAL_INTERSECTION_OBSERVER_HPP
