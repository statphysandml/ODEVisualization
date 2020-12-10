//
// Created by kades on 8/12/19.
//

#ifndef PROJECT_CONDITIONAL_RANGE_OBSERVER_HPP
#define PROJECT_CONDITIONAL_RANGE_OBSERVER_HPP

#include "evolution_observer.hpp"

struct ConditionalRangeObserver;

struct ConditionalRangeObserverParameters : public Parameters
{
    ConditionalRangeObserverParameters() : minimum_delta_t(0.0), maximum_flow_val(0.0)
    {
        std::cout << "Called default constructor of ConditionalRangeObserverParameters" << std::endl;
    }

    ConditionalRangeObserverParameters(const json params_);

    // From file contructor not implemented since these observer class are only used within main modules so far

    ConditionalRangeObserverParameters(
            const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_,
            const std::vector < cudaT > minimum_change_of_state_ = std::vector<cudaT> {},
            const cudaT minimum_delta_t_ = 0.0000001,
            const cudaT maximum_flow_val_ = 1e10
    );

    static std::string name() {  return "conditional_range_observer";  }

    // ToDo: Remove redundancy of parameters
    const cudaT minimum_delta_t;
    const cudaT maximum_flow_val;

    std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges;
    std::vector <cudaT> minimum_change_of_state;

    typedef ConditionalRangeObserver ConditionalObserver;
};


struct ConditionalRangeObserver : public EvolutionObserver
{
public:
    explicit ConditionalRangeObserver(FlowEquationsWrapper * const flow_equations_, const uint N_total_, std::ofstream &os_,
                                             const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges_ = std::vector <std::pair<cudaT, cudaT> > {},
                                             const std::vector < cudaT > minimum_change_of_state_ = std::vector<cudaT> {},
                                             const cudaT minimum_delta_t_ = 0.0000001,
                                             const cudaT maximum_flow_val_ = 1e10
    );

    ConditionalRangeObserver(FlowEquationsWrapper * const flow_equations_, const uint N_total_, std::ofstream &os_, const ConditionalRangeObserverParameters &params);

    void operator() (const DevDatC &coordinates, cudaT t) override;

    void update_out_of_system(const DevDatC &coordinates);

    bool check_for_out_of_system() const;

    void update_for_valid_coordinates(dev_vec_bool& potential_valid_coordinates);

    static DevDatBool compute_side_counter(const DevDatC &coordinates, const std::vector <cudaT>& lambdas, const std::vector<int>& indices_of_lambdas);

    static std::string name() {  return "conditional_range_observer";  }

    FlowEquationsWrapper * const flow_equations;
    const uint8_t dim;
    const uint N_total; // Resulting size of the vector
    const uint N; // Considered number of coordinates
    std::ofstream& os;

    std::vector <cudaT> upper_lambda_ranges;
    std::vector <cudaT> lower_lambda_ranges;
    std::vector<int> indices_of_boundary_lambdas;
    const std::vector <cudaT> minimum_change_of_state;
    const cudaT minimum_delta_t;
    const cudaT maximum_flow_val;

    DevDatC previous_coordinates;
    dev_vec_bool out_of_system;
};


#endif //PROJECT_CONDITIONAL_RANGE_OBSERVER_HPP
