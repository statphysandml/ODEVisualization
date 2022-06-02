//
// Created by kades on 8/12/19.
//

#ifndef PROJECT_CONDITIONAL_RANGE_OBSERVER_HPP
#define PROJECT_CONDITIONAL_RANGE_OBSERVER_HPP

#include <param_helper/params.hpp>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/json_conversions.hpp>
#include <odesolver/evolution/evolution_observer.hpp>
#include <odesolver/flow_equations/flow_equation.hpp>


namespace odesolver {
    namespace evolution {

        struct ConditionalRangeObserver : public EvolutionObserver, param_helper::params::Parameters
        {
        public:
            explicit ConditionalRangeObserver(const json params, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr);

            // From parameters
            static ConditionalRangeObserver generate(
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                const size_t N,
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges = std::vector<std::pair<cudaT, cudaT>> {},
                const std::vector<cudaT> minimum_change_of_state = std::vector<cudaT> {},
                const cudaT minimum_delta_t = 0.0000001,
                const cudaT maximum_flow_val = 1e10
            );

            // From file
            static ConditionalRangeObserver from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr
            );

            void operator() (const odesolver::DevDatC &coordinates, cudaT t) override;

            void update_out_of_system(const odesolver::DevDatC &coordinates);

            bool check_for_out_of_system() const;

            int n_out_of_system();

            dev_vec_bool coordinate_indices_mask();
            
            dev_vec_bool valid_coordinate_incides();

            static odesolver::DevDatBool compute_side_counter(const odesolver::DevDatC &coordinates, const std::vector <cudaT>& variables, const std::vector<int>& indices_of_variables);

            static std::string name() {  return "conditional_range_observer";  }

            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr_;
            const size_t dim_;
            const size_t N_;
            
            std::vector<cudaT> upper_variable_ranges_;
            std::vector<cudaT> lower_variable_ranges_;
            std::vector<int> indices_of_boundary_variables_;
            const std::vector<cudaT> minimum_change_of_state_;
            const cudaT minimum_delta_t_;
            const cudaT maximum_flow_val_;

            odesolver::DevDatC previous_coordinates_;
            dev_vec_bool out_of_system_;
        };
    }
}


#endif //PROJECT_CONDITIONAL_RANGE_OBSERVER_HPP
