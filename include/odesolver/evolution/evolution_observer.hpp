#ifndef PROJECT_EVOLUTION_OBSERVER_HPP
#define PROJECT_EVOLUTION_OBSERVER_HPP

#include <exception>

#include <param_helper/params.hpp>
#include <param_helper/filesystem.hpp>

#include <devdat/header.hpp>
#include <devdat/devdat.hpp>
#include <devdat/util/json_conversions.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/evolution/evolution_observer.hpp>
#include <flowequations/flow_equation.hpp>


namespace odesolver {
    namespace evolution {

        struct StopEvolutionException : public std::exception
        {
            StopEvolutionException(cudaT t, std::string reason) : t_(t), reason_(reason)
            {}

            cudaT get_end_time() const
            {
                return t_;
            }

            std::string get_reason() const
            {
                return reason_;
            }

            cudaT t_;
            std::string reason_;
        };

        struct FlowObserver : param_helper::params::Parameters
        {
            FlowObserver(const json params={});

            virtual void operator() (const devdat::DevDatC &coordinates, cudaT t);

            virtual void initialize(const devdat::DevDatC &coordinates, cudaT t);

            virtual dev_vec_bool& valid_coordinates_mask();

            virtual const dev_vec_bool valid_coordinates_mask() const;
        
            virtual std::string name();

            virtual int n_valid_coordinates() const;

            virtual bool valid_coordinates() const;

            virtual void update_valid_coordinates(dev_vec_bool &conditions);

            std::shared_ptr<dev_vec_bool> valid_coordinates_mask_ptr_;
        };

        struct DivergentFlow : FlowObserver
        {
            explicit DivergentFlow(const json params, std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr);

            // From parameters
            static DivergentFlow generate(
                std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr,
                const cudaT maximum_abs_flow_val = 1e10
            );

            void operator() (const devdat::DevDatC &coordinates, cudaT t) override;

            std::string name() override { return "divergent_flow"; }

            const cudaT maximum_abs_flow_val_;
            std::shared_ptr<flowequations::FlowEquationsWrapper> flow_equations_ptr_;
        };

        struct NoChange : FlowObserver
        {
            explicit NoChange(
                const json params
            );

            // From parameters
            static NoChange generate(
                const std::vector<cudaT> minimum_change_of_state = std::vector<cudaT> {}
            );

            void operator() (const devdat::DevDatC &coordinates, cudaT t) override;

            void initialize(const devdat::DevDatC &coordinates, cudaT t) override;

            std::string name() override { return "no_change"; }

            const std::vector<cudaT> minimum_change_of_state_;

            dev_vec_bool detected_changes_;
            devdat::DevDatC previous_coordinates_;
        };

        struct OutOfRangeCondition : FlowObserver
        {
            explicit OutOfRangeCondition(const json params);

            // From parameters
            static OutOfRangeCondition generate(
                const std::vector<std::pair<cudaT, cudaT>> variable_ranges = std::vector<std::pair<cudaT, cudaT>>{},
                const std::vector<int> observed_dimension_indices = std::vector<int>{}
            );

            void operator() (const devdat::DevDatC &coordinates, cudaT t) override;

            std::string name() override { return "out_of_range_condition"; }

            dev_vec_bool out_of_range_;
            std::vector<std::pair<cudaT, cudaT>> variable_ranges_;
            std::vector<int> observed_dimension_indices_;
        };

        struct Intersection : FlowObserver
        {
            explicit Intersection(
                const json params
            );

            // From parameters
            static Intersection generate(
                const std::vector<cudaT> vicinity_distances,
                const std::vector<cudaT> fixed_variables,
                const std::vector<int> fixed_variable_indices,
                const bool remember_intersections
            );

            void operator() (const devdat::DevDatC &coordinates, cudaT t) override;

            void initialize(const devdat::DevDatC &coordinates, cudaT t) override;

            std::string name() override { return "intersection"; }

            bool valid_coordinates() const override;

            dev_vec_int intersection_counter() const;

            std::vector<std::vector<cudaT>> detected_intersections() const;

            std::vector<bool> detected_intersection_types() const;


            devdat::DevDatC previous_coordinates_;

            std::vector<cudaT> vicinity_distances_;
            std::vector<cudaT> fixed_variables_;
            std::vector<int> fixed_variable_indices_;
            bool remember_intersections_;
            
            dev_vec_bool intersections_;
            dev_vec_bool vicinities_;
            dev_vec_bool intersections_and_vicinities_;

            std::shared_ptr<dev_vec_int> intersection_counter_ptr_; // Storing the number of actual intersection for each coordinate
            dev_vec flattened_intersection_coordinates_; // For temporary storing of intersection coordinates
            dev_vec_bool intersection_type_; // For temporary storing of intersection type; true = actual intersection, false = no actual intersection, but vicinity

            std::shared_ptr<std::vector<std::vector<cudaT>>> detected_intersections_ptr_; // For a permanent storing of intersections
            std::shared_ptr<std::vector<bool>> detected_intersections_types_ptr_; // For a permanent storing of the intersection types
        };

        struct TrajectoryObserver : FlowObserver
        {
            explicit TrajectoryObserver(const json params);

            static TrajectoryObserver generate(std::string file);

            void operator() (const devdat::DevDatC &coordinates, cudaT t) override;

            std::string name() override { return "trajectory_observer"; }

            bool valid_coordinates() const override;

            std::ofstream os_;
        };

        struct EvolutionObserver : FlowObserver
        {
            EvolutionObserver(
                const json params,
                const std::vector<std::shared_ptr<FlowObserver>> observers
            );

            static EvolutionObserver generate(const std::vector<std::shared_ptr<FlowObserver>> observers);
            
            void operator() (const devdat::DevDatC &coordinates, cudaT t) override;

            void initialize(const devdat::DevDatC &coordinates, cudaT t) override;

            std::string name() override { return "evolution_observer"; }

            std::vector<std::shared_ptr<FlowObserver>> observers_;
        };
    }
}

#endif //PROJECT_EVOLUTION_OBSERVER_HPP
