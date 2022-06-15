#ifndef PROJECT_SEPARATRIZES_HPP
#define PROJECT_SEPARATRIZES_HPP

#include <sys/file.h>
#include <tuple>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>
#include <odesolver/util/monitor.hpp>
#include <odesolver/util/json_conversions.hpp>
#include <odesolver/util/random.hpp>
#include <odesolver/flow_equations/flow_equation.hpp>
#include <odesolver/flow_equations/jacobian_equation.hpp>
#include <odesolver/modes/jacobians.hpp>
#include <odesolver/modes/evolution.hpp>
#include <odesolver/modes/ode_visualization.hpp>



using json = nlohmann::json;

// ToDo: Replace eval by smaller functions to be able to compute separatrizes for one fixed only, etc. repulsive vs. attractive, etc...


namespace odesolver {
    namespace modes {
        class Separatrizes : public ODEVisualization
        {
        public:
            // From config
            explicit Separatrizes(
                const json params,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
            );

            // From parameters
            static Separatrizes generate(
                const uint N_per_eigen_dim,
                const std::vector<double> shift_per_dim,
                const uint n_max_steps,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
            );

            // From file
            static Separatrizes from_file(
                const std::string rel_config_dir,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr
            );

            // Main function

            template<typename StepperClass, typename Observer>
            void eval(DevDatC &fixed_points, cudaT delta_t, odesolver::modes::Evolution &evolution, StepperClass &stepper, Observer &observer)
            {
                auto jacobian_elements = odesolver::flowequations::compute_jacobian_elements(fixed_points, jacobians_ptr_.get());

                auto jacobians = odesolver::modes::Jacobians::from_devdat(jacobian_elements);

                jacobians.eval();

                const std::vector<int> saddle_point_indices = jacobians.get_indices_with_saddle_point_characteristics();

                // Iterate over all saddle points
                auto potential_saddle_points = fixed_points.to_vec_vec();
                for(auto saddle_point_index = 0; saddle_point_index < saddle_point_indices.size(); saddle_point_index++)
                {
                    std::cout << "Performing for saddle point with x = " << " " << potential_saddle_points[saddle_point_index][0] << std::endl;

                    std::vector<int> stable_manifold_indices; // reverse streamlines <-> inflow
                    std::vector<int> unstable_manifold_indices; // forward streamflow <-> outflow
                    std::vector<std::vector<cudaT>> manifold_eigenvectors;

                    extract_stable_and_unstable_manifolds(jacobians, saddle_point_index, stable_manifold_indices, unstable_manifold_indices, manifold_eigenvectors);

                    compute_separatrizes_of_manifold(potential_saddle_points[saddle_point_index], manifold_eigenvectors, stable_manifold_indices, -1.0 * delta_t, evolution, stepper, observer);

                    compute_separatrizes_of_manifold(potential_saddle_points[saddle_point_index], manifold_eigenvectors, unstable_manifold_indices, delta_t, evolution, stepper, observer);
                }
            }

            void extract_stable_and_unstable_manifolds(odesolver::modes::Jacobians &jacobians, int saddle_point_index, std::vector<int> &stable_manifold_indices, std::vector<int> &unstable_manifold_indices, std::vector<std::vector<cudaT>> &manifold_eigenvectors);

            DevDatC get_initial_values_to_eigenvector(const std::vector<double> &saddle_point, const std::vector<cudaT> &manifold_eigenvector);

            DevDatC sample_around_saddle_point(const std::vector<double> &saddle_point, const std::vector<std::vector<cudaT>> &manifold_eigenvectors, const std::vector<int> &manifold_indices);

            template<typename StepperClass, typename Observer>
            void compute_separatrizes_of_manifold(
                const std::vector<double> &saddle_point,
                const std::vector<std::vector<cudaT>> &manifold_eigenvectors,
                const std::vector<int> &manifold_indices,
                const cudaT delta_t,
                odesolver::modes::Evolution &evolution,
                StepperClass &stepper,
                Observer &observer
            )
            {
                // Perform computation of separatrix for stable manifold
                odesolver::DevDatC sampled_coordinates;
                // Single line
                if(manifold_indices.size() == 1)
                {
                    sampled_coordinates = get_initial_values_to_eigenvector(saddle_point, manifold_eigenvectors[manifold_indices[0]]);
                }
                else
                {
                    sampled_coordinates = sample_around_saddle_point(saddle_point, manifold_eigenvectors, manifold_indices);
                }

                observer.initialize(sampled_coordinates, 0.0);

                std::cout << "Initial point" << std::endl;
                sampled_coordinates.print_dim_by_dim();
                evolution.evolve_n_steps(stepper, sampled_coordinates, 0.0, delta_t, n_max_steps_, observer);
                std::cout << "End point" << std::endl;
                sampled_coordinates.print_dim_by_dim();
            }

        private:
            uint N_per_eigen_dim_;
            std::vector<double> shift_per_dim_;
            uint n_max_steps_;
        };
    }
}

#endif //PROJECT_SEPARATRIZES_HPP
