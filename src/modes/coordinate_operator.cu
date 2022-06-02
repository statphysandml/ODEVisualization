#include <odesolver/modes/coordinate_operator.hpp>


namespace odesolver {
    namespace modes {
        CoordinateOperator::CoordinateOperator(
            const json params,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const std::string computation_parameters_path,
            const std::vector<std::vector<double>> vecvec_coordinates,
            const odesolver::DevDatC devdat_coordinates
        ) : ODEVisualization(params, flow_equations_ptr, jacobians_ptr, computation_parameters_path),
            dim_(get_entry<json>("flow_equation")["dim"].get<cudaT>())
        {
            if(vecvec_coordinates.size() > 0)
                coordinates_ = odesolver::DevDatC(vecvec_coordinates);
            else if(devdat_coordinates.size() > 0)
                coordinates_ = devdat_coordinates;
        }

        CoordinateOperator CoordinateOperator::generate(
            const odesolver::DevDatC devdat_coordinates,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        )
        {
            return CoordinateOperator(
                json {},
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path,
                {},
                devdat_coordinates
            );
        }

        CoordinateOperator CoordinateOperator::from_vecvec(
                const std::vector<std::vector<double>> vecvec_coordinates,
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
                const std::string computation_parameters_path
        )
        {
            return CoordinateOperator(
                json {},
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path,
                vecvec_coordinates
            );
        }

        CoordinateOperator CoordinateOperator::from_file(
            const std::string rel_config_dir,
            std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
            std::shared_ptr<odesolver::flowequations::JacobianEquationsWrapper> jacobians_ptr,
            const std::string computation_parameters_path
        )
        {
            return CoordinateOperator(
                param_helper::fs::read_parameter_file(
                    param_helper::proj::project_root() + rel_config_dir + "/", "config", false),
                flow_equations_ptr,
                jacobians_ptr,
                computation_parameters_path
            );
        }

        /* template<typename ConditionalObserverParameters>
        CoordinateOperator::EvolveOnConditionParameters<ConditionalObserverParameters>::EvolveOnConditionParameters(const json params_) :
            Parameters(params_),
            observe_every_nth_step(get_entry<cudaT>("observe_every_nth_step")),
            maximum_total_number_of_steps(get_entry<cudaT>("maximum_total_number_of_steps")),
            results_dir(get_entry<std::string>("results_dir")),
            conditional_oberserver_name(get_entry<std::string>("conditional_observer_name"))
        {}

        template<typename ConditionalObserverParameters>
        CoordinateOperator::EvolveOnConditionParameters<ConditionalObserverParameters>::EvolveOnConditionParameters(
            const uint observe_every_nth_step_,
            const uint maximum_total_number_of_steps_,
            const std::string results_dir_) :
            EvolveOnConditionParameters(
                    json {{"observe_every_nth_step", observe_every_nth_step_},
                        {"maximum_total_number_of_steps", maximum_total_number_of_steps_},
                        {"results_dir", results_dir_},
                        {"conditional_observer_name", ConditionalObserverParameters::name()}}
            )
        {} */

        //[ Public functions of CoordinateOperator

        // Velocities
        void CoordinateOperator::compute_velocities()
        {
            velocities_ = compute_flow(coordinates_, flow_equations_ptr_.get());
        }

        void CoordinateOperator::compute_jacobians()
        {
            jacobian_elements_ = compute_jacobian_elements(coordinates_, jacobians_ptr_.get());
            jacobians_ = odesolver::modes::Jacobians(jacobian_elements_.transpose_device_data());
            jacobians_.compute_characteristics();
        }

        void CoordinateOperator::write_characteristics_to_file(const std::string rel_dir) const
        {
            std::vector<std::vector<double>> velocities = velocities_.transpose_device_data();
            std::vector<std::vector<double>> coordinates = coordinates_.transpose_device_data();

            json j;
            j["coordinates"] = odesolver::util::vec_vec_to_json(coordinates);
            j["n_coordinates"] = coordinates.size();
            if(velocities.size() > 0)
            {
                auto velocities_json = odesolver::util::vec_vec_to_json(velocities);
                j["velocities"] = velocities_json;
            }
            if(jacobians_.size() > 0)
            {
                auto jacobians_json = jacobians_.jacobians_to_json();
                j["jacobian"] = jacobians_json;
                auto eigenvectors_json = jacobians_.eigenvectors_to_json();
                j["eigenvectors"] = eigenvectors_json;
                auto eigenvalues_json = jacobians_.eigenvalues_to_json();
                j["eigenvalues"] = eigenvalues_json;
            }
            param_helper::fs::write_parameter_file(j, param_helper::proj::project_root() + "/" + rel_dir + "/", "characteristics", false);
        }

        void CoordinateOperator::set_coordinates(const std::vector<std::vector<double>> coordinates)
        {
            set_coordinates(odesolver::DevDatC(coordinates));
        }

        void CoordinateOperator::set_coordinates(const odesolver::DevDatC coordinates)
        {
            coordinates_ = coordinates;
        }

        odesolver::DevDatC CoordinateOperator::get_coordinates() const
        {
            return coordinates_;
        }

        odesolver::DevDatC CoordinateOperator::get_velocities() const
        {
            return velocities_;
        }

        odesolver::DevDatC CoordinateOperator::get_jacobian_elements() const
        {
            return jacobian_elements_;
        };

        odesolver::modes::Jacobians CoordinateOperator::get_jacobians() const
        {
            return jacobians_;
        }

        //]

        /* template class CoordinateOperator::EvolveOnConditionParameters<ConditionalRangeObserverParameters>::EvolveOnConditionParameters;
        template class CoordinateOperator::EvolveOnConditionParameters<ConditionalIntersectionObserverParameters>::EvolveOnConditionParameters; */
    }
}