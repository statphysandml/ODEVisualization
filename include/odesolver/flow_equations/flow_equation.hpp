#ifndef PROJECT_FLOW_EQUATION_HPP
#define PROJECT_FLOW_EQUATION_HPP

#include <vector>
#include <string>
#include <memory>

#include <param_helper/json.hpp>

#include <odesolver/header.hpp>
#include <odesolver/dev_dat.hpp>


using json = nlohmann::json;


namespace odesolver {
    namespace flowequations {
        class FlowEquationsWrapper : public std::enable_shared_from_this<FlowEquationsWrapper>
        {
        public:
            virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int dim_index) = 0;
            
            virtual size_t get_dim() = 0;

            virtual json get_json() const;

            static std::string name();
        };

        struct FlowEquation
        {
            virtual void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) = 0;
        };

        template<typename FlowEquations, typename... Args>
        std::shared_ptr<FlowEquationsWrapper> generate_flow_equations(Args... args)
        {
            return std::make_shared<FlowEquations>(args...);
        }

        odesolver::DevDatC compute_flow(const odesolver::DevDatC &coordinates, FlowEquationsWrapper * flow_equations);

        void compute_flow(const odesolver::DevDatC &coordinates, odesolver::DevDatC &vertex_velocities, FlowEquationsWrapper * flow_equations);
    }
}

#endif //PROJECT_FLOW_EQUATION_HPP
