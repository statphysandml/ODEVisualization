#ifndef PROJECT_NONLINEARSYSTEMFLOWEQUATION_HPP
#define PROJECT_NONLINEARSYSTEMFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include <flowequations/flow_equation.hpp>


struct NonlinearSystemFlowEquation0 : public flowequations::FlowEquation
{
	NonlinearSystemFlowEquation0(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct NonlinearSystemFlowEquation1 : public flowequations::FlowEquation
{
	NonlinearSystemFlowEquation1(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


class NonlinearSystemFlowEquations : public flowequations::FlowEquationsWrapper
{
public:
	NonlinearSystemFlowEquations(const cudaT k) : k_(k)
	{
		flow_equations_ = std::vector<std::shared_ptr<flowequations::FlowEquation>> {
			std::make_shared<NonlinearSystemFlowEquation0>(k_),
			std::make_shared<NonlinearSystemFlowEquation1>(k_)
		};
	}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables, const int dim_index) override
	{
		(*flow_equations_[dim_index])(derivatives, variables);
	}

	size_t get_dim() override
	{
		return dim_;
	}

	static std::string model_;
	static size_t dim_;
	static std::string explicit_variable_;
	static std::vector<std::string> explicit_functions_;

	json get_json() const override
	{
		return json {
			{"model", model_},
			{"dim", dim_},
			{"explicit_variable", explicit_variable_},
			{"explicit_functions", explicit_functions_}
		};
	}

private:
	const cudaT k_;
	std::vector<std::shared_ptr<flowequations::FlowEquation>> flow_equations_;
};

#endif //PROJECT_NONLINEARSYSTEMFLOWEQUATION_HPP
