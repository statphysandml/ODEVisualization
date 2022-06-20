#ifndef PROJECT_LORENTZATTRACTORFLOWEQUATION_HPP
#define PROJECT_LORENTZATTRACTORFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include <odesolver/flow_equations/flow_equation.hpp>


struct LorentzAttractorFlowEquation0 : public odesolver::flowequations::FlowEquation
{
	LorentzAttractorFlowEquation0(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorFlowEquation1 : public odesolver::flowequations::FlowEquation
{
	LorentzAttractorFlowEquation1(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorFlowEquation2 : public odesolver::flowequations::FlowEquation
{
	LorentzAttractorFlowEquation2(const cudaT k) : k_(k),
		const_expr0_(-8*1.0/3)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


class LorentzAttractorFlowEquations : public odesolver::flowequations::FlowEquationsWrapper
{
public:
	LorentzAttractorFlowEquations(const cudaT k) : k_(k)
	{
		flow_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::FlowEquation>> {
			std::make_shared<LorentzAttractorFlowEquation0>(k_),
			std::make_shared<LorentzAttractorFlowEquation1>(k_),
			std::make_shared<LorentzAttractorFlowEquation2>(k_)
		};
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int dim_index) override
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
	std::vector<std::shared_ptr<odesolver::flowequations::FlowEquation>> flow_equations_;
};

#endif //PROJECT_LORENTZATTRACTORFLOWEQUATION_HPP
