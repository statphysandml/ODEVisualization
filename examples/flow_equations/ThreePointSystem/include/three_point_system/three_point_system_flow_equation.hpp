#ifndef PROJECT_THREEPOINTSYSTEMFLOWEQUATION_HPP
#define PROJECT_THREEPOINTSYSTEMFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include <odesolver/flow_equations/flow_equation.hpp>


struct ThreePointSystemFlowEquation0 : public odesolver::flowequations::FlowEquation
{
	ThreePointSystemFlowEquation0(const cudaT k) : k_(k),
		const_expr0_(-2 * (pow(M_PI, -1))),
		const_expr1_((1*1.0/180) * (pow(M_PI, -1))),
		const_expr2_((1*1.0/12) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
};


struct ThreePointSystemFlowEquation1 : public odesolver::flowequations::FlowEquation
{
	ThreePointSystemFlowEquation1(const cudaT k) : k_(k),
		const_expr0_((6*1.0/5) * (pow(M_PI, -1))),
		const_expr1_(11*1.0/5),
		const_expr2_(1*1.0/2),
		const_expr3_((-5*1.0/19) * (pow(M_PI, -1))),
		const_expr4_(49*1.0/4),
		const_expr5_((-1*1.0/4) * (pow(M_PI, -1))),
		const_expr6_((1*1.0/6) * (pow(M_PI, -1))),
		const_expr7_((1*1.0/8) * (pow(M_PI, -1))),
		const_expr8_((2*1.0/285) * (pow(M_PI, -1))),
		const_expr9_((16*1.0/19) * (pow(M_PI, -1))),
		const_expr10_((4*1.0/57) * (pow(M_PI, -1))),
		const_expr11_((-47*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
	const cudaT const_expr6_;
	const cudaT const_expr7_;
	const cudaT const_expr8_;
	const cudaT const_expr9_;
	const cudaT const_expr10_;
	const cudaT const_expr11_;
};


struct ThreePointSystemFlowEquation2 : public odesolver::flowequations::FlowEquation
{
	ThreePointSystemFlowEquation2(const cudaT k) : k_(k),
		const_expr0_((-5*1.0/19) * (pow(M_PI, -1))),
		const_expr1_(49*1.0/4),
		const_expr2_((2*1.0/285) * (pow(M_PI, -1))),
		const_expr3_((16*1.0/19) * (pow(M_PI, -1))),
		const_expr4_((4*1.0/57) * (pow(M_PI, -1))),
		const_expr5_((-47*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
};


class ThreePointSystemFlowEquations : public odesolver::flowequations::FlowEquationsWrapper
{
public:
	ThreePointSystemFlowEquations(const cudaT k) : k_(k)
	{
		flow_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::FlowEquation>> {
			std::make_shared<ThreePointSystemFlowEquation0>(k_),
			std::make_shared<ThreePointSystemFlowEquation1>(k_),
			std::make_shared<ThreePointSystemFlowEquation2>(k_)
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

#endif //PROJECT_THREEPOINTSYSTEMFLOWEQUATION_HPP
