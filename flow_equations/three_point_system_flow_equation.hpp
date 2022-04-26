#ifndef PROJECT_THREEPOINTSYSTEMFLOWEQUATION_HPP
#define PROJECT_THREEPOINTSYSTEMFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"


struct comp_func_three_point_system0
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_three_point_system0(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (-2 * thrust::get<0>(t)) + (thrust::get<2>(t) * (const_expr0 + ((210 + (-960 * thrust::get<1>(t)) + (1920 * (pow(thrust::get<1>(t), 2)))) * (pow((1 + thrust::get<0>(t)), -3)) * const_expr1) + ((-24 + (48 * thrust::get<1>(t))) * (pow((1 + thrust::get<0>(t)), -2)) * const_expr2)));
	}
};

struct ThreePointSystemFlowEquation0 : public FlowEquation
{
	ThreePointSystemFlowEquation0(const cudaT k_) : k(k_),
		const_expr0(-2 * (pow(M_PI, -1))),
		const_expr1((1*1.0/180) * (pow(M_PI, -1))),
		const_expr2((1*1.0/12) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[1].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[1].end(), variables[2].end(), derivatives.end())), comp_func_three_point_system0(const_expr0, const_expr1, const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_three_point_system1
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr10;
	const cudaT const_expr11;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
	const cudaT const_expr5;
	const cudaT const_expr6;
	const cudaT const_expr7;
	const cudaT const_expr8;
	const cudaT const_expr9;

	comp_func_three_point_system1(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr10_, const cudaT const_expr11_, const cudaT const_expr2_, const cudaT const_expr3_, const cudaT const_expr4_, const cudaT const_expr5_, const cudaT const_expr6_, const cudaT const_expr7_, const cudaT const_expr8_, const cudaT const_expr9_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr10(const_expr10_), const_expr11(const_expr11_), const_expr2(const_expr2_), const_expr3(const_expr3_), const_expr4(const_expr4_), const_expr5(const_expr5_), const_expr6(const_expr6_), const_expr7(const_expr7_), const_expr8(const_expr8_), const_expr9(const_expr9_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (thrust::get<2>(t) * (const_expr0 + ((const_expr1 + (-4 * thrust::get<1>(t) * (3 + (2 * (-2 + thrust::get<1>(t)) * thrust::get<1>(t))))) * (pow((1 + thrust::get<0>(t)), -4)) * const_expr5) + ((24 + (-96 * thrust::get<1>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), -3)) * const_expr6) + ((8 + (-24 * thrust::get<1>(t))) * (pow((1 + thrust::get<0>(t)), -2)) * const_expr7))) + (-1 * thrust::get<1>(t) * (1 + (const_expr2 * (pow(thrust::get<2>(t), -1)) * ((2 * thrust::get<2>(t)) + ((pow(thrust::get<2>(t), 2)) * (const_expr3 + ((-299 + (1780 * thrust::get<1>(t)) + (-3640 * (pow(thrust::get<1>(t), 2))) + (2336 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<0>(t)), -5)) * const_expr8) + ((1 + (-3 * thrust::get<1>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), -4)) * const_expr9) + ((const_expr4 + (-124 * thrust::get<1>(t)) + (169 * (pow(thrust::get<1>(t), 2))) + (864 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<0>(t)), -4)) * const_expr10) + ((15 + (118 * thrust::get<1>(t)) + (-60 * thrust::get<1>(t) * (1 + thrust::get<1>(t)))) * (pow((1 + thrust::get<0>(t)), -3)) * const_expr10) + ((pow((1 + thrust::get<0>(t)), -2)) * const_expr11)))))));
	}
};

struct ThreePointSystemFlowEquation1 : public FlowEquation
{
	ThreePointSystemFlowEquation1(const cudaT k_) : k(k_),
		const_expr0((6*1.0/5) * (pow(M_PI, -1))),
		const_expr1(11*1.0/5),
		const_expr2(1*1.0/2),
		const_expr3((-5*1.0/19) * (pow(M_PI, -1))),
		const_expr4(49*1.0/4),
		const_expr5((-1*1.0/4) * (pow(M_PI, -1))),
		const_expr6((1*1.0/6) * (pow(M_PI, -1))),
		const_expr7((1*1.0/8) * (pow(M_PI, -1))),
		const_expr8((2*1.0/285) * (pow(M_PI, -1))),
		const_expr9((16*1.0/19) * (pow(M_PI, -1))),
		const_expr10((4*1.0/57) * (pow(M_PI, -1))),
		const_expr11((-47*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[1].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[1].end(), variables[2].end(), derivatives.end())), comp_func_three_point_system1(const_expr0, const_expr1, const_expr10, const_expr11, const_expr2, const_expr3, const_expr4, const_expr5, const_expr6, const_expr7, const_expr8, const_expr9));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
	const cudaT const_expr5;
	const cudaT const_expr6;
	const cudaT const_expr7;
	const cudaT const_expr8;
	const cudaT const_expr9;
	const cudaT const_expr10;
	const cudaT const_expr11;
};


struct comp_func_three_point_system2
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
	const cudaT const_expr5;

	comp_func_three_point_system2(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_, const cudaT const_expr3_, const cudaT const_expr4_, const cudaT const_expr5_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_), const_expr3(const_expr3_), const_expr4(const_expr4_), const_expr5(const_expr5_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (2 * thrust::get<2>(t)) + ((pow(thrust::get<2>(t), 2)) * (const_expr0 + ((-299 + (1780 * thrust::get<1>(t)) + (-3640 * (pow(thrust::get<1>(t), 2))) + (2336 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<0>(t)), -5)) * const_expr2) + ((1 + (-3 * thrust::get<1>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), -4)) * const_expr3) + ((const_expr1 + (-124 * thrust::get<1>(t)) + (169 * (pow(thrust::get<1>(t), 2))) + (864 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<0>(t)), -4)) * const_expr4) + ((15 + (118 * thrust::get<1>(t)) + (-60 * thrust::get<1>(t) * (1 + thrust::get<1>(t)))) * (pow((1 + thrust::get<0>(t)), -3)) * const_expr4) + ((pow((1 + thrust::get<0>(t)), -2)) * const_expr5)));
	}
};

struct ThreePointSystemFlowEquation2 : public FlowEquation
{
	ThreePointSystemFlowEquation2(const cudaT k_) : k(k_),
		const_expr0((-5*1.0/19) * (pow(M_PI, -1))),
		const_expr1(49*1.0/4),
		const_expr2((2*1.0/285) * (pow(M_PI, -1))),
		const_expr3((16*1.0/19) * (pow(M_PI, -1))),
		const_expr4((4*1.0/57) * (pow(M_PI, -1))),
		const_expr5((-47*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[1].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[1].end(), variables[2].end(), derivatives.end())), comp_func_three_point_system2(const_expr0, const_expr1, const_expr2, const_expr3, const_expr4, const_expr5));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
	const cudaT const_expr5;
};

class ThreePointSystemFlowEquations : public FlowEquationsWrapper
{
public:
	ThreePointSystemFlowEquations(const cudaT k_) : k(k_)
	{
		flow_equations = std::vector< FlowEquation* > {
			new ThreePointSystemFlowEquation0(k),
			new ThreePointSystemFlowEquation1(k),
			new ThreePointSystemFlowEquation2(k)
		};
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int dim_index) override
	{
		(*flow_equations[dim_index])(derivatives, variables);
	}

	uint8_t get_dim() override
	{
		return dim;
	}

    bool pre_installed_theory()
    {
        return true;
    }

	static std::string name()
	{
		return "three_point_system";
	}

	const static uint8_t dim = 3;

private:
	const cudaT k;
	std::vector < FlowEquation* > flow_equations;
};

# endif //PROJECT_THREEPOINTSYSTEMFLOWEQUATION_HPP
