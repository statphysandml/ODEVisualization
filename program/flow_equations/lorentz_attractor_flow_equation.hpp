#ifndef PROJECT_LORENTZATTRACTORFLOWEQUATION_HPP
#define PROJECT_LORENTZATTRACTORFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"

struct LorentzAttractorFlowEquation0 : public FlowEquation
{
	LorentzAttractorFlowEquation0(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 10 * ((-1 * val1) + val2); });
	}

private:
	const cudaT k;

};


struct comp_func_lorentz_attractor0
{
	comp_func_lorentz_attractor0()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (-1 * thrust::get<2>(t)) + (thrust::get<1>(t) * (28 + (-1 * thrust::get<0>(t))));
	}
};

struct LorentzAttractorFlowEquation1 : public FlowEquation
{
	LorentzAttractorFlowEquation1(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_lorentz_attractor0());
	}

private:
	const cudaT k;

};


struct comp_func_lorentz_attractor1
{
	const cudaT const_expr0;

	comp_func_lorentz_attractor1(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (thrust::get<1>(t) * thrust::get<2>(t)) + (const_expr0 * thrust::get<0>(t));
	}
};

struct LorentzAttractorFlowEquation2 : public FlowEquation
{
	LorentzAttractorFlowEquation2(const cudaT k_) : k(k_),
		const_expr0(-8*1.0/3)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_lorentz_attractor1(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};

class LorentzAttractorFlowEquations : public FlowEquationsWrapper
{
public:
	LorentzAttractorFlowEquations(const cudaT k_) : k(k_)
	{
		flow_equations = std::vector< FlowEquation* > {
			new LorentzAttractorFlowEquation0(k),
			new LorentzAttractorFlowEquation1(k),
			new LorentzAttractorFlowEquation2(k)
		};
	}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int dim_index) override
	{
		(*flow_equations[dim_index])(derivatives, variables);
	}

	uint8_t get_dim() override
	{
		return dim;
	}

    virtual bool pre_installed_theory()
    {
        return true;
    }

	static std::string name()
	{
		return "lorentz_attractor";
	}

	const static uint8_t dim = 3;

private:
	const cudaT k;
	std::vector < FlowEquation* > flow_equations;
};

# endif //PROJECT_LORENTZATTRACTORFLOWEQUATION_HPP
