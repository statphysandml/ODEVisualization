#ifndef PROJECT_SMALLTHREELEVELSYSTEMFLOWEQUATION_HPP
#define PROJECT_SMALLTHREELEVELSYSTEMFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"

struct SmallThreeLevelSystemFlowEquation0 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation0(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[7].begin(), variables[7].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return 2 * val1; });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation1 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation1(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[6].begin(), variables[6].end(), variables[9].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-0.05 * val2) + (2 * val1); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation2 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation2(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[2].begin(), variables[2].end(), variables[10].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-3 * val1) + (-2 * val2); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation3 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation3(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[7].begin(), variables[7].end(), variables[3].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-6 * val2) + (-0.05 * val1); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation4 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation4(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[8].begin(), variables[8].end(), variables[11].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-2 * val1) + (0.05 * val2); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation5 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation5(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[10].begin(), variables[10].end(), variables[5].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-3 * val2) + (0.05 * val1); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation6 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation6(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[1].begin(), variables[1].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return -2 * val1; });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation7 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation7(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[3].begin(), variables[3].end(), variables[0].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (0.05 * val1) + (-2 * val2); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation8 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation8(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[8].begin(), variables[8].end(), variables[4].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-3 * val1) + (2 * val2); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation9 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation9(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[1].begin(), variables[1].end(), variables[9].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 2 + (-6 * val2) + (0.05 * val1); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation10 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation10(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[2].begin(), variables[2].end(), variables[5].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (2 * val1) + (-0.05 * val2); });
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemFlowEquation11 : public FlowEquation
{
	SmallThreeLevelSystemFlowEquation11(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[11].begin(), variables[11].end(), variables[4].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return -2 + (-3 * val1) + (-0.05 * val2); });
	}

private:
	const cudaT k;

};

class SmallThreeLevelSystemFlowEquations : public FlowEquationsWrapper
{
public:
	SmallThreeLevelSystemFlowEquations(const cudaT k_) : k(k_)
	{
		flow_equations = std::vector< FlowEquation* > {
			new SmallThreeLevelSystemFlowEquation0(k),
			new SmallThreeLevelSystemFlowEquation1(k),
			new SmallThreeLevelSystemFlowEquation2(k),
			new SmallThreeLevelSystemFlowEquation3(k),
			new SmallThreeLevelSystemFlowEquation4(k),
			new SmallThreeLevelSystemFlowEquation5(k),
			new SmallThreeLevelSystemFlowEquation6(k),
			new SmallThreeLevelSystemFlowEquation7(k),
			new SmallThreeLevelSystemFlowEquation8(k),
			new SmallThreeLevelSystemFlowEquation9(k),
			new SmallThreeLevelSystemFlowEquation10(k),
			new SmallThreeLevelSystemFlowEquation11(k)
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

	static std::string name()
	{
		return "small_three_level_system";
	}

	const static uint8_t dim = 12;

private:
	const cudaT k;
	std::vector < FlowEquation* > flow_equations;
};

# endif //PROJECT_SMALLTHREELEVELSYSTEMFLOWEQUATION_HPP
