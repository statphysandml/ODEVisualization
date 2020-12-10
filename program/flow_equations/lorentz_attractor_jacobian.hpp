#ifndef PROJECT_LORENTZATTRACTORJACOBIAN_HPP
#define PROJECT_LORENTZATTRACTORJACOBIAN_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/jacobian_equation.hpp"

struct LorentzAttractorJacobianEquation0 : public JacobianEquation
{
	LorentzAttractorJacobianEquation0(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), -10);
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation1 : public JacobianEquation
{
	LorentzAttractorJacobianEquation1(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 10);
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation2 : public JacobianEquation
{
	LorentzAttractorJacobianEquation2(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation3 : public JacobianEquation
{
	LorentzAttractorJacobianEquation3(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[2].begin(), variables[2].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return 28 + (-1 * val1); });
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation4 : public JacobianEquation
{
	LorentzAttractorJacobianEquation4(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), -1);
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation5 : public JacobianEquation
{
	LorentzAttractorJacobianEquation5(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return -1 * val1; });
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation6 : public JacobianEquation
{
	LorentzAttractorJacobianEquation6(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[1].begin(), variables[1].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val) { return val; });
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation7 : public JacobianEquation
{
	LorentzAttractorJacobianEquation7(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val) { return val; });
	}

private:
	const cudaT k;

};

struct LorentzAttractorJacobianEquation8 : public JacobianEquation
{
	LorentzAttractorJacobianEquation8(const cudaT k_) : k(k_),
		const_expr0(-8*1.0/3)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), const_expr0);
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};

class LorentzAttractorJacobianEquations : public JacobianWrapper
{
public:
	LorentzAttractorJacobianEquations(const cudaT k_) : k(k_)
	{
		jacobian_equations = std::vector< JacobianEquation* > {
			new LorentzAttractorJacobianEquation0(k),
			new LorentzAttractorJacobianEquation1(k),
			new LorentzAttractorJacobianEquation2(k),
			new LorentzAttractorJacobianEquation3(k),
			new LorentzAttractorJacobianEquation4(k),
			new LorentzAttractorJacobianEquation5(k),
			new LorentzAttractorJacobianEquation6(k),
			new LorentzAttractorJacobianEquation7(k),
			new LorentzAttractorJacobianEquation8(k)
		};
	}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int row_idx, const int col_idx) override
	{
		(*jacobian_equations[row_idx * dim + col_idx])(derivatives, variables);
	}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int matrix_idx) override
	{
		(*jacobian_equations[matrix_idx])(derivatives, variables);
	}

	uint8_t get_dim() override
	{
		return dim;
	}

	static std::string name()
	{
		return "lorentz_attractor";
	}

	const static uint8_t dim = 3;

private:
	const cudaT k;
	std::vector < JacobianEquation* > jacobian_equations;
};

# endif //PROJECT_LORENTZATTRACTORJACOBIAN_HPP
