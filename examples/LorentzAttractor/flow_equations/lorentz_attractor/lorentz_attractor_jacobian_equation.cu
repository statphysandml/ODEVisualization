#include "lorentz_attractor_jacobian_equation.hpp"

std::string LorentzAttractorJacobianEquations::model_ = "lorentz_attractor";
size_t LorentzAttractorJacobianEquations::dim_ = 3;


void LorentzAttractorJacobianEquation0::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), -10);
}


void LorentzAttractorJacobianEquation1::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 10);
}


void LorentzAttractorJacobianEquation2::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
}


void LorentzAttractorJacobianEquation3::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[2].begin(), variables[2].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return 28 + (-1 * val1); });
}


void LorentzAttractorJacobianEquation4::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), -1);
}


void LorentzAttractorJacobianEquation5::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return -1 * val1; });
}


void LorentzAttractorJacobianEquation6::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[1].begin(), variables[1].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val) { return val; });
}


void LorentzAttractorJacobianEquation7::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val) { return val; });
}


void LorentzAttractorJacobianEquation8::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), const_expr0_);
}

