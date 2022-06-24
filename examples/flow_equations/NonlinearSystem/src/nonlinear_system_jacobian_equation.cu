#include <nonlinear_system/nonlinear_system_jacobian_equation.hpp>

std::string NonlinearSystemJacobianEquations::model_ = "nonlinear_system";
size_t NonlinearSystemJacobianEquations::dim_ = 2;


void NonlinearSystemJacobianEquation0::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
}


void NonlinearSystemJacobianEquation1::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), -1);
}


void NonlinearSystemJacobianEquation2::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return 2 * val1; });
}


void NonlinearSystemJacobianEquation3::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[1].begin(), variables[1].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return -2 * val1; });
}

