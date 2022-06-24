#include <nonlinear_system/nonlinear_system_flow_equation.hpp>

std::string NonlinearSystemFlowEquations::model_ = "nonlinear_system";
size_t NonlinearSystemFlowEquations::dim_ = 2;
std::string NonlinearSystemFlowEquations::explicit_variable_ = "t";
std::vector<std::string> NonlinearSystemFlowEquations::explicit_functions_ = {"u", "v"};


void NonlinearSystemFlowEquation0::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[1].begin(), variables[1].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return 1 + (-1 * val1); });
}


void NonlinearSystemFlowEquation1::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[1].begin(), variables[1].end(), variables[0].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (pow(val2, 2)) + (-1 * (pow(val1, 2))); });
}

