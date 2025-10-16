#include <{{ cookiecutter.project_slug.replace("-", "_") }}/{{ cookiecutter.project_slug.replace("-", "_") }}_jacobian_equation.hpp>

std::string {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations::model_ = "{{ cookiecutter.project_slug.replace("-", "_") }}";
size_t {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations::dim_ = 3;


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation0::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(derivatives.begin(), derivatives.end(), -10);
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation1::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(derivatives.begin(), derivatives.end(), 10);
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation2::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(derivatives.begin(), derivatives.end(), 0);
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation3::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[2].begin(), variables[2].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return 28 + (-1 * val1); });
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation4::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(derivatives.begin(), derivatives.end(), -1);
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation5::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1) { return -1 * val1; });
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation6::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[1].begin(), variables[1].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val) { return val; });
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation7::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), derivatives.begin(), [] __host__ __device__ (const cudaT &val) { return val; });
}


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation8::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::fill(derivatives.begin(), derivatives.end(), const_expr0_);
}

