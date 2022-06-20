#include <{{ cookiecutter.project_slug.replace("-", "_") }}/{{ cookiecutter.project_slug.replace("-", "_") }}_flow_equation.hpp>

std::string {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::model_ = "{{ cookiecutter.project_slug.replace("-", "_") }}";
size_t {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::dim_ = 3;
std::string {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::explicit_variable_ = "k";
std::vector<std::string> {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquations::explicit_functions_ = {"x", "y", "z"};


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquation0::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::transform(variables[1].begin(), variables[1].end(), variables[0].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 10 * ((-1 * val2) + val1); });
}


struct comp_func_{{ cookiecutter.project_slug.replace("-", "_") }}0
{
	comp_func_{{ cookiecutter.project_slug.replace("-", "_") }}0()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (-1 * thrust::get<1>(t)) + (thrust::get<2>(t) * (28 + (-1 * thrust::get<0>(t))));
	}
};


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquation1::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[0].end(), derivatives.end())), comp_func_{{ cookiecutter.project_slug.replace("-", "_") }}0());
}


struct comp_func_{{ cookiecutter.project_slug.replace("-", "_") }}1
{
	const cudaT const_expr0_;

	comp_func_{{ cookiecutter.project_slug.replace("-", "_") }}1(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (thrust::get<2>(t) * thrust::get<1>(t)) + (const_expr0_ * thrust::get<0>(t));
	}
};


void {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}FlowEquation2::operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[0].end(), derivatives.end())), comp_func_{{ cookiecutter.project_slug.replace("-", "_") }}1(const_expr0_));
}

