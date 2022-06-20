#ifndef PROJECT_{{ cookiecutter.project_slug.replace("-", "").upper() }}JACOBIAN_HPP
#define PROJECT_{{ cookiecutter.project_slug.replace("-", "").upper() }}JACOBIAN_HPP

#include <math.h>
#include <tuple>

#include <odesolver/flow_equations/jacobian_equation.hpp>


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation0 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation0(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation1 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation1(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation2 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation2(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation3 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation3(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation4 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation4(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation5 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation5(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation6 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation6(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation7 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation7(const cudaT k) : k_(k)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation8 : public odesolver::flowequations::JacobianEquation
{
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation8(const cudaT k) : k_(k),
		const_expr0_(-8*1.0/3)
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


class {{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations : public odesolver::flowequations::JacobianEquationsWrapper
{
public:
	{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquations(const cudaT k) : k_(k)
	{
		jacobian_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::JacobianEquation>> {
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation0>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation1>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation2>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation3>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation4>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation5>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation6>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation7>(k),
			std::make_shared<{{ cookiecutter.project_slug.replace("-", " ").title().replace(" ", "") }}JacobianEquation8>(k)
		};
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int row_idx, const int col_idx) override
	{
		(*jacobian_equations_[row_idx * dim_ + col_idx])(derivatives, variables);
	}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables, const int matrix_idx) override
	{
		(*jacobian_equations_[matrix_idx])(derivatives, variables);
	}

	size_t get_dim() override
	{
		return dim_;
	}

	static std::string model_;
	static size_t dim_;

private:
	const cudaT k_;
	std::vector<std::shared_ptr<odesolver::flowequations::JacobianEquation>> jacobian_equations_;
};

#endif //PROJECT_{{ cookiecutter.project_slug.replace("-", "").upper() }}JACOBIAN_HPP
