#ifndef PROJECT_NONLINEARSYSTEMJACOBIANEQUATION_HPP
#define PROJECT_NONLINEARSYSTEMJACOBIANEQUATION_HPP

#include <math.h>
#include <tuple>

#include <flowequations/jacobian_equation.hpp>


struct NonlinearSystemJacobianEquation0 : public flowequations::JacobianEquation
{
	NonlinearSystemJacobianEquation0(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct NonlinearSystemJacobianEquation1 : public flowequations::JacobianEquation
{
	NonlinearSystemJacobianEquation1(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct NonlinearSystemJacobianEquation2 : public flowequations::JacobianEquation
{
	NonlinearSystemJacobianEquation2(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct NonlinearSystemJacobianEquation3 : public flowequations::JacobianEquation
{
	NonlinearSystemJacobianEquation3(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


class NonlinearSystemJacobianEquations : public flowequations::JacobianEquationsWrapper
{
public:
	NonlinearSystemJacobianEquations(const cudaT k) : k_(k)
	{
		jacobian_equations_ = std::vector<std::shared_ptr<flowequations::JacobianEquation>> {
			std::make_shared<NonlinearSystemJacobianEquation0>(k),
			std::make_shared<NonlinearSystemJacobianEquation1>(k),
			std::make_shared<NonlinearSystemJacobianEquation2>(k),
			std::make_shared<NonlinearSystemJacobianEquation3>(k)
		};
	}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables, const int row_idx, const int col_idx) override
	{
		(*jacobian_equations_[row_idx * dim_ + col_idx])(derivatives, variables);
	}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables, const int matrix_idx) override
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
	std::vector<std::shared_ptr<flowequations::JacobianEquation>> jacobian_equations_;
};

#endif //PROJECT_NONLINEARSYSTEMJACOBIANEQUATION_HPP
