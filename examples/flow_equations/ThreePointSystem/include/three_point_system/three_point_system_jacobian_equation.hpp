#ifndef PROJECT_THREEPOINTSYSTEMJACOBIANEQUATION_HPP
#define PROJECT_THREEPOINTSYSTEMJACOBIANEQUATION_HPP

#include <math.h>
#include <tuple>

#include <odesolver/flow_equations/jacobian_equation.hpp>


struct ThreePointSystemJacobianEquation0 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation0(const cudaT k) : k_(k),
		const_expr0_((1*1.0/2) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation1 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation1(const cudaT k) : k_(k),
		const_expr0_((4*1.0/3) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation2 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation2(const cudaT k) : k_(k),
		const_expr0_((1*1.0/6) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation3 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation3(const cudaT k) : k_(k),
		const_expr0_((1*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation4 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation4(const cudaT k) : k_(k),
		const_expr0_(-1140 * M_PI),
		const_expr1_((1*1.0/570) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
};


struct ThreePointSystemJacobianEquation5 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation5(const cudaT k) : k_(k),
		const_expr0_((1*1.0/1140) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation6 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation6(const cudaT k) : k_(k),
		const_expr0_((2*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation7 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation7(const cudaT k) : k_(k),
		const_expr0_((8*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


struct ThreePointSystemJacobianEquation8 : public odesolver::flowequations::JacobianEquation
{
	ThreePointSystemJacobianEquation8(const cudaT k) : k_(k),
		const_expr0_(570 * M_PI),
		const_expr1_((1*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (odesolver::DimensionIteratorC &derivatives, const odesolver::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
	const cudaT const_expr1_;
};


class ThreePointSystemJacobianEquations : public odesolver::flowequations::JacobianEquationsWrapper
{
public:
	ThreePointSystemJacobianEquations(const cudaT k) : k_(k)
	{
		jacobian_equations_ = std::vector<std::shared_ptr<odesolver::flowequations::JacobianEquation>> {
			std::make_shared<ThreePointSystemJacobianEquation0>(k),
			std::make_shared<ThreePointSystemJacobianEquation1>(k),
			std::make_shared<ThreePointSystemJacobianEquation2>(k),
			std::make_shared<ThreePointSystemJacobianEquation3>(k),
			std::make_shared<ThreePointSystemJacobianEquation4>(k),
			std::make_shared<ThreePointSystemJacobianEquation5>(k),
			std::make_shared<ThreePointSystemJacobianEquation6>(k),
			std::make_shared<ThreePointSystemJacobianEquation7>(k),
			std::make_shared<ThreePointSystemJacobianEquation8>(k)
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

#endif //PROJECT_THREEPOINTSYSTEMJACOBIANEQUATION_HPP
