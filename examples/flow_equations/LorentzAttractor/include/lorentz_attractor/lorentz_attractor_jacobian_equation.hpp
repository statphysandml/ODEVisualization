#ifndef PROJECT_LORENTZATTRACTORJACOBIAN_HPP
#define PROJECT_LORENTZATTRACTORJACOBIAN_HPP

#include <math.h>
#include <tuple>

#include <flowequations/jacobian_equation.hpp>


struct LorentzAttractorJacobianEquation0 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation0(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation1 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation1(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation2 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation2(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation3 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation3(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation4 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation4(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation5 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation5(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation6 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation6(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation7 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation7(const cudaT k) : k_(k)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
};


struct LorentzAttractorJacobianEquation8 : public flowequations::JacobianEquation
{
	LorentzAttractorJacobianEquation8(const cudaT k) : k_(k),
		const_expr0_(-8*1.0/3)
	{}

	void operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables) override;

private:
	const cudaT k_;
	const cudaT const_expr0_;
};


class LorentzAttractorJacobianEquations : public flowequations::JacobianEquationsWrapper
{
public:
	LorentzAttractorJacobianEquations(const cudaT k) : k_(k)
	{
		jacobian_equations_ = std::vector<std::shared_ptr<flowequations::JacobianEquation>> {
			std::make_shared<LorentzAttractorJacobianEquation0>(k),
			std::make_shared<LorentzAttractorJacobianEquation1>(k),
			std::make_shared<LorentzAttractorJacobianEquation2>(k),
			std::make_shared<LorentzAttractorJacobianEquation3>(k),
			std::make_shared<LorentzAttractorJacobianEquation4>(k),
			std::make_shared<LorentzAttractorJacobianEquation5>(k),
			std::make_shared<LorentzAttractorJacobianEquation6>(k),
			std::make_shared<LorentzAttractorJacobianEquation7>(k),
			std::make_shared<LorentzAttractorJacobianEquation8>(k)
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

#endif //PROJECT_LORENTZATTRACTORJACOBIAN_HPP
