#ifndef PROJECT_THREELEVELSYSTEMJACOBIAN_HPP
#define PROJECT_THREELEVELSYSTEMJACOBIAN_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/jacobian_equation.hpp"

struct ThreeLevelSystemJacobianEquation0 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation0(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation1 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation1(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation2 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation2(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation3 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation3(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation4 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation4(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation5 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation5(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation6 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation6(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation7 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation7(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation8 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation8(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation9 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation9(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation10 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation10(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation11 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation11(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation12 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation12(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation13 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation13(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation14 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation14(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation15 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation15(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation16 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation16(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation17 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation17(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation18 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation18(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation19 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation19(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation20 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation20(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation21 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation21(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation22 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation22(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation23 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation23(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation24 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation24(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation25 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation25(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation26 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation26(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation27 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation27(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation28 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation28(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation29 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation29(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation30 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation30(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation31 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation31(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation32 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation32(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation33 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation33(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation34 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation34(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation35 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation35(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation36 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation36(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation37 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation37(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation38 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation38(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation39 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation39(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation40 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation40(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation41 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation41(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation42 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation42(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation43 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation43(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation44 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation44(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation45 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation45(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation46 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation46(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation47 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation47(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation48 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation48(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation49 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation49(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation50 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation50(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation51 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation51(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation52 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation52(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation53 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation53(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation54 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation54(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation55 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation55(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation56 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation56(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation57 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation57(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation58 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation58(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation59 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation59(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation60 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation60(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation61 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation61(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation62 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation62(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation63 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation63(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation64 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation64(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation65 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation65(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation66 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation66(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation67 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation67(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation68 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation68(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation69 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation69(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation70 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation70(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation71 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation71(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation72 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation72(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation73 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation73(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation74 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation74(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation75 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation75(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation76 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation76(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation77 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation77(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation78 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation78(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation79 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation79(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation80 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation80(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation81 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation81(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation82 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation82(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation83 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation83(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation84 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation84(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation85 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation85(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation86 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation86(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation87 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation87(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation88 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation88(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation89 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation89(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation90 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation90(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation91 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation91(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation92 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation92(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation93 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation93(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation94 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation94(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation95 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation95(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation96 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation96(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation97 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation97(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation98 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation98(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation99 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation99(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation100 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation100(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation101 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation101(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation102 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation102(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation103 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation103(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation104 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation104(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation105 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation105(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation106 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation106(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation107 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation107(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation108 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation108(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation109 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation109(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation110 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation110(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation111 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation111(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation112 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation112(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation113 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation113(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation114 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation114(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation115 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation115(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation116 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation116(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation117 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation117(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation118 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation118(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation119 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation119(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation120 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation120(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation121 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation121(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation122 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation122(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation123 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation123(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation124 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation124(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation125 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation125(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation126 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation126(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation127 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation127(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation128 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation128(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation129 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation129(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation130 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation130(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation131 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation131(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation132 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation132(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation133 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation133(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation134 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation134(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation135 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation135(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation136 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation136(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation137 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation137(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation138 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation138(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation139 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation139(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation140 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation140(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation141 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation141(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation142 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation142(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation143 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation143(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation144 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation144(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation145 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation145(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation146 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation146(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation147 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation147(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation148 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation148(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation149 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation149(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation150 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation150(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation151 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation151(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation152 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation152(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation153 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation153(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation154 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation154(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation155 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation155(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation156 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation156(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation157 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation157(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation158 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation158(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation159 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation159(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation160 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation160(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation161 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation161(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation162 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation162(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation163 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation163(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation164 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation164(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation165 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation165(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation166 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation166(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation167 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation167(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation168 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation168(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation169 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation169(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation170 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation170(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation171 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation171(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation172 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation172(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation173 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation173(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation174 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation174(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation175 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation175(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation176 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation176(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation177 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation177(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation178 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation178(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation179 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation179(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation180 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation180(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation181 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation181(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation182 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation182(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation183 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation183(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation184 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation184(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation185 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation185(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation186 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation186(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation187 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation187(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation188 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation188(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation189 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation189(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation190 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation190(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation191 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation191(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation192 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation192(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation193 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation193(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation194 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation194(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation195 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation195(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation196 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation196(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation197 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation197(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation198 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation198(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation199 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation199(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation200 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation200(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation201 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation201(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation202 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation202(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation203 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation203(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation204 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation204(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation205 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation205(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation206 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation206(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation207 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation207(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation208 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation208(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation209 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation209(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation210 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation210(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation211 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation211(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation212 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation212(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation213 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation213(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation214 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation214(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation215 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation215(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation216 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation216(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation217 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation217(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation218 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation218(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation219 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation219(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation220 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation220(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation221 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation221(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation222 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation222(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation223 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation223(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation224 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation224(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation225 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation225(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation226 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation226(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation227 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation227(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation228 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation228(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation229 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation229(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation230 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation230(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation231 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation231(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation232 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation232(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation233 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation233(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation234 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation234(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation235 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation235(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation236 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation236(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation237 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation237(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation238 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation238(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation239 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation239(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation240 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation240(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation241 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation241(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation242 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation242(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation243 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation243(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation244 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation244(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation245 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation245(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation246 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation246(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation247 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation247(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation248 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation248(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation249 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation249(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation250 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation250(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation251 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation251(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation252 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation252(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation253 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation253(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation254 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation254(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation255 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation255(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation256 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation256(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation257 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation257(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation258 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation258(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation259 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation259(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation260 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation260(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation261 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation261(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation262 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation262(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation263 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation263(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation264 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation264(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation265 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation265(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation266 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation266(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation267 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation267(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation268 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation268(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation269 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation269(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation270 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation270(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation271 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation271(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation272 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation272(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation273 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation273(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation274 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation274(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation275 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation275(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation276 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation276(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation277 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation277(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation278 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation278(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation279 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation279(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation280 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation280(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation281 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation281(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation282 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation282(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation283 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation283(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation284 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation284(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation285 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation285(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation286 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation286(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation287 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation287(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation288 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation288(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation289 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation289(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation290 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation290(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation291 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation291(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation292 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation292(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation293 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation293(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation294 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation294(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation295 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation295(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation296 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation296(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation297 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation297(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation298 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation298(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation299 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation299(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation300 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation300(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation301 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation301(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation302 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation302(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation303 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation303(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation304 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation304(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation305 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation305(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation306 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation306(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation307 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation307(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation308 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation308(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation309 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation309(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation310 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation310(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation311 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation311(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation312 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation312(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation313 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation313(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation314 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation314(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation315 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation315(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation316 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation316(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation317 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation317(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation318 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation318(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation319 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation319(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation320 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation320(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation321 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation321(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation322 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation322(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemJacobianEquation323 : public JacobianEquation
{
	ThreeLevelSystemJacobianEquation323(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

class ThreeLevelSystemJacobianEquations : public JacobianWrapper
{
public:
	ThreeLevelSystemJacobianEquations(const cudaT k_) : k(k_)
	{
		jacobian_equations = std::vector< JacobianEquation* > {
			new ThreeLevelSystemJacobianEquation0(k),
			new ThreeLevelSystemJacobianEquation1(k),
			new ThreeLevelSystemJacobianEquation2(k),
			new ThreeLevelSystemJacobianEquation3(k),
			new ThreeLevelSystemJacobianEquation4(k),
			new ThreeLevelSystemJacobianEquation5(k),
			new ThreeLevelSystemJacobianEquation6(k),
			new ThreeLevelSystemJacobianEquation7(k),
			new ThreeLevelSystemJacobianEquation8(k),
			new ThreeLevelSystemJacobianEquation9(k),
			new ThreeLevelSystemJacobianEquation10(k),
			new ThreeLevelSystemJacobianEquation11(k),
			new ThreeLevelSystemJacobianEquation12(k),
			new ThreeLevelSystemJacobianEquation13(k),
			new ThreeLevelSystemJacobianEquation14(k),
			new ThreeLevelSystemJacobianEquation15(k),
			new ThreeLevelSystemJacobianEquation16(k),
			new ThreeLevelSystemJacobianEquation17(k),
			new ThreeLevelSystemJacobianEquation18(k),
			new ThreeLevelSystemJacobianEquation19(k),
			new ThreeLevelSystemJacobianEquation20(k),
			new ThreeLevelSystemJacobianEquation21(k),
			new ThreeLevelSystemJacobianEquation22(k),
			new ThreeLevelSystemJacobianEquation23(k),
			new ThreeLevelSystemJacobianEquation24(k),
			new ThreeLevelSystemJacobianEquation25(k),
			new ThreeLevelSystemJacobianEquation26(k),
			new ThreeLevelSystemJacobianEquation27(k),
			new ThreeLevelSystemJacobianEquation28(k),
			new ThreeLevelSystemJacobianEquation29(k),
			new ThreeLevelSystemJacobianEquation30(k),
			new ThreeLevelSystemJacobianEquation31(k),
			new ThreeLevelSystemJacobianEquation32(k),
			new ThreeLevelSystemJacobianEquation33(k),
			new ThreeLevelSystemJacobianEquation34(k),
			new ThreeLevelSystemJacobianEquation35(k),
			new ThreeLevelSystemJacobianEquation36(k),
			new ThreeLevelSystemJacobianEquation37(k),
			new ThreeLevelSystemJacobianEquation38(k),
			new ThreeLevelSystemJacobianEquation39(k),
			new ThreeLevelSystemJacobianEquation40(k),
			new ThreeLevelSystemJacobianEquation41(k),
			new ThreeLevelSystemJacobianEquation42(k),
			new ThreeLevelSystemJacobianEquation43(k),
			new ThreeLevelSystemJacobianEquation44(k),
			new ThreeLevelSystemJacobianEquation45(k),
			new ThreeLevelSystemJacobianEquation46(k),
			new ThreeLevelSystemJacobianEquation47(k),
			new ThreeLevelSystemJacobianEquation48(k),
			new ThreeLevelSystemJacobianEquation49(k),
			new ThreeLevelSystemJacobianEquation50(k),
			new ThreeLevelSystemJacobianEquation51(k),
			new ThreeLevelSystemJacobianEquation52(k),
			new ThreeLevelSystemJacobianEquation53(k),
			new ThreeLevelSystemJacobianEquation54(k),
			new ThreeLevelSystemJacobianEquation55(k),
			new ThreeLevelSystemJacobianEquation56(k),
			new ThreeLevelSystemJacobianEquation57(k),
			new ThreeLevelSystemJacobianEquation58(k),
			new ThreeLevelSystemJacobianEquation59(k),
			new ThreeLevelSystemJacobianEquation60(k),
			new ThreeLevelSystemJacobianEquation61(k),
			new ThreeLevelSystemJacobianEquation62(k),
			new ThreeLevelSystemJacobianEquation63(k),
			new ThreeLevelSystemJacobianEquation64(k),
			new ThreeLevelSystemJacobianEquation65(k),
			new ThreeLevelSystemJacobianEquation66(k),
			new ThreeLevelSystemJacobianEquation67(k),
			new ThreeLevelSystemJacobianEquation68(k),
			new ThreeLevelSystemJacobianEquation69(k),
			new ThreeLevelSystemJacobianEquation70(k),
			new ThreeLevelSystemJacobianEquation71(k),
			new ThreeLevelSystemJacobianEquation72(k),
			new ThreeLevelSystemJacobianEquation73(k),
			new ThreeLevelSystemJacobianEquation74(k),
			new ThreeLevelSystemJacobianEquation75(k),
			new ThreeLevelSystemJacobianEquation76(k),
			new ThreeLevelSystemJacobianEquation77(k),
			new ThreeLevelSystemJacobianEquation78(k),
			new ThreeLevelSystemJacobianEquation79(k),
			new ThreeLevelSystemJacobianEquation80(k),
			new ThreeLevelSystemJacobianEquation81(k),
			new ThreeLevelSystemJacobianEquation82(k),
			new ThreeLevelSystemJacobianEquation83(k),
			new ThreeLevelSystemJacobianEquation84(k),
			new ThreeLevelSystemJacobianEquation85(k),
			new ThreeLevelSystemJacobianEquation86(k),
			new ThreeLevelSystemJacobianEquation87(k),
			new ThreeLevelSystemJacobianEquation88(k),
			new ThreeLevelSystemJacobianEquation89(k),
			new ThreeLevelSystemJacobianEquation90(k),
			new ThreeLevelSystemJacobianEquation91(k),
			new ThreeLevelSystemJacobianEquation92(k),
			new ThreeLevelSystemJacobianEquation93(k),
			new ThreeLevelSystemJacobianEquation94(k),
			new ThreeLevelSystemJacobianEquation95(k),
			new ThreeLevelSystemJacobianEquation96(k),
			new ThreeLevelSystemJacobianEquation97(k),
			new ThreeLevelSystemJacobianEquation98(k),
			new ThreeLevelSystemJacobianEquation99(k),
			new ThreeLevelSystemJacobianEquation100(k),
			new ThreeLevelSystemJacobianEquation101(k),
			new ThreeLevelSystemJacobianEquation102(k),
			new ThreeLevelSystemJacobianEquation103(k),
			new ThreeLevelSystemJacobianEquation104(k),
			new ThreeLevelSystemJacobianEquation105(k),
			new ThreeLevelSystemJacobianEquation106(k),
			new ThreeLevelSystemJacobianEquation107(k),
			new ThreeLevelSystemJacobianEquation108(k),
			new ThreeLevelSystemJacobianEquation109(k),
			new ThreeLevelSystemJacobianEquation110(k),
			new ThreeLevelSystemJacobianEquation111(k),
			new ThreeLevelSystemJacobianEquation112(k),
			new ThreeLevelSystemJacobianEquation113(k),
			new ThreeLevelSystemJacobianEquation114(k),
			new ThreeLevelSystemJacobianEquation115(k),
			new ThreeLevelSystemJacobianEquation116(k),
			new ThreeLevelSystemJacobianEquation117(k),
			new ThreeLevelSystemJacobianEquation118(k),
			new ThreeLevelSystemJacobianEquation119(k),
			new ThreeLevelSystemJacobianEquation120(k),
			new ThreeLevelSystemJacobianEquation121(k),
			new ThreeLevelSystemJacobianEquation122(k),
			new ThreeLevelSystemJacobianEquation123(k),
			new ThreeLevelSystemJacobianEquation124(k),
			new ThreeLevelSystemJacobianEquation125(k),
			new ThreeLevelSystemJacobianEquation126(k),
			new ThreeLevelSystemJacobianEquation127(k),
			new ThreeLevelSystemJacobianEquation128(k),
			new ThreeLevelSystemJacobianEquation129(k),
			new ThreeLevelSystemJacobianEquation130(k),
			new ThreeLevelSystemJacobianEquation131(k),
			new ThreeLevelSystemJacobianEquation132(k),
			new ThreeLevelSystemJacobianEquation133(k),
			new ThreeLevelSystemJacobianEquation134(k),
			new ThreeLevelSystemJacobianEquation135(k),
			new ThreeLevelSystemJacobianEquation136(k),
			new ThreeLevelSystemJacobianEquation137(k),
			new ThreeLevelSystemJacobianEquation138(k),
			new ThreeLevelSystemJacobianEquation139(k),
			new ThreeLevelSystemJacobianEquation140(k),
			new ThreeLevelSystemJacobianEquation141(k),
			new ThreeLevelSystemJacobianEquation142(k),
			new ThreeLevelSystemJacobianEquation143(k),
			new ThreeLevelSystemJacobianEquation144(k),
			new ThreeLevelSystemJacobianEquation145(k),
			new ThreeLevelSystemJacobianEquation146(k),
			new ThreeLevelSystemJacobianEquation147(k),
			new ThreeLevelSystemJacobianEquation148(k),
			new ThreeLevelSystemJacobianEquation149(k),
			new ThreeLevelSystemJacobianEquation150(k),
			new ThreeLevelSystemJacobianEquation151(k),
			new ThreeLevelSystemJacobianEquation152(k),
			new ThreeLevelSystemJacobianEquation153(k),
			new ThreeLevelSystemJacobianEquation154(k),
			new ThreeLevelSystemJacobianEquation155(k),
			new ThreeLevelSystemJacobianEquation156(k),
			new ThreeLevelSystemJacobianEquation157(k),
			new ThreeLevelSystemJacobianEquation158(k),
			new ThreeLevelSystemJacobianEquation159(k),
			new ThreeLevelSystemJacobianEquation160(k),
			new ThreeLevelSystemJacobianEquation161(k),
			new ThreeLevelSystemJacobianEquation162(k),
			new ThreeLevelSystemJacobianEquation163(k),
			new ThreeLevelSystemJacobianEquation164(k),
			new ThreeLevelSystemJacobianEquation165(k),
			new ThreeLevelSystemJacobianEquation166(k),
			new ThreeLevelSystemJacobianEquation167(k),
			new ThreeLevelSystemJacobianEquation168(k),
			new ThreeLevelSystemJacobianEquation169(k),
			new ThreeLevelSystemJacobianEquation170(k),
			new ThreeLevelSystemJacobianEquation171(k),
			new ThreeLevelSystemJacobianEquation172(k),
			new ThreeLevelSystemJacobianEquation173(k),
			new ThreeLevelSystemJacobianEquation174(k),
			new ThreeLevelSystemJacobianEquation175(k),
			new ThreeLevelSystemJacobianEquation176(k),
			new ThreeLevelSystemJacobianEquation177(k),
			new ThreeLevelSystemJacobianEquation178(k),
			new ThreeLevelSystemJacobianEquation179(k),
			new ThreeLevelSystemJacobianEquation180(k),
			new ThreeLevelSystemJacobianEquation181(k),
			new ThreeLevelSystemJacobianEquation182(k),
			new ThreeLevelSystemJacobianEquation183(k),
			new ThreeLevelSystemJacobianEquation184(k),
			new ThreeLevelSystemJacobianEquation185(k),
			new ThreeLevelSystemJacobianEquation186(k),
			new ThreeLevelSystemJacobianEquation187(k),
			new ThreeLevelSystemJacobianEquation188(k),
			new ThreeLevelSystemJacobianEquation189(k),
			new ThreeLevelSystemJacobianEquation190(k),
			new ThreeLevelSystemJacobianEquation191(k),
			new ThreeLevelSystemJacobianEquation192(k),
			new ThreeLevelSystemJacobianEquation193(k),
			new ThreeLevelSystemJacobianEquation194(k),
			new ThreeLevelSystemJacobianEquation195(k),
			new ThreeLevelSystemJacobianEquation196(k),
			new ThreeLevelSystemJacobianEquation197(k),
			new ThreeLevelSystemJacobianEquation198(k),
			new ThreeLevelSystemJacobianEquation199(k),
			new ThreeLevelSystemJacobianEquation200(k),
			new ThreeLevelSystemJacobianEquation201(k),
			new ThreeLevelSystemJacobianEquation202(k),
			new ThreeLevelSystemJacobianEquation203(k),
			new ThreeLevelSystemJacobianEquation204(k),
			new ThreeLevelSystemJacobianEquation205(k),
			new ThreeLevelSystemJacobianEquation206(k),
			new ThreeLevelSystemJacobianEquation207(k),
			new ThreeLevelSystemJacobianEquation208(k),
			new ThreeLevelSystemJacobianEquation209(k),
			new ThreeLevelSystemJacobianEquation210(k),
			new ThreeLevelSystemJacobianEquation211(k),
			new ThreeLevelSystemJacobianEquation212(k),
			new ThreeLevelSystemJacobianEquation213(k),
			new ThreeLevelSystemJacobianEquation214(k),
			new ThreeLevelSystemJacobianEquation215(k),
			new ThreeLevelSystemJacobianEquation216(k),
			new ThreeLevelSystemJacobianEquation217(k),
			new ThreeLevelSystemJacobianEquation218(k),
			new ThreeLevelSystemJacobianEquation219(k),
			new ThreeLevelSystemJacobianEquation220(k),
			new ThreeLevelSystemJacobianEquation221(k),
			new ThreeLevelSystemJacobianEquation222(k),
			new ThreeLevelSystemJacobianEquation223(k),
			new ThreeLevelSystemJacobianEquation224(k),
			new ThreeLevelSystemJacobianEquation225(k),
			new ThreeLevelSystemJacobianEquation226(k),
			new ThreeLevelSystemJacobianEquation227(k),
			new ThreeLevelSystemJacobianEquation228(k),
			new ThreeLevelSystemJacobianEquation229(k),
			new ThreeLevelSystemJacobianEquation230(k),
			new ThreeLevelSystemJacobianEquation231(k),
			new ThreeLevelSystemJacobianEquation232(k),
			new ThreeLevelSystemJacobianEquation233(k),
			new ThreeLevelSystemJacobianEquation234(k),
			new ThreeLevelSystemJacobianEquation235(k),
			new ThreeLevelSystemJacobianEquation236(k),
			new ThreeLevelSystemJacobianEquation237(k),
			new ThreeLevelSystemJacobianEquation238(k),
			new ThreeLevelSystemJacobianEquation239(k),
			new ThreeLevelSystemJacobianEquation240(k),
			new ThreeLevelSystemJacobianEquation241(k),
			new ThreeLevelSystemJacobianEquation242(k),
			new ThreeLevelSystemJacobianEquation243(k),
			new ThreeLevelSystemJacobianEquation244(k),
			new ThreeLevelSystemJacobianEquation245(k),
			new ThreeLevelSystemJacobianEquation246(k),
			new ThreeLevelSystemJacobianEquation247(k),
			new ThreeLevelSystemJacobianEquation248(k),
			new ThreeLevelSystemJacobianEquation249(k),
			new ThreeLevelSystemJacobianEquation250(k),
			new ThreeLevelSystemJacobianEquation251(k),
			new ThreeLevelSystemJacobianEquation252(k),
			new ThreeLevelSystemJacobianEquation253(k),
			new ThreeLevelSystemJacobianEquation254(k),
			new ThreeLevelSystemJacobianEquation255(k),
			new ThreeLevelSystemJacobianEquation256(k),
			new ThreeLevelSystemJacobianEquation257(k),
			new ThreeLevelSystemJacobianEquation258(k),
			new ThreeLevelSystemJacobianEquation259(k),
			new ThreeLevelSystemJacobianEquation260(k),
			new ThreeLevelSystemJacobianEquation261(k),
			new ThreeLevelSystemJacobianEquation262(k),
			new ThreeLevelSystemJacobianEquation263(k),
			new ThreeLevelSystemJacobianEquation264(k),
			new ThreeLevelSystemJacobianEquation265(k),
			new ThreeLevelSystemJacobianEquation266(k),
			new ThreeLevelSystemJacobianEquation267(k),
			new ThreeLevelSystemJacobianEquation268(k),
			new ThreeLevelSystemJacobianEquation269(k),
			new ThreeLevelSystemJacobianEquation270(k),
			new ThreeLevelSystemJacobianEquation271(k),
			new ThreeLevelSystemJacobianEquation272(k),
			new ThreeLevelSystemJacobianEquation273(k),
			new ThreeLevelSystemJacobianEquation274(k),
			new ThreeLevelSystemJacobianEquation275(k),
			new ThreeLevelSystemJacobianEquation276(k),
			new ThreeLevelSystemJacobianEquation277(k),
			new ThreeLevelSystemJacobianEquation278(k),
			new ThreeLevelSystemJacobianEquation279(k),
			new ThreeLevelSystemJacobianEquation280(k),
			new ThreeLevelSystemJacobianEquation281(k),
			new ThreeLevelSystemJacobianEquation282(k),
			new ThreeLevelSystemJacobianEquation283(k),
			new ThreeLevelSystemJacobianEquation284(k),
			new ThreeLevelSystemJacobianEquation285(k),
			new ThreeLevelSystemJacobianEquation286(k),
			new ThreeLevelSystemJacobianEquation287(k),
			new ThreeLevelSystemJacobianEquation288(k),
			new ThreeLevelSystemJacobianEquation289(k),
			new ThreeLevelSystemJacobianEquation290(k),
			new ThreeLevelSystemJacobianEquation291(k),
			new ThreeLevelSystemJacobianEquation292(k),
			new ThreeLevelSystemJacobianEquation293(k),
			new ThreeLevelSystemJacobianEquation294(k),
			new ThreeLevelSystemJacobianEquation295(k),
			new ThreeLevelSystemJacobianEquation296(k),
			new ThreeLevelSystemJacobianEquation297(k),
			new ThreeLevelSystemJacobianEquation298(k),
			new ThreeLevelSystemJacobianEquation299(k),
			new ThreeLevelSystemJacobianEquation300(k),
			new ThreeLevelSystemJacobianEquation301(k),
			new ThreeLevelSystemJacobianEquation302(k),
			new ThreeLevelSystemJacobianEquation303(k),
			new ThreeLevelSystemJacobianEquation304(k),
			new ThreeLevelSystemJacobianEquation305(k),
			new ThreeLevelSystemJacobianEquation306(k),
			new ThreeLevelSystemJacobianEquation307(k),
			new ThreeLevelSystemJacobianEquation308(k),
			new ThreeLevelSystemJacobianEquation309(k),
			new ThreeLevelSystemJacobianEquation310(k),
			new ThreeLevelSystemJacobianEquation311(k),
			new ThreeLevelSystemJacobianEquation312(k),
			new ThreeLevelSystemJacobianEquation313(k),
			new ThreeLevelSystemJacobianEquation314(k),
			new ThreeLevelSystemJacobianEquation315(k),
			new ThreeLevelSystemJacobianEquation316(k),
			new ThreeLevelSystemJacobianEquation317(k),
			new ThreeLevelSystemJacobianEquation318(k),
			new ThreeLevelSystemJacobianEquation319(k),
			new ThreeLevelSystemJacobianEquation320(k),
			new ThreeLevelSystemJacobianEquation321(k),
			new ThreeLevelSystemJacobianEquation322(k),
			new ThreeLevelSystemJacobianEquation323(k)
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
		return "three_level_system";
	}

	const static uint8_t dim = 18;

private:
	const cudaT k;
	std::vector < JacobianEquation* > jacobian_equations;
};

# endif //PROJECT_THREELEVELSYSTEMJACOBIAN_HPP
