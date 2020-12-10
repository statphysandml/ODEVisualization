#ifndef PROJECT_SMALLTHREELEVELSYSTEMJACOBIAN_HPP
#define PROJECT_SMALLTHREELEVELSYSTEMJACOBIAN_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/jacobian_equation.hpp"

struct SmallThreeLevelSystemJacobianEquation0 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation0(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation1 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation1(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation2 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation2(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation3 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation3(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation4 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation4(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation5 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation5(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation6 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation6(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation7 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation7(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation8 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation8(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation9 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation9(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation10 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation10(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation11 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation11(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation12 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation12(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation13 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation13(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation14 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation14(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation15 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation15(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation16 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation16(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation17 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation17(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation18 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation18(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation19 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation19(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation20 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation20(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation21 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation21(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation22 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation22(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation23 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation23(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation24 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation24(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation25 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation25(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation26 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation26(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation27 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation27(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation28 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation28(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation29 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation29(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation30 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation30(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation31 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation31(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation32 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation32(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation33 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation33(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation34 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation34(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation35 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation35(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation36 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation36(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation37 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation37(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation38 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation38(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation39 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation39(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation40 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation40(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation41 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation41(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation42 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation42(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation43 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation43(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation44 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation44(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation45 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation45(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation46 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation46(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation47 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation47(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation48 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation48(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation49 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation49(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation50 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation50(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation51 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation51(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation52 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation52(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation53 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation53(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation54 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation54(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation55 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation55(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation56 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation56(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation57 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation57(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation58 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation58(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation59 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation59(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation60 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation60(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation61 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation61(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation62 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation62(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation63 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation63(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation64 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation64(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation65 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation65(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation66 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation66(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation67 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation67(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation68 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation68(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation69 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation69(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation70 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation70(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation71 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation71(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation72 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation72(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation73 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation73(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation74 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation74(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation75 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation75(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation76 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation76(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation77 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation77(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation78 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation78(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation79 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation79(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation80 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation80(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation81 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation81(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation82 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation82(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation83 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation83(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation84 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation84(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation85 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation85(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation86 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation86(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation87 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation87(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation88 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation88(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation89 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation89(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation90 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation90(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation91 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation91(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation92 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation92(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation93 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation93(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation94 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation94(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation95 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation95(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation96 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation96(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation97 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation97(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation98 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation98(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation99 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation99(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation100 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation100(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation101 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation101(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation102 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation102(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation103 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation103(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation104 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation104(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation105 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation105(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation106 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation106(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation107 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation107(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation108 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation108(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation109 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation109(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation110 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation110(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation111 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation111(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation112 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation112(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation113 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation113(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation114 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation114(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation115 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation115(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation116 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation116(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation117 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation117(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation118 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation118(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation119 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation119(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation120 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation120(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation121 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation121(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation122 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation122(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation123 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation123(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation124 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation124(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation125 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation125(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation126 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation126(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation127 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation127(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation128 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation128(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation129 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation129(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation130 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation130(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation131 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation131(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation132 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation132(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation133 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation133(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation134 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation134(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation135 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation135(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation136 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation136(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation137 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation137(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation138 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation138(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation139 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation139(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation140 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation140(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation141 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation141(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation142 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation142(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

struct SmallThreeLevelSystemJacobianEquation143 : public JacobianEquation
{
	SmallThreeLevelSystemJacobianEquation143(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::fill(thrust::device, derivatives.begin(), derivatives.end(), 0);
	}

private:
	const cudaT k;

};

class SmallThreeLevelSystemJacobianEquations : public JacobianWrapper
{
public:
	SmallThreeLevelSystemJacobianEquations(const cudaT k_) : k(k_)
	{
		jacobian_equations = std::vector< JacobianEquation* > {
			new SmallThreeLevelSystemJacobianEquation0(k),
			new SmallThreeLevelSystemJacobianEquation1(k),
			new SmallThreeLevelSystemJacobianEquation2(k),
			new SmallThreeLevelSystemJacobianEquation3(k),
			new SmallThreeLevelSystemJacobianEquation4(k),
			new SmallThreeLevelSystemJacobianEquation5(k),
			new SmallThreeLevelSystemJacobianEquation6(k),
			new SmallThreeLevelSystemJacobianEquation7(k),
			new SmallThreeLevelSystemJacobianEquation8(k),
			new SmallThreeLevelSystemJacobianEquation9(k),
			new SmallThreeLevelSystemJacobianEquation10(k),
			new SmallThreeLevelSystemJacobianEquation11(k),
			new SmallThreeLevelSystemJacobianEquation12(k),
			new SmallThreeLevelSystemJacobianEquation13(k),
			new SmallThreeLevelSystemJacobianEquation14(k),
			new SmallThreeLevelSystemJacobianEquation15(k),
			new SmallThreeLevelSystemJacobianEquation16(k),
			new SmallThreeLevelSystemJacobianEquation17(k),
			new SmallThreeLevelSystemJacobianEquation18(k),
			new SmallThreeLevelSystemJacobianEquation19(k),
			new SmallThreeLevelSystemJacobianEquation20(k),
			new SmallThreeLevelSystemJacobianEquation21(k),
			new SmallThreeLevelSystemJacobianEquation22(k),
			new SmallThreeLevelSystemJacobianEquation23(k),
			new SmallThreeLevelSystemJacobianEquation24(k),
			new SmallThreeLevelSystemJacobianEquation25(k),
			new SmallThreeLevelSystemJacobianEquation26(k),
			new SmallThreeLevelSystemJacobianEquation27(k),
			new SmallThreeLevelSystemJacobianEquation28(k),
			new SmallThreeLevelSystemJacobianEquation29(k),
			new SmallThreeLevelSystemJacobianEquation30(k),
			new SmallThreeLevelSystemJacobianEquation31(k),
			new SmallThreeLevelSystemJacobianEquation32(k),
			new SmallThreeLevelSystemJacobianEquation33(k),
			new SmallThreeLevelSystemJacobianEquation34(k),
			new SmallThreeLevelSystemJacobianEquation35(k),
			new SmallThreeLevelSystemJacobianEquation36(k),
			new SmallThreeLevelSystemJacobianEquation37(k),
			new SmallThreeLevelSystemJacobianEquation38(k),
			new SmallThreeLevelSystemJacobianEquation39(k),
			new SmallThreeLevelSystemJacobianEquation40(k),
			new SmallThreeLevelSystemJacobianEquation41(k),
			new SmallThreeLevelSystemJacobianEquation42(k),
			new SmallThreeLevelSystemJacobianEquation43(k),
			new SmallThreeLevelSystemJacobianEquation44(k),
			new SmallThreeLevelSystemJacobianEquation45(k),
			new SmallThreeLevelSystemJacobianEquation46(k),
			new SmallThreeLevelSystemJacobianEquation47(k),
			new SmallThreeLevelSystemJacobianEquation48(k),
			new SmallThreeLevelSystemJacobianEquation49(k),
			new SmallThreeLevelSystemJacobianEquation50(k),
			new SmallThreeLevelSystemJacobianEquation51(k),
			new SmallThreeLevelSystemJacobianEquation52(k),
			new SmallThreeLevelSystemJacobianEquation53(k),
			new SmallThreeLevelSystemJacobianEquation54(k),
			new SmallThreeLevelSystemJacobianEquation55(k),
			new SmallThreeLevelSystemJacobianEquation56(k),
			new SmallThreeLevelSystemJacobianEquation57(k),
			new SmallThreeLevelSystemJacobianEquation58(k),
			new SmallThreeLevelSystemJacobianEquation59(k),
			new SmallThreeLevelSystemJacobianEquation60(k),
			new SmallThreeLevelSystemJacobianEquation61(k),
			new SmallThreeLevelSystemJacobianEquation62(k),
			new SmallThreeLevelSystemJacobianEquation63(k),
			new SmallThreeLevelSystemJacobianEquation64(k),
			new SmallThreeLevelSystemJacobianEquation65(k),
			new SmallThreeLevelSystemJacobianEquation66(k),
			new SmallThreeLevelSystemJacobianEquation67(k),
			new SmallThreeLevelSystemJacobianEquation68(k),
			new SmallThreeLevelSystemJacobianEquation69(k),
			new SmallThreeLevelSystemJacobianEquation70(k),
			new SmallThreeLevelSystemJacobianEquation71(k),
			new SmallThreeLevelSystemJacobianEquation72(k),
			new SmallThreeLevelSystemJacobianEquation73(k),
			new SmallThreeLevelSystemJacobianEquation74(k),
			new SmallThreeLevelSystemJacobianEquation75(k),
			new SmallThreeLevelSystemJacobianEquation76(k),
			new SmallThreeLevelSystemJacobianEquation77(k),
			new SmallThreeLevelSystemJacobianEquation78(k),
			new SmallThreeLevelSystemJacobianEquation79(k),
			new SmallThreeLevelSystemJacobianEquation80(k),
			new SmallThreeLevelSystemJacobianEquation81(k),
			new SmallThreeLevelSystemJacobianEquation82(k),
			new SmallThreeLevelSystemJacobianEquation83(k),
			new SmallThreeLevelSystemJacobianEquation84(k),
			new SmallThreeLevelSystemJacobianEquation85(k),
			new SmallThreeLevelSystemJacobianEquation86(k),
			new SmallThreeLevelSystemJacobianEquation87(k),
			new SmallThreeLevelSystemJacobianEquation88(k),
			new SmallThreeLevelSystemJacobianEquation89(k),
			new SmallThreeLevelSystemJacobianEquation90(k),
			new SmallThreeLevelSystemJacobianEquation91(k),
			new SmallThreeLevelSystemJacobianEquation92(k),
			new SmallThreeLevelSystemJacobianEquation93(k),
			new SmallThreeLevelSystemJacobianEquation94(k),
			new SmallThreeLevelSystemJacobianEquation95(k),
			new SmallThreeLevelSystemJacobianEquation96(k),
			new SmallThreeLevelSystemJacobianEquation97(k),
			new SmallThreeLevelSystemJacobianEquation98(k),
			new SmallThreeLevelSystemJacobianEquation99(k),
			new SmallThreeLevelSystemJacobianEquation100(k),
			new SmallThreeLevelSystemJacobianEquation101(k),
			new SmallThreeLevelSystemJacobianEquation102(k),
			new SmallThreeLevelSystemJacobianEquation103(k),
			new SmallThreeLevelSystemJacobianEquation104(k),
			new SmallThreeLevelSystemJacobianEquation105(k),
			new SmallThreeLevelSystemJacobianEquation106(k),
			new SmallThreeLevelSystemJacobianEquation107(k),
			new SmallThreeLevelSystemJacobianEquation108(k),
			new SmallThreeLevelSystemJacobianEquation109(k),
			new SmallThreeLevelSystemJacobianEquation110(k),
			new SmallThreeLevelSystemJacobianEquation111(k),
			new SmallThreeLevelSystemJacobianEquation112(k),
			new SmallThreeLevelSystemJacobianEquation113(k),
			new SmallThreeLevelSystemJacobianEquation114(k),
			new SmallThreeLevelSystemJacobianEquation115(k),
			new SmallThreeLevelSystemJacobianEquation116(k),
			new SmallThreeLevelSystemJacobianEquation117(k),
			new SmallThreeLevelSystemJacobianEquation118(k),
			new SmallThreeLevelSystemJacobianEquation119(k),
			new SmallThreeLevelSystemJacobianEquation120(k),
			new SmallThreeLevelSystemJacobianEquation121(k),
			new SmallThreeLevelSystemJacobianEquation122(k),
			new SmallThreeLevelSystemJacobianEquation123(k),
			new SmallThreeLevelSystemJacobianEquation124(k),
			new SmallThreeLevelSystemJacobianEquation125(k),
			new SmallThreeLevelSystemJacobianEquation126(k),
			new SmallThreeLevelSystemJacobianEquation127(k),
			new SmallThreeLevelSystemJacobianEquation128(k),
			new SmallThreeLevelSystemJacobianEquation129(k),
			new SmallThreeLevelSystemJacobianEquation130(k),
			new SmallThreeLevelSystemJacobianEquation131(k),
			new SmallThreeLevelSystemJacobianEquation132(k),
			new SmallThreeLevelSystemJacobianEquation133(k),
			new SmallThreeLevelSystemJacobianEquation134(k),
			new SmallThreeLevelSystemJacobianEquation135(k),
			new SmallThreeLevelSystemJacobianEquation136(k),
			new SmallThreeLevelSystemJacobianEquation137(k),
			new SmallThreeLevelSystemJacobianEquation138(k),
			new SmallThreeLevelSystemJacobianEquation139(k),
			new SmallThreeLevelSystemJacobianEquation140(k),
			new SmallThreeLevelSystemJacobianEquation141(k),
			new SmallThreeLevelSystemJacobianEquation142(k),
			new SmallThreeLevelSystemJacobianEquation143(k)
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
		return "small_three_level_system";
	}

	const static uint8_t dim = 12;

private:
	const cudaT k;
	std::vector < JacobianEquation* > jacobian_equations;
};

# endif //PROJECT_SMALLTHREELEVELSYSTEMJACOBIAN_HPP
