#ifndef PROJECT_FOURPOINTSYSTEMJACOBIAN_HPP
#define PROJECT_FOURPOINTSYSTEMJACOBIAN_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/jacobian_equation.hpp"


struct comp_func_four_point_system36
{
	const cudaT const_expr0;

	comp_func_four_point_system36(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -2 + ((pow((1 + thrust::get<0>(t)), -4)) * ((thrust::get<4>(t) * (-7 + (32 * (1 + (-2 * thrust::get<2>(t))) * thrust::get<2>(t)))) + (-8 * thrust::get<1>(t) * (-1 + (2 * thrust::get<3>(t))) * (1 + thrust::get<0>(t)))) * const_expr0);
	}
};

struct FourPointSystemJacobianEquation0 : public JacobianEquation
{
	FourPointSystemJacobianEquation0(const cudaT k_) : k(k_),
		const_expr0((1*1.0/2) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system36(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system37
{
	const cudaT const_expr0;

	comp_func_four_point_system37(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<2>(t) * (-1 + (4 * thrust::get<0>(t))) * (pow((1 + thrust::get<1>(t)), -3)) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation1 : public JacobianEquation
{
	FourPointSystemJacobianEquation1(const cudaT k_) : k(k_),
		const_expr0((16*1.0/3) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system37(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system38
{
	const cudaT const_expr0;

	comp_func_four_point_system38(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return val1 * (pow((1 + val2), -2)) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation2 : public JacobianEquation
{
	FourPointSystemJacobianEquation2(const cudaT k_) : k(k_),
		const_expr0(4 * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[4].begin(), variables[4].end(), variables[0].begin(), derivatives.begin(), comp_func_four_point_system38(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system39
{
	const cudaT const_expr0;

	comp_func_four_point_system39(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (-12 + ((7 + (32 * val1 * (-1 + (2 * val1)))) * (pow((1 + val2), -3)))) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation3 : public JacobianEquation
{
	FourPointSystemJacobianEquation3(const cudaT k_) : k(k_),
		const_expr0((1*1.0/6) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[1].begin(), variables[1].end(), variables[0].begin(), derivatives.begin(), comp_func_four_point_system39(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system40
{
	const cudaT const_expr0;

	comp_func_four_point_system40(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return const_expr0 * (-2 + (4 * val1)) * (pow((1 + val2), -2));
	}
};

struct FourPointSystemJacobianEquation4 : public JacobianEquation
{
	FourPointSystemJacobianEquation4(const cudaT k_) : k(k_),
		const_expr0(pow(M_PI, -1))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[2].begin(), variables[2].end(), variables[0].begin(), derivatives.begin(), comp_func_four_point_system40(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system41
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;

	comp_func_four_point_system41(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_, const cudaT const_expr3_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_), const_expr3(const_expr3_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<4>(t), const_expr0)) * (pow((1 + thrust::get<0>(t)), -6)) * ((15 * (pow(thrust::get<1>(t), const_expr1)) * (-38 + (67 * thrust::get<2>(t))) * (pow((1 + thrust::get<0>(t)), 3))) + (-30 * (pow(thrust::get<4>(t), const_expr2)) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((114 * thrust::get<3>(t) * (1 + thrust::get<0>(t))) + (-1 * thrust::get<2>(t) * (15 + (590 * thrust::get<3>(t)) + (15 * thrust::get<0>(t)) + (574 * thrust::get<3>(t) * thrust::get<0>(t)))) + (12 * (pow(thrust::get<2>(t), 2)) * (5 + (9 * thrust::get<3>(t)) + (5 * (1 + thrust::get<3>(t)) * thrust::get<0>(t)))))) + ((pow(thrust::get<4>(t), const_expr1)) * ((627 * (1 + thrust::get<0>(t))) + (5 * thrust::get<2>(t) * (-885 + (-586 * thrust::get<0>(t)) + (4 * thrust::get<2>(t) * (425 + (-686 * thrust::get<2>(t)) + (2312 * (pow(thrust::get<2>(t), 2))) + (4 * (-5 + (8 * thrust::get<2>(t) * (7 + (54 * thrust::get<2>(t))))) * thrust::get<0>(t))))))))) * const_expr3;
	}
};

struct FourPointSystemJacobianEquation5 : public JacobianEquation
{
	FourPointSystemJacobianEquation5(const cudaT k_) : k(k_),
		const_expr0(-1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2(1*1.0/2),
		const_expr3((1*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system41(const_expr0, const_expr1, const_expr2, const_expr3));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
};


struct comp_func_four_point_system42
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;

	comp_func_four_point_system42(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_, const cudaT const_expr3_, const cudaT const_expr4_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_), const_expr3(const_expr3_), const_expr4(const_expr4_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<4>(t), const_expr0)) * (pow((1 + thrust::get<0>(t)), -5)) * ((-1005 * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 3))) + (-20 * (pow(thrust::get<4>(t), const_expr2)) * (1 + thrust::get<0>(t)) * (((pow((1 + thrust::get<0>(t)), 4)) * const_expr3) + (thrust::get<1>(t) * (15 + (586 * thrust::get<3>(t)) + (15 * thrust::get<0>(t)) + (574 * thrust::get<3>(t) * thrust::get<0>(t)) + (-24 * thrust::get<2>(t) * (5 + (8 * thrust::get<3>(t)) + (5 * (1 + thrust::get<3>(t)) * thrust::get<0>(t)))))))) + ((pow(thrust::get<4>(t), const_expr1)) * (2138 + (-1680 * (pow(thrust::get<2>(t), 2)) * (-9 + (4 * thrust::get<0>(t)))) + (80 * thrust::get<2>(t) * (-84 + (5 * thrust::get<0>(t)))) + (-256 * (pow(thrust::get<2>(t), 3)) * (343 + (270 * thrust::get<0>(t)))) + (5 * thrust::get<0>(t) * (368 + (15 * thrust::get<0>(t) * (10 + (thrust::get<0>(t) * (10 + (thrust::get<0>(t) * (5 + thrust::get<0>(t)))))))))))) * const_expr4;
	}
};

struct FourPointSystemJacobianEquation6 : public JacobianEquation
{
	FourPointSystemJacobianEquation6(const cudaT k_) : k(k_),
		const_expr0(-1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2(1*1.0/2),
		const_expr3(57 * M_PI),
		const_expr4((1*1.0/570) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system42(const_expr0, const_expr1, const_expr2, const_expr3, const_expr4));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
};


struct comp_func_four_point_system43
{
	const cudaT const_expr0;

	comp_func_four_point_system43(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<0>(t) * (pow((1 + thrust::get<2>(t)), -4)) * ((57 * (1 + thrust::get<2>(t))) + (thrust::get<1>(t) * (-293 + (-287 * thrust::get<2>(t)) + (6 * thrust::get<1>(t) * (8 + (5 * thrust::get<2>(t))))))) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation7 : public JacobianEquation
{
	FourPointSystemJacobianEquation7(const cudaT k_) : k(k_),
		const_expr0((4*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[0].end(), derivatives.end())), comp_func_four_point_system43(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system44
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system44(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (pow(thrust::get<3>(t), const_expr0)) * (pow((1 + thrust::get<2>(t)), -5)) * ((15 * (pow(thrust::get<0>(t), const_expr1)) * (-38 + (67 * thrust::get<1>(t))) * (pow((1 + thrust::get<2>(t)), 3))) + ((pow(thrust::get<3>(t), const_expr1)) * ((-1120 * (pow(thrust::get<1>(t), 3)) * (-9 + (4 * thrust::get<2>(t)))) + (80 * (pow(thrust::get<1>(t), 2)) * (-84 + (5 * thrust::get<2>(t)))) + (-128 * (pow(thrust::get<1>(t), 4)) * (343 + (270 * thrust::get<2>(t)))) + (57 * (1 + thrust::get<2>(t)) * (13 + (24 * thrust::get<2>(t) * (2 + thrust::get<2>(t)) * (2 + (thrust::get<2>(t) * (2 + thrust::get<2>(t))))))) + (2 * thrust::get<1>(t) * (2138 + (5 * thrust::get<2>(t) * (368 + (15 * thrust::get<2>(t) * (10 + (thrust::get<2>(t) * (10 + (thrust::get<2>(t) * (5 + thrust::get<2>(t)))))))))))))) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation8 : public JacobianEquation
{
	FourPointSystemJacobianEquation8(const cudaT k_) : k(k_),
		const_expr0(-3*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((1*1.0/1140) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[0].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system44(const_expr0, const_expr1, const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system45
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system45(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<4>(t), const_expr0)) * (pow((1 + thrust::get<0>(t)), -4)) * ((-9 * (pow(thrust::get<1>(t), const_expr1)) * (-38 + (67 * thrust::get<2>(t))) * (pow((1 + thrust::get<0>(t)), 2))) + (8 * (pow(thrust::get<4>(t), const_expr1)) * ((114 * thrust::get<3>(t) * (1 + thrust::get<0>(t))) + (-1 * thrust::get<2>(t) * (15 + (586 * thrust::get<3>(t)) + (15 * thrust::get<0>(t)) + (574 * thrust::get<3>(t) * thrust::get<0>(t)))) + (12 * (pow(thrust::get<2>(t), 2)) * (5 + (8 * thrust::get<3>(t)) + (5 * (1 + thrust::get<3>(t)) * thrust::get<0>(t))))))) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation9 : public JacobianEquation
{
	FourPointSystemJacobianEquation9(const cudaT k_) : k(k_),
		const_expr0(-1*1.0/2),
		const_expr1(1*1.0/2),
		const_expr2((1*1.0/228) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system45(const_expr0, const_expr1, const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system50
{
	comp_func_four_point_system50()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<2>(t), 2)) * (87158987 + (-250441606 * thrust::get<1>(t)) + (8 * thrust::get<0>(t) * ((3 * (-67448349 + (39498775 * thrust::get<1>(t)))) + (thrust::get<0>(t) * (1437041213 + (55818725 * thrust::get<1>(t)) + (2 * thrust::get<0>(t) * (-1892461901 + (1919359561 * thrust::get<0>(t)) + (59 * (-6820303 + (12239165 * thrust::get<0>(t))) * thrust::get<1>(t)))))))));
	}
};


struct comp_func_four_point_system49
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_four_point_system49(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -6 * (pow(thrust::get<3>(t), const_expr0)) * (pow(thrust::get<0>(t), const_expr1)) * (pow((1 + thrust::get<2>(t)), 2)) * (15318099 + (45656452 * thrust::get<2>(t)) + (4 * thrust::get<1>(t) * (3527699 + (-19686405 * thrust::get<2>(t)) + (3 * thrust::get<1>(t) * (-6453845 + (249079 * thrust::get<2>(t)))))));
	}
};


struct comp_func_four_point_system48
{
	comp_func_four_point_system48()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<0>(t), 2)) * (pow((1 + thrust::get<2>(t)), 2)) * (204371563 + ((pow(thrust::get<1>(t), 2)) * (33939672 + (-308793960 * thrust::get<2>(t)))) + (5 * thrust::get<2>(t) * (34874168 + (19698225 * thrust::get<2>(t)))) + (-2 * thrust::get<1>(t) * (225513035 + (45973707 * thrust::get<2>(t)))));
	}
};


struct comp_func_four_point_system47
{
	comp_func_four_point_system47()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -1 * thrust::get<4>(t) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * (19809651 + (-124579062 * thrust::get<3>(t)) + (4 * thrust::get<2>(t) * (-344680080 + (thrust::get<2>(t) * (1343716751 + (-2042942830 * thrust::get<3>(t)))) + (953515386 * thrust::get<3>(t)))) + (-692814214 * thrust::get<0>(t)) + (4 * ((261448352 * thrust::get<3>(t)) + (thrust::get<2>(t) * (308050240 + (571199341 * thrust::get<2>(t)) + (48 * (-13982433 + (7550336 * thrust::get<2>(t))) * thrust::get<3>(t))))) * thrust::get<0>(t)));
	}
};


struct comp_func_four_point_system46
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;

	comp_func_four_point_system46(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_, const cudaT const_expr3_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_), const_expr3(const_expr3_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = ((pow(thrust::get<5>(t), -1)) * (pow((1 + thrust::get<3>(t)), -6)) * ((-30 * (pow(thrust::get<8>(t), 2)) * (-6706343 + (16 * thrust::get<6>(t) * (2609977 + (2 * thrust::get<6>(t) * (-1876430 + (-3673410 * thrust::get<6>(t)) + (7755021 * (pow(thrust::get<6>(t), 2))))))))) + (32 * thrust::get<8>(t) * thrust::get<5>(t) * ((-187643 * (21 + (10 * thrust::get<6>(t) * (-9 + (16 * thrust::get<6>(t)))))) + (5 * (1688787 + (4 * thrust::get<6>(t) * (-3002288 + (3408219 * thrust::get<6>(t))))) * thrust::get<7>(t))) * (1 + thrust::get<3>(t))) + (60 * (pow(thrust::get<8>(t), const_expr0)) * (pow(thrust::get<5>(t), const_expr1)) * (1313501 + (16 * thrust::get<6>(t) * (-1490929 + (5213144 * thrust::get<6>(t))))) * (pow((1 + thrust::get<3>(t)), 2))) + (30 * (pow(thrust::get<5>(t), 2)) * (pow((1 + thrust::get<3>(t)), 2)) * (-4274929 + (8 * thrust::get<7>(t) * (-750572 + (7079359 * thrust::get<7>(t)))) + (-5963716 * thrust::get<3>(t)) + (16390004 * thrust::get<6>(t) * (1 + thrust::get<3>(t)))))) * const_expr2) + ((pow(thrust::get<5>(t), -1)) * thrust::get<7>(t) * (pow((1 + thrust::get<3>(t)), -7)) * (thrust::get<2>(t) + thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<4>(t)) * const_expr3);
	}
};

struct FourPointSystemJacobianEquation10 : public JacobianEquation
{
	FourPointSystemJacobianEquation10(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((1*1.0/66938580) * (pow(M_PI, -1))),
		const_expr3((-1*1.0/20447283) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec3.end())), comp_func_four_point_system50());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec1.end())), comp_func_four_point_system49(const_expr0, const_expr1));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[0].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[0].end(), inter_med_vec0.end())), comp_func_four_point_system48());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec2.end())), comp_func_four_point_system47());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), variables[0].begin(), inter_med_vec3.begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), variables[0].end(), inter_med_vec3.end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system46(const_expr0, const_expr1, const_expr2, const_expr3));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
};


struct comp_func_four_point_system54
{
	comp_func_four_point_system54()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 2 * thrust::get<4>(t) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((115103685719070 * (1 + thrust::get<0>(t))) + (20 * thrust::get<3>(t) * ((-10040787 * thrust::get<3>(t) * (-8730291 + (9321622 * thrust::get<0>(t)))) + (88 * (-571875192758 + (255640730419 * thrust::get<0>(t)))))) + (thrust::get<2>(t) * ((-409257549223360 * (1 + thrust::get<0>(t))) + (3 * thrust::get<3>(t) * ((2975048 * thrust::get<3>(t) * (-976169399 + (226510080 * thrust::get<0>(t)))) + (5 * (504191436061413 + (274363798502845 * thrust::get<0>(t)))))))));
	}
};


struct comp_func_four_point_system53
{
	comp_func_four_point_system53()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -24 * (pow(thrust::get<3>(t), 2)) * ((17788979447497 * (1 + thrust::get<2>(t))) + (1859405 * thrust::get<1>(t) * (-29774297 + (23699265 * thrust::get<2>(t)))) + (10 * thrust::get<0>(t) * ((-5115719365292 * (1 + thrust::get<2>(t))) + (371881 * thrust::get<1>(t) * (241367493 + (11163745 * thrust::get<2>(t)))))) + (8 * (pow(thrust::get<0>(t), 3)) * ((52856369685981 * (1 + thrust::get<2>(t))) + (1859405 * thrust::get<1>(t) * (343963618 + (144422147 * thrust::get<2>(t)))))) + (-6 * (pow(thrust::get<0>(t), 2)) * ((25037084615010 * (1 + thrust::get<2>(t))) + (371881 * thrust::get<1>(t) * (1644117897 + (402397877 * thrust::get<2>(t)))))));
	}
};


struct comp_func_four_point_system52
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_four_point_system52(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 40 * (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 2)) * ((-20323631463938 * (1 + thrust::get<0>(t))) + (142126174250336 * thrust::get<2>(t) * (1 + thrust::get<0>(t))) + (6693858 * thrust::get<2>(t) * thrust::get<3>(t) * (-4778114 + (249079 * thrust::get<0>(t)))) + (-3346929 * thrust::get<3>(t) * (758609 + (6562135 * thrust::get<0>(t)))));
	}
};


struct comp_func_four_point_system51
{
	const cudaT const_expr2;

	comp_func_four_point_system51(const cudaT const_expr2_)
		: const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = (pow(thrust::get<4>(t), -1)) * (pow((1 + thrust::get<1>(t)), -6)) * (thrust::get<2>(t) + thrust::get<5>(t) + thrust::get<3>(t) + thrust::get<0>(t)) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation11 : public JacobianEquation
{
	FourPointSystemJacobianEquation11(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((-1*1.0/114059340739845) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec0.end())), comp_func_four_point_system54());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[0].end(), variables[3].end(), inter_med_vec2.end())), comp_func_four_point_system53());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec3.end())), comp_func_four_point_system52(const_expr0, const_expr1));
		thrust::transform(variables[4].begin(), variables[4].end(), variables[0].begin(), inter_med_vec1.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 418913812698915 * (pow(val1, 2)) * (pow((1 + val2), 4)); });
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), variables[0].begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), variables[4].begin(), inter_med_vec3.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), variables[0].end(), inter_med_vec1.end(), inter_med_vec2.end(), variables[4].end(), inter_med_vec3.end(), derivatives.end())), comp_func_four_point_system51(const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system59
{
	comp_func_four_point_system59()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = -1115643 * (pow(thrust::get<2>(t), 2)) * (130409755 + (-89131340 * thrust::get<1>(t)) + (16 * ((5 * (pow(thrust::get<0>(t), 2)) * (241367493 + (11163745 * thrust::get<1>(t)))) + (5 * thrust::get<0>(t) * (-29774297 + (23699265 * thrust::get<1>(t)))) + (10 * (pow(thrust::get<0>(t), 4)) * (343963618 + (144422147 * thrust::get<1>(t)))) + (-2 * (pow(thrust::get<0>(t), 3)) * (1644117897 + (402397877 * thrust::get<1>(t)))) + (4289082 * (pow(thrust::get<1>(t), 2)) * (15 + (thrust::get<1>(t) * (20 + (thrust::get<1>(t) * (15 + (thrust::get<1>(t) * (6 + thrust::get<1>(t))))))))))));
	}
};


struct comp_func_four_point_system58
{
	const cudaT const_expr2;

	comp_func_four_point_system58(const cudaT const_expr2_)
		: const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = 5 * thrust::get<0>(t) * (pow((1 + thrust::get<2>(t)), 2)) * (((pow((1 + thrust::get<2>(t)), 4)) * const_expr2) + (thrust::get<0>(t) * ((160652592 * (pow(thrust::get<1>(t), 2)) * (718663 + (4288805 * thrust::get<2>(t)))) + (8 * thrust::get<1>(t) * (163674634714241 + (113599186087265 * thrust::get<2>(t)))) + (-9 * (22851632995047 + (thrust::get<2>(t) * (27096679526074 + (12208992685375 * thrust::get<2>(t)))))))));
	}
};


struct comp_func_four_point_system57
{
	comp_func_four_point_system57()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<4>(t) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((-160 * thrust::get<2>(t) * ((44 * (571875192758 + (-255640730419 * thrust::get<0>(t)))) + (10040787 * thrust::get<3>(t) * (-8730291 + (9321622 * thrust::get<0>(t)))))) + (15 * (-7470553046365 + (-113475063262391 * thrust::get<0>(t)) + (2975048 * thrust::get<3>(t) * (13686929 + (130724176 * thrust::get<0>(t)))))) + (6 * (pow(thrust::get<2>(t), 2)) * ((5950096 * thrust::get<3>(t) * (-976169399 + (226510080 * thrust::get<0>(t)))) + (5 * (504191436061413 + (274363798502845 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system56
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_four_point_system56(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 5578215 * (pow(thrust::get<3>(t), const_expr0)) * (pow(thrust::get<0>(t), const_expr1)) * (pow((1 + thrust::get<2>(t)), 2)) * (91610749 + (182625808 * thrust::get<2>(t)) + (48 * thrust::get<1>(t) * (-758609 + (-6562135 * thrust::get<2>(t)) + (thrust::get<1>(t) * (-4778114 + (249079 * thrust::get<2>(t)))))));
	}
};


struct comp_func_four_point_system55
{
	const cudaT const_expr3;

	comp_func_four_point_system55(const cudaT const_expr3_)
		: const_expr3(const_expr3_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = (pow(thrust::get<4>(t), -1)) * (pow((1 + thrust::get<1>(t)), -6)) * (thrust::get<0>(t) + thrust::get<5>(t) + thrust::get<2>(t) + thrust::get<3>(t)) * const_expr3;
	}
};

struct FourPointSystemJacobianEquation12 : public JacobianEquation
{
	FourPointSystemJacobianEquation12(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2(91247472591876 * M_PI),
		const_expr3((-1*1.0/228118681479690) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec2.end())), comp_func_four_point_system59());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[0].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[0].end(), inter_med_vec1.end())), comp_func_four_point_system58(const_expr2));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec3.end())), comp_func_four_point_system57());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec0.end())), comp_func_four_point_system56(const_expr0, const_expr1));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), variables[0].begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), variables[4].begin(), inter_med_vec3.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), variables[0].end(), inter_med_vec1.end(), inter_med_vec2.end(), variables[4].end(), inter_med_vec3.end(), derivatives.end())), comp_func_four_point_system55(const_expr3));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
};


struct comp_func_four_point_system62
{
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system62(const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 1115643 * thrust::get<3>(t) * ((-15 * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 2)) * (91610749 + (182625808 * thrust::get<0>(t)) + (48 * thrust::get<2>(t) * (-758609 + (-6562135 * thrust::get<0>(t)) + (thrust::get<2>(t) * (-4778114 + (249079 * thrust::get<0>(t)))))))) + (-6 * (pow(thrust::get<4>(t), const_expr2)) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((5 * (-61357561 + (54747716 * thrust::get<3>(t)) + (-346407107 * thrust::get<0>(t)) + (522896704 * thrust::get<3>(t) * thrust::get<0>(t)))) + (2 * (pow(thrust::get<2>(t), 2)) * (5946066345 + (-7809355192 * thrust::get<3>(t)) + (5 * (571199341 + (362416128 * thrust::get<3>(t))) * thrust::get<0>(t)))) + (-80 * thrust::get<2>(t) * (26766752 + (-38506280 * thrust::get<0>(t)) + (9 * thrust::get<3>(t) * (-8730291 + (9321622 * thrust::get<0>(t)))))))) + (12 * (pow(thrust::get<4>(t), const_expr1)) * (130409755 + (-89131340 * thrust::get<0>(t)) + (16 * ((5 * (pow(thrust::get<2>(t), 2)) * (241367493 + (11163745 * thrust::get<0>(t)))) + (5 * thrust::get<2>(t) * (-29774297 + (23699265 * thrust::get<0>(t)))) + (10 * (pow(thrust::get<2>(t), 4)) * (343963618 + (144422147 * thrust::get<0>(t)))) + (-2 * (pow(thrust::get<2>(t), 3)) * (1644117897 + (402397877 * thrust::get<0>(t)))) + (4289082 * (pow(thrust::get<0>(t), 2)) * (15 + (thrust::get<0>(t) * (20 + (thrust::get<0>(t) * (15 + (thrust::get<0>(t) * (6 + thrust::get<0>(t))))))))))))));
	}
};


struct comp_func_four_point_system61
{
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system61(const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -40894566 * (1 + thrust::get<0>(t)) * ((6 * (pow(thrust::get<4>(t), const_expr1)) * (6706343 + (-16 * thrust::get<2>(t) * (2609977 + (2 * thrust::get<2>(t) * (-1876430 + (-3673410 * thrust::get<2>(t)) + (7755021 * (pow(thrust::get<2>(t), 2))))))))) + (-4 * (pow(thrust::get<4>(t), const_expr2)) * thrust::get<1>(t) * (3940503 + (-68164380 * (pow(thrust::get<2>(t), 2)) * thrust::get<3>(t)) + (-8443935 * ((2 * thrust::get<2>(t)) + thrust::get<3>(t))) + (30022880 * thrust::get<2>(t) * (thrust::get<2>(t) + (2 * thrust::get<3>(t))))) * (1 + thrust::get<0>(t))) + (5 * (pow(thrust::get<1>(t), const_expr1)) * (1313501 + (16 * thrust::get<2>(t) * (-1490929 + (5213144 * thrust::get<2>(t))))) * (pow((1 + thrust::get<0>(t)), 2))) + (-104262880 * (pow(thrust::get<4>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 5))));
	}
};


struct comp_func_four_point_system60
{
	const cudaT const_expr0;
	const cudaT const_expr3;

	comp_func_four_point_system60(const cudaT const_expr0_, const cudaT const_expr3_)
		: const_expr0(const_expr0_), const_expr3(const_expr3_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<3>(t), -1)) * (pow((1 + thrust::get<1>(t)), -6)) * (thrust::get<0>(t) + thrust::get<2>(t)) * const_expr3;
	}
};

struct FourPointSystemJacobianEquation13 : public JacobianEquation
{
	FourPointSystemJacobianEquation13(const cudaT k_) : k(k_),
		const_expr0(-1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2(1*1.0/2),
		const_expr3((1*1.0/1368712088878140) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec1.end())), comp_func_four_point_system62(const_expr1, const_expr2));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec0.end())), comp_func_four_point_system61(const_expr1, const_expr2));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), variables[0].begin(), inter_med_vec1.begin(), variables[4].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), variables[0].end(), inter_med_vec1.end(), variables[4].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system60(const_expr0, const_expr3));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
};


struct comp_func_four_point_system66
{
	comp_func_four_point_system66()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 2 * (pow(thrust::get<3>(t), 2)) * ((240 * (pow(thrust::get<0>(t), 2)) * ((-5115719365292 * (1 + thrust::get<2>(t))) + (371881 * thrust::get<1>(t) * (241367493 + (11163745 * thrust::get<2>(t)))))) + (48 * thrust::get<0>(t) * ((17788979447497 * (1 + thrust::get<2>(t))) + (1859405 * thrust::get<1>(t) * (-29774297 + (23699265 * thrust::get<2>(t)))))) + (96 * (pow(thrust::get<0>(t), 4)) * ((52856369685981 * (1 + thrust::get<2>(t))) + (1859405 * thrust::get<1>(t) * (343963618 + (144422147 * thrust::get<2>(t)))))) + (-96 * (pow(thrust::get<0>(t), 3)) * ((25037084615010 * (1 + thrust::get<2>(t))) + (371881 * thrust::get<1>(t) * (1644117897 + (402397877 * thrust::get<2>(t)))))) + (6815761 * (1 + thrust::get<2>(t)) * (32012411 + (52131440 * thrust::get<2>(t) * (5 + (thrust::get<2>(t) * (10 + (thrust::get<2>(t) * (10 + (thrust::get<2>(t) * (5 + thrust::get<2>(t))))))))))) + (1115643 * thrust::get<1>(t) * (130409755 + (4 * thrust::get<2>(t) * (-22282835 + (17156328 * thrust::get<2>(t) * (15 + (thrust::get<2>(t) * (20 + (thrust::get<2>(t) * (15 + (thrust::get<2>(t) * (6 + thrust::get<2>(t))))))))))))));
	}
};


struct comp_func_four_point_system65
{
	comp_func_four_point_system65()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 10 * (pow(thrust::get<0>(t), 2)) * (pow((1 + thrust::get<3>(t)), 2)) * ((167565525079566 * thrust::get<1>(t) * (pow((1 + thrust::get<3>(t)), 2))) + (-20447283 * (1 + thrust::get<3>(t)) * (2418929 + (2981858 * thrust::get<3>(t)))) + (53550864 * (pow(thrust::get<2>(t), 3)) * (718663 + (4288805 * thrust::get<3>(t)))) + ((pow(thrust::get<2>(t), 2)) * (654698538856964 + (454396744349060 * thrust::get<3>(t)))) + (-9 * thrust::get<2>(t) * (22851632995047 + (thrust::get<3>(t) * (27096679526074 + (12208992685375 * thrust::get<3>(t)))))));
	}
};


struct comp_func_four_point_system64
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_four_point_system64(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 5 * (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 2)) * ((17905017778522 * (1 + thrust::get<0>(t))) + (1115643 * thrust::get<3>(t) * (91610749 + (182625808 * thrust::get<0>(t)))) + (16 * (pow(thrust::get<2>(t), 2)) * ((71063087125168 * (1 + thrust::get<0>(t))) + (3346929 * thrust::get<3>(t) * (-4778114 + (249079 * thrust::get<0>(t)))))) + (-16 * thrust::get<2>(t) * ((20323631463938 * (1 + thrust::get<0>(t))) + (3346929 * thrust::get<3>(t) * (758609 + (6562135 * thrust::get<0>(t)))))));
	}
};


struct comp_func_four_point_system63
{
	const cudaT const_expr2;

	comp_func_four_point_system63(const cudaT const_expr2_)
		: const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<3>(t), -2)) * (pow((1 + thrust::get<2>(t)), -6)) * (thrust::get<1>(t) + thrust::get<0>(t) + thrust::get<4>(t)) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation14 : public JacobianEquation
{
	FourPointSystemJacobianEquation14(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((-1*1.0/456237362959380) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[0].end(), variables[3].end(), inter_med_vec2.end())), comp_func_four_point_system66());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[0].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[2].end(), variables[0].end(), inter_med_vec0.end())), comp_func_four_point_system65());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec1.end())), comp_func_four_point_system64(const_expr0, const_expr1));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), variables[0].begin(), variables[4].begin(), inter_med_vec2.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), variables[0].end(), variables[4].end(), inter_med_vec2.end(), derivatives.end())), comp_func_four_point_system63(const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system67
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system67(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<0>(t)), -6)) * ((282 * (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 3))) + (12 * thrust::get<4>(t) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((-15 * (1 + thrust::get<0>(t))) + (-2 * thrust::get<3>(t) * (67 + (59 * thrust::get<0>(t)))) + (12 * thrust::get<2>(t) * (5 + (9 * thrust::get<3>(t)) + (5 * (1 + thrust::get<3>(t)) * thrust::get<0>(t)))))) + (-2 * (pow(thrust::get<4>(t), 2)) * (-201 + (98 * thrust::get<0>(t)) + (4 * thrust::get<2>(t) * (197 + (-248 * thrust::get<0>(t)) + (2 * thrust::get<2>(t) * (-286 + (169 * thrust::get<0>(t)) + (4 * thrust::get<2>(t) * (289 + (216 * thrust::get<0>(t))))))))))) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation15 : public JacobianEquation
{
	FourPointSystemJacobianEquation15(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((1*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system67(const_expr0, const_expr1, const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system68
{
	const cudaT const_expr0;

	comp_func_four_point_system68(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<4>(t) * (pow((1 + thrust::get<0>(t)), -5)) * ((-30 * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * (5 + (8 * thrust::get<3>(t)) + (5 * (1 + thrust::get<3>(t)) * thrust::get<0>(t)))) + (thrust::get<4>(t) * (135 + (-310 * thrust::get<0>(t)) + (thrust::get<2>(t) * (-975 + (845 * thrust::get<0>(t)) + (24 * thrust::get<2>(t) * (343 + (270 * thrust::get<0>(t))))))))) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation16 : public JacobianEquation
{
	FourPointSystemJacobianEquation16(const cudaT k_) : k(k_),
		const_expr0((8*1.0/285) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system68(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system69
{
	const cudaT const_expr0;

	comp_func_four_point_system69(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = thrust::get<3>(t) * thrust::get<0>(t) * (pow((1 + thrust::get<2>(t)), -4)) * (-65 + (-59 * thrust::get<2>(t)) + (6 * thrust::get<1>(t) * (8 + (5 * thrust::get<2>(t))))) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation17 : public JacobianEquation
{
	FourPointSystemJacobianEquation17(const cudaT k_) : k(k_),
		const_expr0((-8*1.0/57) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[0].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system69(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system70
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
	const cudaT const_expr5;
	const cudaT const_expr6;

	comp_func_four_point_system70(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_, const cudaT const_expr3_, const cudaT const_expr4_, const cudaT const_expr5_, const cudaT const_expr6_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_), const_expr3(const_expr3_), const_expr4(const_expr4_), const_expr5(const_expr5_), const_expr6(const_expr6_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 2 + (((-10 * thrust::get<4>(t)) + (const_expr0 * thrust::get<4>(t) * (-299 + (4 * thrust::get<2>(t) * (445 + (-910 * thrust::get<2>(t)) + (584 * (pow(thrust::get<2>(t), 2)))))) * (pow((1 + thrust::get<0>(t)), -5))) + (const_expr1 * thrust::get<4>(t) * (49 + (4 * thrust::get<2>(t) * (-124 + (thrust::get<2>(t) * (169 + (864 * thrust::get<2>(t))))))) * (pow((1 + thrust::get<0>(t)), -4))) + (16 * thrust::get<1>(t) * (1 + (-3 * thrust::get<2>(t))) * thrust::get<3>(t) * (pow((1 + thrust::get<0>(t)), -4))) + (const_expr2 * thrust::get<1>(t) * (15 + (118 * thrust::get<3>(t)) + (-60 * thrust::get<2>(t) * (1 + thrust::get<3>(t)))) * (pow((1 + thrust::get<0>(t)), -3))) + (const_expr3 * (pow(thrust::get<4>(t), const_expr4)) * (pow(thrust::get<1>(t), const_expr5)) * (pow((1 + thrust::get<0>(t)), -2)))) * const_expr6);
	}
};

struct FourPointSystemJacobianEquation18 : public JacobianEquation
{
	FourPointSystemJacobianEquation18(const cudaT k_) : k(k_),
		const_expr0(4*1.0/15),
		const_expr1(2*1.0/3),
		const_expr2(4*1.0/3),
		const_expr3(-47*1.0/2),
		const_expr4(-1*1.0/2),
		const_expr5(3*1.0/2),
		const_expr6((1*1.0/19) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system70(const_expr0, const_expr1, const_expr2, const_expr3, const_expr4, const_expr5, const_expr6));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
	const cudaT const_expr4;
	const cudaT const_expr5;
	const cudaT const_expr6;
};


struct comp_func_four_point_system71
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_four_point_system71(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<0>(t)), -4)) * ((120 * thrust::get<4>(t) * (1 + thrust::get<0>(t))) + (-423 * (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<1>(t), const_expr0)) * (pow((1 + thrust::get<0>(t)), 2))) + (16 * thrust::get<4>(t) * thrust::get<3>(t) * (65 + (59 * thrust::get<0>(t)))) + (-96 * thrust::get<4>(t) * thrust::get<2>(t) * (5 + (8 * thrust::get<3>(t)) + (5 * (1 + thrust::get<3>(t)) * thrust::get<0>(t))))) * const_expr1;
	}
};

struct FourPointSystemJacobianEquation19 : public JacobianEquation
{
	FourPointSystemJacobianEquation19(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1((1*1.0/114) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system71(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_four_point_system76
{
	comp_func_four_point_system76()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<2>(t), 2)) * (87158987 + (-250441606 * thrust::get<1>(t)) + (8 * thrust::get<0>(t) * ((3 * (-67448349 + (39498775 * thrust::get<1>(t)))) + (thrust::get<0>(t) * (1437041213 + (55818725 * thrust::get<1>(t)) + (2 * thrust::get<0>(t) * (-1892461901 + (1919359561 * thrust::get<0>(t)) + (59 * (-6820303 + (12239165 * thrust::get<0>(t))) * thrust::get<1>(t)))))))));
	}
};


struct comp_func_four_point_system75
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_four_point_system75(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -6 * (pow(thrust::get<3>(t), const_expr0)) * (pow(thrust::get<0>(t), const_expr1)) * (pow((1 + thrust::get<2>(t)), 2)) * (15318099 + (45656452 * thrust::get<2>(t)) + (4 * thrust::get<1>(t) * (3527699 + (-19686405 * thrust::get<2>(t)) + (3 * thrust::get<1>(t) * (-6453845 + (249079 * thrust::get<2>(t)))))));
	}
};


struct comp_func_four_point_system74
{
	comp_func_four_point_system74()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<0>(t), 2)) * (pow((1 + thrust::get<2>(t)), 2)) * (204371563 + ((pow(thrust::get<1>(t), 2)) * (33939672 + (-308793960 * thrust::get<2>(t)))) + (5 * thrust::get<2>(t) * (34874168 + (19698225 * thrust::get<2>(t)))) + (-2 * thrust::get<1>(t) * (225513035 + (45973707 * thrust::get<2>(t)))));
	}
};


struct comp_func_four_point_system73
{
	comp_func_four_point_system73()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -1 * thrust::get<4>(t) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * (19809651 + (-124579062 * thrust::get<3>(t)) + (4 * thrust::get<2>(t) * (-344680080 + (thrust::get<2>(t) * (1343716751 + (-2042942830 * thrust::get<3>(t)))) + (953515386 * thrust::get<3>(t)))) + (-692814214 * thrust::get<0>(t)) + (4 * ((261448352 * thrust::get<3>(t)) + (thrust::get<2>(t) * (308050240 + (571199341 * thrust::get<2>(t)) + (48 * (-13982433 + (7550336 * thrust::get<2>(t))) * thrust::get<3>(t))))) * thrust::get<0>(t)));
	}
};


struct comp_func_four_point_system72
{
	const cudaT const_expr2;

	comp_func_four_point_system72(const cudaT const_expr2_)
		: const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<2>(t)), -7)) * (thrust::get<3>(t) + thrust::get<4>(t) + thrust::get<1>(t) + thrust::get<0>(t)) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation20 : public JacobianEquation
{
	FourPointSystemJacobianEquation20(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((1*1.0/20447283) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec0.end())), comp_func_four_point_system76());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec1.end())), comp_func_four_point_system75(const_expr0, const_expr1));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[4].begin(), variables[2].begin(), variables[0].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[4].end(), variables[2].end(), variables[0].end(), inter_med_vec3.end())), comp_func_four_point_system74());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec2.end())), comp_func_four_point_system73());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), variables[0].begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), variables[0].end(), inter_med_vec2.end(), inter_med_vec3.end(), derivatives.end())), comp_func_four_point_system72(const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system77
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system77(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<0>(t)), -6)) * ((120 * (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 2)) * (-758609 + (-6562135 * thrust::get<0>(t)) + (thrust::get<2>(t) * (-9556228 + (498158 * thrust::get<0>(t)))))) + (2 * thrust::get<4>(t) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((thrust::get<2>(t) * (5946066345 + (-7809355192 * thrust::get<3>(t)) + (5 * (571199341 + (362416128 * thrust::get<3>(t))) * thrust::get<0>(t)))) + (-20 * (26766752 + (-38506280 * thrust::get<0>(t)) + (9 * thrust::get<3>(t) * (-8730291 + (9321622 * thrust::get<0>(t)))))))) + (-8 * (pow(thrust::get<4>(t), 2)) * ((5 * (-29774297 + (23699265 * thrust::get<0>(t)))) + (2 * thrust::get<2>(t) * (1206837465 + (55818725 * thrust::get<0>(t)) + (thrust::get<2>(t) * (-4932353691 + (6879272360 * thrust::get<2>(t)) + (59 * (-20460909 + (48956660 * thrust::get<2>(t))) * thrust::get<0>(t))))))))) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation21 : public JacobianEquation
{
	FourPointSystemJacobianEquation21(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2((1*1.0/102236415) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system77(const_expr0, const_expr1, const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};


struct comp_func_four_point_system78
{
	const cudaT const_expr0;

	comp_func_four_point_system78(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = thrust::get<1>(t) * (pow((1 + thrust::get<0>(t)), -5)) * ((5 * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * (60209401 + (15324569 * thrust::get<0>(t)) + (24 * thrust::get<3>(t) * (718663 + (4288805 * thrust::get<0>(t)))))) + (thrust::get<4>(t) * (68434645 + (653620880 * thrust::get<0>(t)) + (4 * thrust::get<2>(t) * (392863095 + (-976169399 * thrust::get<2>(t)) + (30 * (-13982433 + (7550336 * thrust::get<2>(t))) * thrust::get<0>(t))))))) * const_expr0;
	}
};

struct FourPointSystemJacobianEquation22 : public JacobianEquation
{
	FourPointSystemJacobianEquation22(const cudaT k_) : k(k_),
		const_expr0((2*1.0/102236415) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system78(const_expr0));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
};


struct comp_func_four_point_system79
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;

	comp_func_four_point_system79(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_, const cudaT const_expr3_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_), const_expr3(const_expr3_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow(thrust::get<4>(t), const_expr0)) * (pow((1 + thrust::get<0>(t)), -6)) * ((5 * (pow(thrust::get<1>(t), const_expr1)) * (pow((1 + thrust::get<0>(t)), 2)) * (91610749 + (182625808 * thrust::get<0>(t)) + (48 * thrust::get<2>(t) * (-758609 + (-6562135 * thrust::get<0>(t)) + (thrust::get<2>(t) * (-4778114 + (249079 * thrust::get<0>(t)))))))) + (2 * (pow(thrust::get<4>(t), const_expr2)) * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * ((5 * (-61357561 + (54747716 * thrust::get<3>(t)) + (-346407107 * thrust::get<0>(t)) + (522896704 * thrust::get<3>(t) * thrust::get<0>(t)))) + (2 * (pow(thrust::get<2>(t), 2)) * (5946066345 + (-7809355192 * thrust::get<3>(t)) + (5 * (571199341 + (362416128 * thrust::get<3>(t))) * thrust::get<0>(t)))) + (-80 * thrust::get<2>(t) * (26766752 + (-38506280 * thrust::get<0>(t)) + (9 * thrust::get<3>(t) * (-8730291 + (9321622 * thrust::get<0>(t)))))))) + (-4 * (pow(thrust::get<4>(t), const_expr1)) * (130409755 + (-89131340 * thrust::get<0>(t)) + (16 * ((5 * (pow(thrust::get<2>(t), 2)) * (241367493 + (11163745 * thrust::get<0>(t)))) + (5 * thrust::get<2>(t) * (-29774297 + (23699265 * thrust::get<0>(t)))) + (10 * (pow(thrust::get<2>(t), 4)) * (343963618 + (144422147 * thrust::get<0>(t)))) + (-2 * (pow(thrust::get<2>(t), 3)) * (1644117897 + (402397877 * thrust::get<0>(t)))) + (4289082 * (pow(thrust::get<0>(t), 2)) * (15 + (thrust::get<0>(t) * (20 + (thrust::get<0>(t) * (15 + (thrust::get<0>(t) * (6 + thrust::get<0>(t)))))))))))))) * const_expr3;
	}
};

struct FourPointSystemJacobianEquation23 : public JacobianEquation
{
	FourPointSystemJacobianEquation23(const cudaT k_) : k(k_),
		const_expr0(-1*1.0/2),
		const_expr1(3*1.0/2),
		const_expr2(1*1.0/2),
		const_expr3((1*1.0/408945660) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system79(const_expr0, const_expr1, const_expr2, const_expr3));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
	const cudaT const_expr3;
};


struct comp_func_four_point_system80
{
	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;

	comp_func_four_point_system80(const cudaT const_expr0_, const cudaT const_expr1_, const cudaT const_expr2_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_), const_expr2(const_expr2_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (pow((1 + thrust::get<0>(t)), -5)) * (((pow((1 + thrust::get<0>(t)), 5)) * const_expr1) + (20 * thrust::get<1>(t) * (1 + thrust::get<0>(t)) * (-442988327 + (144 * (pow(thrust::get<3>(t), 2)) * (718663 + (4288805 * thrust::get<0>(t)))) + (12 * thrust::get<3>(t) * (60209401 + (15324569 * thrust::get<0>(t)))) + (-5 * thrust::get<0>(t) * (109144786 + (59094675 * thrust::get<0>(t)))))) + (45 * (pow(thrust::get<4>(t), const_expr0)) * (pow(thrust::get<1>(t), const_expr0)) * (1 + thrust::get<0>(t)) * (91610749 + (182625808 * thrust::get<0>(t)) + (48 * thrust::get<2>(t) * (-758609 + (-6562135 * thrust::get<0>(t)) + (thrust::get<2>(t) * (-4778114 + (249079 * thrust::get<0>(t)))))))) + (6 * thrust::get<4>(t) * ((5 * (-61357561 + (54747716 * thrust::get<3>(t)) + (-346407107 * thrust::get<0>(t)) + (522896704 * thrust::get<3>(t) * thrust::get<0>(t)))) + (2 * (pow(thrust::get<2>(t), 2)) * (5946066345 + (-7809355192 * thrust::get<3>(t)) + (5 * (571199341 + (362416128 * thrust::get<3>(t))) * thrust::get<0>(t)))) + (-80 * thrust::get<2>(t) * (26766752 + (-38506280 * thrust::get<0>(t)) + (9 * thrust::get<3>(t) * (-8730291 + (9321622 * thrust::get<0>(t))))))))) * const_expr2;
	}
};

struct FourPointSystemJacobianEquation24 : public JacobianEquation
{
	FourPointSystemJacobianEquation24(const cudaT k_) : k(k_),
		const_expr0(1*1.0/2),
		const_expr1(2453673960 * M_PI),
		const_expr2((1*1.0/1226836980) * (pow(M_PI, -1)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[0].begin(), variables[4].begin(), variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[0].end(), variables[4].end(), variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_four_point_system80(const_expr0, const_expr1, const_expr2));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
	const cudaT const_expr2;
};

class FourPointSystemJacobianEquations : public JacobianWrapper
{
public:
	FourPointSystemJacobianEquations(const cudaT k_) : k(k_)
	{
		jacobian_equations = std::vector< JacobianEquation* > {
			new FourPointSystemJacobianEquation0(k),
			new FourPointSystemJacobianEquation1(k),
			new FourPointSystemJacobianEquation2(k),
			new FourPointSystemJacobianEquation3(k),
			new FourPointSystemJacobianEquation4(k),
			new FourPointSystemJacobianEquation5(k),
			new FourPointSystemJacobianEquation6(k),
			new FourPointSystemJacobianEquation7(k),
			new FourPointSystemJacobianEquation8(k),
			new FourPointSystemJacobianEquation9(k),
			new FourPointSystemJacobianEquation10(k),
			new FourPointSystemJacobianEquation11(k),
			new FourPointSystemJacobianEquation12(k),
			new FourPointSystemJacobianEquation13(k),
			new FourPointSystemJacobianEquation14(k),
			new FourPointSystemJacobianEquation15(k),
			new FourPointSystemJacobianEquation16(k),
			new FourPointSystemJacobianEquation17(k),
			new FourPointSystemJacobianEquation18(k),
			new FourPointSystemJacobianEquation19(k),
			new FourPointSystemJacobianEquation20(k),
			new FourPointSystemJacobianEquation21(k),
			new FourPointSystemJacobianEquation22(k),
			new FourPointSystemJacobianEquation23(k),
			new FourPointSystemJacobianEquation24(k)
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
		return "four_point_system";
	}

	const static uint8_t dim = 5;

private:
	const cudaT k;
	std::vector < JacobianEquation* > jacobian_equations;
};

# endif //PROJECT_FOURPOINTSYSTEMJACOBIAN_HPP
