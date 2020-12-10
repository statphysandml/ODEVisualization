#ifndef PROJECT_SCALARTHEORYFLOWEQUATION_HPP
#define PROJECT_SCALARTHEORYFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"


struct comp_func_scalar_theory32
{
	const cudaT const_expr0;

	comp_func_scalar_theory32(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -216 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 2)) * ((15 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory31
{
	const cudaT const_expr0;

	comp_func_scalar_theory31(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 168 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<5>(t))) * ((13 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory30
{
	const cudaT const_expr0;

	comp_func_scalar_theory30(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -1512 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((13 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory29
{
	const cudaT const_expr0;

	comp_func_scalar_theory29(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 2016 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 3)) * ((13 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory28
{
	const cudaT const_expr0;

	comp_func_scalar_theory28(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 252 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory27
{
	const cudaT const_expr0;

	comp_func_scalar_theory27(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<7>(t) = -3024 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<6>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory26
{
	const cudaT const_expr0;

	comp_func_scalar_theory26(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -2268 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))), 2)) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory25
{
	const cudaT const_expr0;

	comp_func_scalar_theory25(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 18144 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))), 2)) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory24
{
	const cudaT const_expr0;

	comp_func_scalar_theory24(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -15120 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 4)) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory23
{
	comp_func_scalar_theory23()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<0>(t) + thrust::get<8>(t) + thrust::get<3>(t) + thrust::get<1>(t) + thrust::get<4>(t) + thrust::get<5>(t) + thrust::get<2>(t) + thrust::get<7>(t) + thrust::get<6>(t);
	}
};


struct comp_func_scalar_theory22
{
	const cudaT const_expr0;

	comp_func_scalar_theory22(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -1890 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))) * (pow(((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t))), 2));
	}
};


struct comp_func_scalar_theory21
{
	const cudaT const_expr0;

	comp_func_scalar_theory21(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -7560 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<5>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory20
{
	const cudaT const_expr0;

	comp_func_scalar_theory20(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 30240 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 2)) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<5>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory19
{
	const cudaT const_expr0;

	comp_func_scalar_theory19(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 45360 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * (pow(((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))), 2)) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory18
{
	const cudaT const_expr0;

	comp_func_scalar_theory18(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -151200 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 3)) * ((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory17
{
	const cudaT const_expr0;

	comp_func_scalar_theory17(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 90720 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -7)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 5)) * ((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory16
{
	const cudaT const_expr0;

	comp_func_scalar_theory16(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -1680 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -4)) * (pow(((7 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t))), 3));
	}
};


struct comp_func_scalar_theory15
{
	const cudaT const_expr0;

	comp_func_scalar_theory15(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 60480 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -5)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))) * (pow(((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t))), 2));
	}
};


struct comp_func_scalar_theory14
{
	const cudaT const_expr0;

	comp_func_scalar_theory14(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -100800 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 3)) * (pow(((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t))), 2));
	}
};


struct comp_func_scalar_theory13
{
	comp_func_scalar_theory13()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<0>(t) + thrust::get<3>(t) + thrust::get<4>(t) + thrust::get<6>(t) + thrust::get<5>(t) + thrust::get<8>(t) + thrust::get<1>(t) + thrust::get<7>(t) + thrust::get<2>(t);
	}
};


struct comp_func_scalar_theory12
{
	const cudaT const_expr0;

	comp_func_scalar_theory12(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 30240 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))), 3)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory11
{
	const cudaT const_expr0;

	comp_func_scalar_theory11(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -453600 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 2)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))), 2)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory10
{
	const cudaT const_expr0;

	comp_func_scalar_theory10(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 907200 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -7)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 4)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory9
{
	const cudaT const_expr0;

	comp_func_scalar_theory9(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -423360 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -8)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 6)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory8
{
	const cudaT const_expr0;

	comp_func_scalar_theory8(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -113400 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -6)) * ((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 4));
	}
};


struct comp_func_scalar_theory7
{
	const cudaT const_expr0;

	comp_func_scalar_theory7(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 907200 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -7)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 3)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 3));
	}
};


struct comp_func_scalar_theory6
{
	const cudaT const_expr0;

	comp_func_scalar_theory6(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -1905120 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -8)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 5)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 2));
	}
};


struct comp_func_scalar_theory5
{
	const cudaT const_expr0;

	comp_func_scalar_theory5(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 1451520 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -9)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 7)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory4
{
	const cudaT const_expr0;

	comp_func_scalar_theory4(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = -362880 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -10)) * (pow(((3 * thrust::get<1>(t)) + (-6 * thrust::get<2>(t))), 9));
	}
};


struct comp_func_scalar_theory3
{
	comp_func_scalar_theory3()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<4>(t) + thrust::get<5>(t) + thrust::get<1>(t) + thrust::get<6>(t) + thrust::get<7>(t) + thrust::get<2>(t) + thrust::get<8>(t) + thrust::get<0>(t) + thrust::get<3>(t);
	}
};


struct comp_func_scalar_theory2
{
	const cudaT const_expr0;

	comp_func_scalar_theory2(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 306 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -3)) * ((3 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t))) * thrust::get<2>(t);
	}
};


struct comp_func_scalar_theory1
{
	const cudaT const_expr0;

	comp_func_scalar_theory1(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 72 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((15 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory0
{
	const cudaT const_expr1;

	comp_func_scalar_theory0(const cudaT const_expr1_)
		: const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (thrust::get<4>(t) + thrust::get<1>(t) + thrust::get<3>(t) + thrust::get<0>(t) + thrust::get<2>(t)) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation0 : public FlowEquation
{
	ScalarTheoryFlowEquation0(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		dev_vec inter_med_vec4(derivatives.size());
		dev_vec inter_med_vec5(derivatives.size());
		dev_vec inter_med_vec6(derivatives.size());
		dev_vec inter_med_vec7(derivatives.size());
		dev_vec inter_med_vec8(derivatives.size());
		dev_vec inter_med_vec9(derivatives.size());
		dev_vec inter_med_vec10(derivatives.size());
		dev_vec inter_med_vec11(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec5.end())), comp_func_scalar_theory32(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[5].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[5].end(), inter_med_vec4.end())), comp_func_scalar_theory31(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), inter_med_vec9.end())), comp_func_scalar_theory30(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec6.end())), comp_func_scalar_theory29(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[7].begin(), variables[5].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[7].end(), variables[5].end(), inter_med_vec7.end())), comp_func_scalar_theory28(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec10.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), variables[5].end(), inter_med_vec10.end())), comp_func_scalar_theory27(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), inter_med_vec8.end())), comp_func_scalar_theory26(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), inter_med_vec3.end())), comp_func_scalar_theory25(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec11.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec11.end())), comp_func_scalar_theory24(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec11.begin(), inter_med_vec10.begin(), inter_med_vec9.begin(), inter_med_vec8.begin(), inter_med_vec7.begin(), inter_med_vec6.begin(), inter_med_vec5.begin(), inter_med_vec4.begin(), inter_med_vec3.begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec11.end(), inter_med_vec10.end(), inter_med_vec9.end(), inter_med_vec8.end(), inter_med_vec7.end(), inter_med_vec6.end(), inter_med_vec5.end(), inter_med_vec4.end(), inter_med_vec3.end(), inter_med_vec2.end())), comp_func_scalar_theory23());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[3].end(), variables[5].end(), inter_med_vec9.end())), comp_func_scalar_theory22(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec4.end())), comp_func_scalar_theory21(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec10.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec10.end())), comp_func_scalar_theory20(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec3.end())), comp_func_scalar_theory19(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec6.end())), comp_func_scalar_theory18(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[3].end(), variables[5].end(), inter_med_vec5.end())), comp_func_scalar_theory17(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[5].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[5].end(), inter_med_vec7.end())), comp_func_scalar_theory16(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec8.end())), comp_func_scalar_theory15(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec11.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec11.end())), comp_func_scalar_theory14(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec11.begin(), inter_med_vec10.begin(), inter_med_vec9.begin(), inter_med_vec8.begin(), inter_med_vec7.begin(), inter_med_vec6.begin(), inter_med_vec5.begin(), inter_med_vec4.begin(), inter_med_vec3.begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec11.end(), inter_med_vec10.end(), inter_med_vec9.end(), inter_med_vec8.end(), inter_med_vec7.end(), inter_med_vec6.end(), inter_med_vec5.end(), inter_med_vec4.end(), inter_med_vec3.end(), inter_med_vec0.end())), comp_func_scalar_theory13());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec6.end())), comp_func_scalar_theory12(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec4.end())), comp_func_scalar_theory11(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec11.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec11.end())), comp_func_scalar_theory10(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec5.end())), comp_func_scalar_theory9(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec10.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec10.end())), comp_func_scalar_theory8(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec9.end())), comp_func_scalar_theory7(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec1.end())), comp_func_scalar_theory6(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec8.end())), comp_func_scalar_theory5(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec7.end())), comp_func_scalar_theory4(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec4.begin(), inter_med_vec1.begin(), inter_med_vec5.begin(), inter_med_vec6.begin(), inter_med_vec7.begin(), inter_med_vec8.begin(), inter_med_vec9.begin(), inter_med_vec10.begin(), inter_med_vec11.begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec4.end(), inter_med_vec1.end(), inter_med_vec5.end(), inter_med_vec6.end(), inter_med_vec7.end(), inter_med_vec8.end(), inter_med_vec9.end(), inter_med_vec10.end(), inter_med_vec11.end(), inter_med_vec3.end())), comp_func_scalar_theory3());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[0].end(), variables[3].end(), inter_med_vec1.end())), comp_func_scalar_theory2(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[4].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[4].end(), variables[0].end(), variables[3].end(), inter_med_vec4.end())), comp_func_scalar_theory1(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), inter_med_vec4.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), inter_med_vec3.end(), inter_med_vec4.end(), derivatives.end())), comp_func_scalar_theory0(const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory33
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_scalar_theory33(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -2)) * ((3 * thrust::get<1>(t)) + (-6 * thrust::get<2>(t))) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation1 : public FlowEquation
{
	ScalarTheoryFlowEquation1(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((-1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_scalar_theory33(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory34
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_scalar_theory34(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = ((2 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -3)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 2))) + (-1 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -2)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))))) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation2 : public FlowEquation
{
	ScalarTheoryFlowEquation2(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), derivatives.end())), comp_func_scalar_theory34(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory35
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_scalar_theory35(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = ((-6 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 3))) + (6 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -3)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t)))) + (-1 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -2)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t))))) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation3 : public FlowEquation
{
	ScalarTheoryFlowEquation3(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), derivatives.end())), comp_func_scalar_theory35(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory36
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_scalar_theory36(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = ((24 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 4))) + (-36 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 2)) * ((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t)))) + (6 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * (pow(((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))), 2))) + (8 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<5>(t)))) + (-1 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -2)) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t))))) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation4 : public FlowEquation
{
	ScalarTheoryFlowEquation4(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), derivatives.end())), comp_func_scalar_theory36(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory37
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_scalar_theory37(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<7>(t) = ((-120 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))), 5))) + (240 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))), 3)) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t)))) + (-90 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))) * (pow(((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))), 2))) + (-60 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))), 2)) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<6>(t)))) + (20 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<6>(t)))) + (10 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))) * ((9 * thrust::get<6>(t)) + (-6 * thrust::get<1>(t)))) + (-1 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -2)) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t))))) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation5 : public FlowEquation
{
	ScalarTheoryFlowEquation5(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), variables[5].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), variables[5].end(), derivatives.end())), comp_func_scalar_theory37(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory50
{
	const cudaT const_expr0;

	comp_func_scalar_theory50(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 30 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory49
{
	const cudaT const_expr0;

	comp_func_scalar_theory49(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -90 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 2)) * ((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory48
{
	const cudaT const_expr0;

	comp_func_scalar_theory48(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 20 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -3)) * (pow(((7 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t))), 2));
	}
};


struct comp_func_scalar_theory47
{
	const cudaT const_expr0;

	comp_func_scalar_theory47(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -360 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory46
{
	const cudaT const_expr0;

	comp_func_scalar_theory46(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 480 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 3)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory45
{
	const cudaT const_expr0;

	comp_func_scalar_theory45(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -90 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -4)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 3));
	}
};


struct comp_func_scalar_theory44
{
	const cudaT const_expr0;

	comp_func_scalar_theory44(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 1080 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -5)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 2)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 2));
	}
};


struct comp_func_scalar_theory43
{
	const cudaT const_expr0;

	comp_func_scalar_theory43(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -1800 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -6)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 4)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory42
{
	const cudaT const_expr0;

	comp_func_scalar_theory42(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = 720 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -7)) * (pow(((3 * thrust::get<1>(t)) + (-6 * thrust::get<2>(t))), 6));
	}
};


struct comp_func_scalar_theory41
{
	comp_func_scalar_theory41()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<4>(t) + thrust::get<6>(t) + thrust::get<2>(t) + thrust::get<8>(t) + thrust::get<5>(t) + thrust::get<0>(t) + thrust::get<7>(t) + thrust::get<3>(t) + thrust::get<1>(t);
	}
};


struct comp_func_scalar_theory40
{
	const cudaT const_expr0;

	comp_func_scalar_theory40(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -1 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -2)) * ((13 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory39
{
	const cudaT const_expr0;

	comp_func_scalar_theory39(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 12 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory38
{
	const cudaT const_expr1;

	comp_func_scalar_theory38(const cudaT const_expr1_)
		: const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (thrust::get<0>(t) + thrust::get<2>(t) + thrust::get<1>(t)) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation6 : public FlowEquation
{
	ScalarTheoryFlowEquation6(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		dev_vec inter_med_vec4(derivatives.size());
		dev_vec inter_med_vec5(derivatives.size());
		dev_vec inter_med_vec6(derivatives.size());
		dev_vec inter_med_vec7(derivatives.size());
		dev_vec inter_med_vec8(derivatives.size());
		dev_vec inter_med_vec9(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec0.end())), comp_func_scalar_theory50(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[3].end(), variables[5].end(), inter_med_vec4.end())), comp_func_scalar_theory49(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[5].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[5].end(), inter_med_vec8.end())), comp_func_scalar_theory48(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec2.end())), comp_func_scalar_theory47(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec6.end())), comp_func_scalar_theory46(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec9.end())), comp_func_scalar_theory45(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec3.end())), comp_func_scalar_theory44(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec7.end())), comp_func_scalar_theory43(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec5.end())), comp_func_scalar_theory42(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec2.begin(), inter_med_vec0.begin(), inter_med_vec3.begin(), inter_med_vec4.begin(), inter_med_vec5.begin(), inter_med_vec6.begin(), inter_med_vec7.begin(), inter_med_vec8.begin(), inter_med_vec9.begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec2.end(), inter_med_vec0.end(), inter_med_vec3.end(), inter_med_vec4.end(), inter_med_vec5.end(), inter_med_vec6.end(), inter_med_vec7.end(), inter_med_vec8.end(), inter_med_vec9.end(), inter_med_vec1.end())), comp_func_scalar_theory41());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[7].begin(), variables[8].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[7].end(), variables[8].end(), inter_med_vec2.end())), comp_func_scalar_theory40(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec0.end())), comp_func_scalar_theory39(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), derivatives.end())), comp_func_scalar_theory38(const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory67
{
	const cudaT const_expr0;

	comp_func_scalar_theory67(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 840 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 3)) * ((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory66
{
	const cudaT const_expr0;

	comp_func_scalar_theory66(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -420 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))) * (pow(((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t))), 2));
	}
};


struct comp_func_scalar_theory65
{
	const cudaT const_expr0;

	comp_func_scalar_theory65(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -630 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))), 2)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory64
{
	const cudaT const_expr0;

	comp_func_scalar_theory64(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 5040 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 2)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory63
{
	const cudaT const_expr0;

	comp_func_scalar_theory63(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -4200 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 4)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory62
{
	const cudaT const_expr0;

	comp_func_scalar_theory62(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 2520 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -5)) * ((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 3));
	}
};


struct comp_func_scalar_theory61
{
	const cudaT const_expr0;

	comp_func_scalar_theory61(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -12600 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -6)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 3)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 2));
	}
};


struct comp_func_scalar_theory60
{
	const cudaT const_expr0;

	comp_func_scalar_theory60(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 15120 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -7)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 5)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory59
{
	const cudaT const_expr0;

	comp_func_scalar_theory59(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = -5040 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -8)) * (pow(((3 * thrust::get<1>(t)) + (-6 * thrust::get<2>(t))), 7));
	}
};


struct comp_func_scalar_theory58
{
	comp_func_scalar_theory58()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<3>(t) + thrust::get<5>(t) + thrust::get<1>(t) + thrust::get<4>(t) + thrust::get<2>(t) + thrust::get<8>(t) + thrust::get<7>(t) + thrust::get<6>(t) + thrust::get<0>(t);
	}
};


struct comp_func_scalar_theory57
{
	const cudaT const_expr0;

	comp_func_scalar_theory57(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -1 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -2)) * ((15 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t)));
	}
};


struct comp_func_scalar_theory56
{
	const cudaT const_expr0;

	comp_func_scalar_theory56(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 14 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * ((13 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory55
{
	const cudaT const_expr0;

	comp_func_scalar_theory55(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 42 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory54
{
	const cudaT const_expr0;

	comp_func_scalar_theory54(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -126 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 2)) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory53
{
	const cudaT const_expr0;

	comp_func_scalar_theory53(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 70 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<4>(t))) * ((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory52
{
	const cudaT const_expr0;

	comp_func_scalar_theory52(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -630 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * ((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory51
{
	const cudaT const_expr1;

	comp_func_scalar_theory51(const cudaT const_expr1_)
		: const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<7>(t) = (thrust::get<5>(t) + thrust::get<0>(t) + thrust::get<1>(t) + thrust::get<3>(t) + thrust::get<6>(t) + thrust::get<4>(t) + thrust::get<2>(t)) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation7 : public FlowEquation
{
	ScalarTheoryFlowEquation7(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		dev_vec inter_med_vec4(derivatives.size());
		dev_vec inter_med_vec5(derivatives.size());
		dev_vec inter_med_vec6(derivatives.size());
		dev_vec inter_med_vec7(derivatives.size());
		dev_vec inter_med_vec8(derivatives.size());
		dev_vec inter_med_vec9(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[3].end(), variables[5].end(), inter_med_vec6.end())), comp_func_scalar_theory67(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec7.end())), comp_func_scalar_theory66(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec8.end())), comp_func_scalar_theory65(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec9.end())), comp_func_scalar_theory64(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec4.end())), comp_func_scalar_theory63(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec1.end())), comp_func_scalar_theory62(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec5.end())), comp_func_scalar_theory61(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec0.end())), comp_func_scalar_theory60(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec3.end())), comp_func_scalar_theory59(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec6.begin(), inter_med_vec5.begin(), inter_med_vec4.begin(), inter_med_vec3.begin(), inter_med_vec1.begin(), inter_med_vec0.begin(), inter_med_vec7.begin(), inter_med_vec8.begin(), inter_med_vec9.begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec6.end(), inter_med_vec5.end(), inter_med_vec4.end(), inter_med_vec3.end(), inter_med_vec1.end(), inter_med_vec0.end(), inter_med_vec7.end(), inter_med_vec8.end(), inter_med_vec9.end(), inter_med_vec2.end())), comp_func_scalar_theory58());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[0].begin(), variables[8].begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[0].end(), variables[8].end(), inter_med_vec4.end())), comp_func_scalar_theory57(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec6.end())), comp_func_scalar_theory56(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), inter_med_vec3.end())), comp_func_scalar_theory55(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec1.end())), comp_func_scalar_theory54(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[5].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[5].end(), inter_med_vec0.end())), comp_func_scalar_theory53(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec5.end())), comp_func_scalar_theory52(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), inter_med_vec4.begin(), inter_med_vec5.begin(), inter_med_vec6.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), inter_med_vec3.end(), inter_med_vec4.end(), inter_med_vec5.end(), inter_med_vec6.end(), derivatives.end())), comp_func_scalar_theory51(const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory92
{
	const cudaT const_expr0;

	comp_func_scalar_theory92(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 112 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<5>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory91
{
	const cudaT const_expr0;

	comp_func_scalar_theory91(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -1008 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<5>(t))) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory90
{
	const cudaT const_expr0;

	comp_func_scalar_theory90(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 1344 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 3)) * ((11 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory89
{
	const cudaT const_expr0;

	comp_func_scalar_theory89(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 70 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -3)) * (pow(((9 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))), 2));
	}
};


struct comp_func_scalar_theory88
{
	const cudaT const_expr0;

	comp_func_scalar_theory88(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -1680 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * ((7 * thrust::get<3>(t)) + (-6 * thrust::get<5>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory87
{
	const cudaT const_expr0;

	comp_func_scalar_theory87(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = -1260 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))), 2)) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory86
{
	const cudaT const_expr0;

	comp_func_scalar_theory86(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 10080 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 2)) * ((5 * thrust::get<4>(t)) + (-6 * thrust::get<3>(t))) * ((9 * thrust::get<5>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory85
{
	const cudaT const_expr0;

	comp_func_scalar_theory85(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -8400 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 4)) * ((9 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory84
{
	const cudaT const_expr0;

	comp_func_scalar_theory84(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -1680 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -4)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))) * (pow(((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t))), 2));
	}
};


struct comp_func_scalar_theory83
{
	comp_func_scalar_theory83()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<1>(t) + thrust::get<7>(t) + thrust::get<5>(t) + thrust::get<6>(t) + thrust::get<3>(t) + thrust::get<4>(t) + thrust::get<0>(t) + thrust::get<2>(t) + thrust::get<8>(t);
	}
};


struct comp_func_scalar_theory82
{
	const cudaT const_expr0;

	comp_func_scalar_theory82(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 6720 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -5)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 2)) * (pow(((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t))), 2));
	}
};


struct comp_func_scalar_theory81
{
	const cudaT const_expr0;

	comp_func_scalar_theory81(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 20160 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -5)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))), 2)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory80
{
	const cudaT const_expr0;

	comp_func_scalar_theory80(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -67200 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -6)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 3)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<2>(t))) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory79
{
	const cudaT const_expr0;

	comp_func_scalar_theory79(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 40320 * (pow((const_expr0 + thrust::get<1>(t) + (-6 * thrust::get<0>(t))), -7)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<3>(t))), 5)) * ((7 * thrust::get<2>(t)) + (-6 * thrust::get<4>(t)));
	}
};


struct comp_func_scalar_theory78
{
	const cudaT const_expr0;

	comp_func_scalar_theory78(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 2520 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -5)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 4));
	}
};


struct comp_func_scalar_theory77
{
	const cudaT const_expr0;

	comp_func_scalar_theory77(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -50400 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -6)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 2)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 3));
	}
};


struct comp_func_scalar_theory76
{
	const cudaT const_expr0;

	comp_func_scalar_theory76(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 151200 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -7)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 4)) * (pow(((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t))), 2));
	}
};


struct comp_func_scalar_theory75
{
	const cudaT const_expr0;

	comp_func_scalar_theory75(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -141120 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<2>(t))), -8)) * (pow(((3 * thrust::get<2>(t)) + (-6 * thrust::get<3>(t))), 6)) * ((5 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory74
{
	const cudaT const_expr0;

	comp_func_scalar_theory74(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = 40320 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -9)) * (pow(((3 * thrust::get<1>(t)) + (-6 * thrust::get<2>(t))), 8));
	}
};


struct comp_func_scalar_theory73
{
	comp_func_scalar_theory73()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<9>(t) = thrust::get<5>(t) + thrust::get<4>(t) + thrust::get<1>(t) + thrust::get<0>(t) + thrust::get<7>(t) + thrust::get<8>(t) + thrust::get<2>(t) + thrust::get<6>(t) + thrust::get<3>(t);
	}
};


struct comp_func_scalar_theory72
{
	const cudaT const_expr0;

	comp_func_scalar_theory72(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = -17 * (pow((const_expr0 + thrust::get<0>(t) + (-6 * thrust::get<1>(t))), -2)) * thrust::get<2>(t);
	}
};


struct comp_func_scalar_theory71
{
	const cudaT const_expr0;

	comp_func_scalar_theory71(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = 16 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))) * ((15 * thrust::get<1>(t)) + (-6 * thrust::get<3>(t)));
	}
};


struct comp_func_scalar_theory70
{
	const cudaT const_expr0;

	comp_func_scalar_theory70(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = 56 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -3)) * ((5 * thrust::get<5>(t)) + (-6 * thrust::get<3>(t))) * ((13 * thrust::get<4>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory69
{
	const cudaT const_expr0;

	comp_func_scalar_theory69(const cudaT const_expr0_)
		: const_expr0(const_expr0_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = -168 * (pow((const_expr0 + thrust::get<2>(t) + (-6 * thrust::get<0>(t))), -4)) * (pow(((3 * thrust::get<0>(t)) + (-6 * thrust::get<4>(t))), 2)) * ((13 * thrust::get<3>(t)) + (-6 * thrust::get<1>(t)));
	}
};


struct comp_func_scalar_theory68
{
	const cudaT const_expr1;

	comp_func_scalar_theory68(const cudaT const_expr1_)
		: const_expr1(const_expr1_) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<6>(t) = (thrust::get<2>(t) + thrust::get<0>(t) + thrust::get<5>(t) + thrust::get<3>(t) + thrust::get<1>(t) + thrust::get<4>(t)) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation8 : public FlowEquation
{
	ScalarTheoryFlowEquation8(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		dev_vec inter_med_vec0(derivatives.size());
		dev_vec inter_med_vec1(derivatives.size());
		dev_vec inter_med_vec2(derivatives.size());
		dev_vec inter_med_vec3(derivatives.size());
		dev_vec inter_med_vec4(derivatives.size());
		dev_vec inter_med_vec5(derivatives.size());
		dev_vec inter_med_vec6(derivatives.size());
		dev_vec inter_med_vec7(derivatives.size());
		dev_vec inter_med_vec8(derivatives.size());
		dev_vec inter_med_vec9(derivatives.size());
		dev_vec inter_med_vec10(derivatives.size());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[5].begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[5].end(), inter_med_vec1.end())), comp_func_scalar_theory92(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), inter_med_vec8.end())), comp_func_scalar_theory91(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec10.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec10.end())), comp_func_scalar_theory90(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[6].begin(), variables[5].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[6].end(), variables[5].end(), inter_med_vec6.end())), comp_func_scalar_theory89(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec7.end())), comp_func_scalar_theory88(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec3.end())), comp_func_scalar_theory87(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec5.end())), comp_func_scalar_theory86(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[6].begin(), variables[1].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[6].end(), variables[1].end(), variables[3].end(), variables[5].end(), inter_med_vec2.end())), comp_func_scalar_theory85(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec9.end())), comp_func_scalar_theory84(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec10.begin(), inter_med_vec9.begin(), inter_med_vec8.begin(), inter_med_vec7.begin(), inter_med_vec6.begin(), inter_med_vec5.begin(), inter_med_vec3.begin(), inter_med_vec2.begin(), inter_med_vec1.begin(), inter_med_vec4.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec10.end(), inter_med_vec9.end(), inter_med_vec8.end(), inter_med_vec7.end(), inter_med_vec6.end(), inter_med_vec5.end(), inter_med_vec3.end(), inter_med_vec2.end(), inter_med_vec1.end(), inter_med_vec4.end())), comp_func_scalar_theory83());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec0.end())), comp_func_scalar_theory82(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec8.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec8.end())), comp_func_scalar_theory81(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec2.end())), comp_func_scalar_theory80(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[4].begin(), variables[3].begin(), variables[5].begin(), inter_med_vec10.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[4].end(), variables[3].end(), variables[5].end(), inter_med_vec10.end())), comp_func_scalar_theory79(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec9.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec9.end())), comp_func_scalar_theory78(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec5.end())), comp_func_scalar_theory77(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec3.end())), comp_func_scalar_theory76(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[4].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec6.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[4].end(), variables[2].end(), variables[3].end(), inter_med_vec6.end())), comp_func_scalar_theory75(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[3].begin(), inter_med_vec7.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[3].end(), inter_med_vec7.end())), comp_func_scalar_theory74(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec5.begin(), inter_med_vec3.begin(), inter_med_vec2.begin(), inter_med_vec0.begin(), inter_med_vec6.begin(), inter_med_vec7.begin(), inter_med_vec8.begin(), inter_med_vec9.begin(), inter_med_vec10.begin(), inter_med_vec1.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec5.end(), inter_med_vec3.end(), inter_med_vec2.end(), inter_med_vec0.end(), inter_med_vec6.end(), inter_med_vec7.end(), inter_med_vec8.end(), inter_med_vec9.end(), inter_med_vec10.end(), inter_med_vec1.end())), comp_func_scalar_theory73());
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[2].begin(), variables[0].begin(), inter_med_vec3.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[2].end(), variables[0].end(), inter_med_vec3.end())), comp_func_scalar_theory72(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[0].begin(), variables[3].begin(), inter_med_vec5.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[0].end(), variables[3].end(), inter_med_vec5.end())), comp_func_scalar_theory71(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[4].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec0.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[4].end(), variables[7].end(), variables[3].end(), inter_med_vec0.end())), comp_func_scalar_theory70(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[8].begin(), variables[1].begin(), variables[7].begin(), variables[3].begin(), inter_med_vec2.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[8].end(), variables[1].end(), variables[7].end(), variables[3].end(), inter_med_vec2.end())), comp_func_scalar_theory69(const_expr0));
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.begin(), inter_med_vec1.begin(), inter_med_vec2.begin(), inter_med_vec3.begin(), inter_med_vec4.begin(), inter_med_vec5.begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(inter_med_vec0.end(), inter_med_vec1.end(), inter_med_vec2.end(), inter_med_vec3.end(), inter_med_vec4.end(), inter_med_vec5.end(), derivatives.end())), comp_func_scalar_theory68(const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};


struct comp_func_scalar_theory93
{
	const cudaT const_expr0;
	const cudaT const_expr1;

	comp_func_scalar_theory93(const cudaT const_expr0_, const cudaT const_expr1_)
		: const_expr0(const_expr0_), const_expr1(const_expr1_) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (pow((const_expr0 + val1 + (-6 * val2)), -1)) * const_expr1;
	}
};

struct ScalarTheoryFlowEquation9 : public FlowEquation
{
	ScalarTheoryFlowEquation9(const cudaT k_) : k(k_),
		const_expr0(pow(k, 2)),
		const_expr1((1*1.0/32) * (pow(k, 5)) * (pow(M_PI, -2)))
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[1].begin(), variables[1].end(), variables[2].begin(), derivatives.begin(), comp_func_scalar_theory93(const_expr0, const_expr1));
	}

private:
	const cudaT k;

	const cudaT const_expr0;
	const cudaT const_expr1;
};

class ScalarTheoryFlowEquations : public FlowEquationsWrapper
{
public:
	ScalarTheoryFlowEquations(const cudaT k_) : k(k_)
	{
		flow_equations = std::vector< FlowEquation* > {
			new ScalarTheoryFlowEquation0(k),
			new ScalarTheoryFlowEquation1(k),
			new ScalarTheoryFlowEquation2(k),
			new ScalarTheoryFlowEquation3(k),
			new ScalarTheoryFlowEquation4(k),
			new ScalarTheoryFlowEquation5(k),
			new ScalarTheoryFlowEquation6(k),
			new ScalarTheoryFlowEquation7(k),
			new ScalarTheoryFlowEquation8(k),
			new ScalarTheoryFlowEquation9(k)
		};
	}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables, const int dim_index) override
	{
		(*flow_equations[dim_index])(derivatives, variables);
	}

	uint8_t get_dim() override
	{
		return dim;
	}

	static std::string name()
	{
		return "scalar_theory";
	}

	const static uint8_t dim = 10;

private:
	const cudaT k;
	std::vector < FlowEquation* > flow_equations;
};

# endif //PROJECT_SCALARTHEORYFLOWEQUATION_HPP
