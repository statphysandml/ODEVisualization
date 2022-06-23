#include <three_point_system/three_point_system_flow_equation.hpp>

std::string ThreePointSystemFlowEquations::model_ = "three_point_system";
size_t ThreePointSystemFlowEquations::dim_ = 3;
std::string ThreePointSystemFlowEquations::explicit_variable_ = "k";
std::vector<std::string> ThreePointSystemFlowEquations::explicit_functions_ = {"mh2", "Lam3", "gn"};


struct comp_func_three_point_system0
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;

	comp_func_three_point_system0(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (-2 * thrust::get<2>(t)) + (thrust::get<0>(t) * (const_expr0_ + ((210 + (-960 * thrust::get<1>(t)) + (1920 * (pow(thrust::get<1>(t), 2)))) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr1_) + ((-24 + (48 * thrust::get<1>(t))) * (pow((1 + thrust::get<2>(t)), -2)) * const_expr2_)));
	}
};


void ThreePointSystemFlowEquation0::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[0].end(), derivatives.end())), comp_func_three_point_system0(const_expr0_, const_expr1_, const_expr2_));
}


struct comp_func_three_point_system1
{
	const cudaT const_expr0_;
	const cudaT const_expr10_;
	const cudaT const_expr11_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;
	const cudaT const_expr6_;
	const cudaT const_expr7_;
	const cudaT const_expr8_;
	const cudaT const_expr9_;

	comp_func_three_point_system1(const cudaT const_expr0, const cudaT const_expr10, const cudaT const_expr11, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4, const cudaT const_expr5, const cudaT const_expr6, const cudaT const_expr7, const cudaT const_expr8, const cudaT const_expr9)
		: const_expr0_(const_expr0), const_expr10_(const_expr10), const_expr11_(const_expr11), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4), const_expr5_(const_expr5), const_expr6_(const_expr6), const_expr7_(const_expr7), const_expr8_(const_expr8), const_expr9_(const_expr9) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (thrust::get<0>(t) * (const_expr0_ + ((const_expr1_ + (-4 * thrust::get<1>(t) * (3 + (2 * (-2 + thrust::get<1>(t)) * thrust::get<1>(t))))) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr5_) + ((24 + (-96 * thrust::get<1>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr6_) + ((8 + (-24 * thrust::get<1>(t))) * (pow((1 + thrust::get<2>(t)), -2)) * const_expr7_))) + (-1 * thrust::get<1>(t) * (1 + (const_expr2_ * (pow(thrust::get<0>(t), -1)) * ((2 * thrust::get<0>(t)) + ((pow(thrust::get<0>(t), 2)) * (const_expr3_ + ((-299 + (1780 * thrust::get<1>(t)) + (-3640 * (pow(thrust::get<1>(t), 2))) + (2336 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<2>(t)), -5)) * const_expr8_) + ((1 + (-3 * thrust::get<1>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr9_) + ((const_expr4_ + (-124 * thrust::get<1>(t)) + (169 * (pow(thrust::get<1>(t), 2))) + (864 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr10_) + ((15 + (118 * thrust::get<1>(t)) + (-60 * thrust::get<1>(t) * (1 + thrust::get<1>(t)))) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr10_) + ((pow((1 + thrust::get<2>(t)), -2)) * const_expr11_)))))));
	}
};


void ThreePointSystemFlowEquation1::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[0].end(), derivatives.end())), comp_func_three_point_system1(const_expr0_, const_expr10_, const_expr11_, const_expr1_, const_expr2_, const_expr3_, const_expr4_, const_expr5_, const_expr6_, const_expr7_, const_expr8_, const_expr9_));
}


struct comp_func_three_point_system2
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;
	const cudaT const_expr2_;
	const cudaT const_expr3_;
	const cudaT const_expr4_;
	const cudaT const_expr5_;

	comp_func_three_point_system2(const cudaT const_expr0, const cudaT const_expr1, const cudaT const_expr2, const cudaT const_expr3, const cudaT const_expr4, const cudaT const_expr5)
		: const_expr0_(const_expr0), const_expr1_(const_expr1), const_expr2_(const_expr2), const_expr3_(const_expr3), const_expr4_(const_expr4), const_expr5_(const_expr5) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (2 * thrust::get<0>(t)) + ((pow(thrust::get<0>(t), 2)) * (const_expr0_ + ((-299 + (1780 * thrust::get<1>(t)) + (-3640 * (pow(thrust::get<1>(t), 2))) + (2336 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<2>(t)), -5)) * const_expr2_) + ((1 + (-3 * thrust::get<1>(t))) * thrust::get<1>(t) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr3_) + ((const_expr1_ + (-124 * thrust::get<1>(t)) + (169 * (pow(thrust::get<1>(t), 2))) + (864 * (pow(thrust::get<1>(t), 3)))) * (pow((1 + thrust::get<2>(t)), -4)) * const_expr4_) + ((15 + (118 * thrust::get<1>(t)) + (-60 * thrust::get<1>(t) * (1 + thrust::get<1>(t)))) * (pow((1 + thrust::get<2>(t)), -3)) * const_expr4_) + ((pow((1 + thrust::get<2>(t)), -2)) * const_expr5_)));
	}
};


void ThreePointSystemFlowEquation2::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[1].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[1].end(), variables[0].end(), derivatives.end())), comp_func_three_point_system2(const_expr0_, const_expr1_, const_expr2_, const_expr3_, const_expr4_, const_expr5_));
}

