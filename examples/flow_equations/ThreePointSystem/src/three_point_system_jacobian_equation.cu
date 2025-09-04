#include <three_point_system/three_point_system_jacobian_equation.hpp>

std::string ThreePointSystemJacobianEquations::model_ = "three_point_system";
size_t ThreePointSystemJacobianEquations::dim_ = 3;


struct comp_func_three_point_system3
{
	const cudaT const_expr0_;

	comp_func_three_point_system3(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = -2 + (thrust::get<0>(t) * (pow((1 + thrust::get<1>(t)), -4)) * (1 + (8 * thrust::get<1>(t)) + (-16 * thrust::get<2>(t) * (-1 + (4 * thrust::get<2>(t)) + thrust::get<1>(t)))) * const_expr0_);
	}
};


void ThreePointSystemJacobianEquation0::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system3(const_expr0_));
}


struct comp_func_three_point_system4
{
	const cudaT const_expr0_;

	comp_func_three_point_system4(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<0>(t) * (pow((1 + thrust::get<1>(t)), -3)) * (-1 + (16 * thrust::get<2>(t)) + (3 * thrust::get<1>(t))) * const_expr0_;
	}
};


void ThreePointSystemJacobianEquation1::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system4(const_expr0_));
}


struct comp_func_three_point_system5
{
	const cudaT const_expr0_;

	comp_func_three_point_system5(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (pow((1 + val1), -3)) * (-17 + (64 * (pow(val2, 2))) + (8 * val2 * (-1 + (3 * val1))) + (-12 * val1 * (4 + (val1 * (3 + val1))))) * const_expr0_;
	}
};


void ThreePointSystemJacobianEquation2::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), comp_func_three_point_system5(const_expr0_));
}


struct comp_func_three_point_system6
{
	const cudaT const_expr0_;

	comp_func_three_point_system6(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = thrust::get<0>(t) * (pow((1 + thrust::get<1>(t)), -6)) * ((160 * (pow(thrust::get<2>(t), 4)) * (289 + (216 * thrust::get<1>(t)))) + (-57 * (1 + thrust::get<1>(t)) * (-1 + (10 * thrust::get<1>(t) * (2 + thrust::get<1>(t))))) + (-40 * (pow(thrust::get<2>(t), 3)) * (424 + (thrust::get<1>(t) * (14 + (45 * thrust::get<1>(t)))))) + (20 * (pow(thrust::get<2>(t), 2)) * (1220 + (thrust::get<1>(t) * (1546 + (771 * thrust::get<1>(t)))))) + (5 * thrust::get<2>(t) * (-1278 + (thrust::get<1>(t) * (-1171 + (3 * thrust::get<1>(t) * (3 + (67 * thrust::get<1>(t))))))))) * const_expr0_;
	}
};


void ThreePointSystemJacobianEquation3::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system6(const_expr0_));
}


struct comp_func_three_point_system7
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_three_point_system7(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow((1 + thrust::get<1>(t)), -5)) * (((pow((1 + thrust::get<1>(t)), 5)) * const_expr0_) + (thrust::get<0>(t) * (3113 + (-256 * (pow(thrust::get<2>(t), 3)) * (343 + (270 * thrust::get<1>(t)))) + (240 * (pow(thrust::get<2>(t), 2)) * (87 + (thrust::get<1>(t) * (11 + (15 * thrust::get<1>(t)))))) + (-80 * thrust::get<2>(t) * (347 + (thrust::get<1>(t) * (515 + (257 * thrust::get<1>(t)))))) + (5 * thrust::get<1>(t) * (557 + (3 * thrust::get<1>(t) * (-19 + (thrust::get<1>(t) * (-17 + (5 * thrust::get<1>(t) * (5 + thrust::get<1>(t)))))))))))) * const_expr1_;
	}
};


void ThreePointSystemJacobianEquation4::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system7(const_expr0_, const_expr1_));
}


struct comp_func_three_point_system8
{
	const cudaT const_expr0_;

	comp_func_three_point_system8(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	__host__ __device__
	cudaT operator()(const cudaT &val1, const cudaT &val2)
	{
		return (pow((1 + val1), -5)) * ((-128 * (pow(val2, 4)) * (343 + (270 * val1))) + (160 * (pow(val2, 3)) * (87 + (val1 * (11 + (15 * val1))))) + (-80 * (pow(val2, 2)) * (347 + (val1 * (515 + (257 * val1))))) + (57 * (1 + val1) * (33 + (4 * val1 * (2 + val1) * (17 + (6 * val1 * (2 + val1)))))) + (2 * val2 * (3113 + (5 * val1 * (557 + (3 * val1 * (-19 + (val1 * (-17 + (5 * val1 * (5 + val1))))))))))) * const_expr0_;
	}
};


void ThreePointSystemJacobianEquation5::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::transform(variables[0].begin(), variables[0].end(), variables[1].begin(), derivatives.begin(), comp_func_three_point_system8(const_expr0_));
}


struct comp_func_three_point_system9
{
	const cudaT const_expr0_;

	comp_func_three_point_system9(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<0>(t), 2)) * (pow((1 + thrust::get<1>(t)), -6)) * (252 + (-32 * (pow(thrust::get<2>(t), 3)) * (289 + (216 * thrust::get<1>(t)))) + (8 * (pow(thrust::get<2>(t), 2)) * (367 + (thrust::get<1>(t) * (-43 + (45 * thrust::get<1>(t)))))) + (thrust::get<1>(t) * (145 + (3 * thrust::get<1>(t) * (111 + (47 * thrust::get<1>(t)))))) + (-4 * thrust::get<2>(t) * (308 + (thrust::get<1>(t) * (-50 + (87 * thrust::get<1>(t))))))) * const_expr0_;
	}
};


void ThreePointSystemJacobianEquation6::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system9(const_expr0_));
}


struct comp_func_three_point_system10
{
	const cudaT const_expr0_;

	comp_func_three_point_system10(const cudaT const_expr0)
		: const_expr0_(const_expr0) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow(thrust::get<0>(t), 2)) * (pow((1 + thrust::get<1>(t)), -5)) * ((24 * (pow(thrust::get<2>(t), 2)) * (343 + (270 * thrust::get<1>(t)))) + (5 * (62 + (thrust::get<1>(t) * (2 + (29 * thrust::get<1>(t)))))) + (-5 * thrust::get<2>(t) * (291 + (thrust::get<1>(t) * (-13 + (60 * thrust::get<1>(t))))))) * const_expr0_;
	}
};


void ThreePointSystemJacobianEquation7::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system10(const_expr0_));
}


struct comp_func_three_point_system11
{
	const cudaT const_expr0_;
	const cudaT const_expr1_;

	comp_func_three_point_system11(const cudaT const_expr0, const cudaT const_expr1)
		: const_expr0_(const_expr0), const_expr1_(const_expr1) {}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (pow((1 + thrust::get<1>(t)), -5)) * (((pow((1 + thrust::get<1>(t)), 5)) * const_expr0_) + (2 * thrust::get<0>(t) * (-833 + (64 * (pow(thrust::get<2>(t), 3)) * (343 + (270 * thrust::get<1>(t)))) + (40 * thrust::get<2>(t) * (62 + (thrust::get<1>(t) * (2 + (29 * thrust::get<1>(t)))))) + (-20 * (pow(thrust::get<2>(t), 2)) * (291 + (thrust::get<1>(t) * (-13 + (60 * thrust::get<1>(t)))))) + (-5 * thrust::get<1>(t) * (329 + (3 * thrust::get<1>(t) * (171 + (thrust::get<1>(t) * (97 + (5 * thrust::get<1>(t) * (5 + thrust::get<1>(t)))))))))))) * const_expr1_;
	}
};


void ThreePointSystemJacobianEquation8::operator() (devdat::DimensionIteratorC &derivatives, const devdat::DevDatC &variables)
{
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[0].begin(), variables[1].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[0].end(), variables[1].end(), derivatives.end())), comp_func_three_point_system11(const_expr0_, const_expr1_));
}

