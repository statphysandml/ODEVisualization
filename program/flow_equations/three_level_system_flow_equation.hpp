#ifndef PROJECT_THREELEVELSYSTEMFLOWEQUATION_HPP
#define PROJECT_THREELEVELSYSTEMFLOWEQUATION_HPP

#include <math.h>
#include <tuple>

#include "../include/flow_equation_interface/flow_equation.hpp"


struct comp_func_three_level_system0
{
	comp_func_three_level_system0()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (6 * thrust::get<2>(t)) + (-0.05 * thrust::get<1>(t)) + (0.05 * thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation0 : public FlowEquation
{
	ThreeLevelSystemFlowEquation0(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[10].begin(), variables[12].begin(), variables[4].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[10].end(), variables[12].end(), variables[4].end(), derivatives.end())), comp_func_three_level_system0());
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system1
{
	comp_func_three_level_system1()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (-0.05 * thrust::get<1>(t)) + (-3 * thrust::get<3>(t)) + (0.05 * thrust::get<2>(t)) + (2 * thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation1 : public FlowEquation
{
	ThreeLevelSystemFlowEquation1(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[11].begin(), variables[13].begin(), variables[9].begin(), variables[4].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[11].end(), variables[13].end(), variables[9].end(), variables[4].end(), derivatives.end())), comp_func_three_level_system1());
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemFlowEquation2 : public FlowEquation
{
	ThreeLevelSystemFlowEquation2(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[10].begin(), variables[10].end(), variables[14].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-0.05 * val2) + (2 * val1); });
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system2
{
	comp_func_three_level_system2()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (0.05 * thrust::get<2>(t)) + (-3 * thrust::get<0>(t)) + (-0.05 * thrust::get<3>(t)) + (-2 * thrust::get<1>(t));
	}
};

struct ThreeLevelSystemFlowEquation3 : public FlowEquation
{
	ThreeLevelSystemFlowEquation3(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[3].begin(), variables[15].begin(), variables[13].begin(), variables[9].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[3].end(), variables[15].end(), variables[13].end(), variables[9].end(), derivatives.end())), comp_func_three_level_system2());
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system3
{
	comp_func_three_level_system3()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (-6 * thrust::get<4>(t)) + (0.05 * thrust::get<1>(t)) + (2 * thrust::get<2>(t)) + (-0.05 * thrust::get<3>(t)) + (-2 * thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation4 : public FlowEquation
{
	ThreeLevelSystemFlowEquation4(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[16].begin(), variables[12].begin(), variables[14].begin(), variables[10].begin(), variables[4].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[16].end(), variables[12].end(), variables[14].end(), variables[10].end(), variables[4].end(), derivatives.end())), comp_func_three_level_system3());
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system4
{
	comp_func_three_level_system4()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (2 * thrust::get<3>(t)) + (-6 * thrust::get<0>(t)) + (-0.05 * thrust::get<1>(t)) + (-2 * thrust::get<2>(t));
	}
};

struct ThreeLevelSystemFlowEquation5 : public FlowEquation
{
	ThreeLevelSystemFlowEquation5(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[5].begin(), variables[11].begin(), variables[17].begin(), variables[13].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[5].end(), variables[11].end(), variables[17].end(), variables[13].end(), derivatives.end())), comp_func_three_level_system4());
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemFlowEquation6 : public FlowEquation
{
	ThreeLevelSystemFlowEquation6(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[16].begin(), variables[16].end(), variables[12].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-2 * val2) + (0.05 * val1); });
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system5
{
	comp_func_three_level_system5()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (-2 * thrust::get<3>(t)) + (-3 * thrust::get<1>(t)) + (0.05 * thrust::get<2>(t)) + (2 * thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation7 : public FlowEquation
{
	ThreeLevelSystemFlowEquation7(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[17].begin(), variables[7].begin(), variables[15].begin(), variables[13].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[17].end(), variables[7].end(), variables[15].end(), variables[13].end(), derivatives.end())), comp_func_three_level_system5());
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemFlowEquation8 : public FlowEquation
{
	ThreeLevelSystemFlowEquation8(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[16].begin(), variables[16].end(), variables[14].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (-2 * val2) + (2 * val1); });
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system6
{
	comp_func_three_level_system6()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<3>(t) = (6 * thrust::get<2>(t)) + (0.05 * thrust::get<0>(t)) + (-0.05 * thrust::get<1>(t));
	}
};

struct ThreeLevelSystemFlowEquation9 : public FlowEquation
{
	ThreeLevelSystemFlowEquation9(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[3].begin(), variables[1].begin(), variables[13].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[3].end(), variables[1].end(), variables[13].end(), derivatives.end())), comp_func_three_level_system6());
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system7
{
	comp_func_three_level_system7()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (-3 * thrust::get<1>(t)) + (0.05 * thrust::get<3>(t)) + (-0.05 * thrust::get<2>(t)) + (-2 * thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation10 : public FlowEquation
{
	ThreeLevelSystemFlowEquation10(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[2].begin(), variables[13].begin(), variables[0].begin(), variables[4].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[2].end(), variables[13].end(), variables[0].end(), variables[4].end(), derivatives.end())), comp_func_three_level_system7());
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemFlowEquation11 : public FlowEquation
{
	ThreeLevelSystemFlowEquation11(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[1].begin(), variables[1].end(), variables[5].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (0.05 * val2) + (-2 * val1); });
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system8
{
	comp_func_three_level_system8()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = (-0.05 * thrust::get<2>(t)) + (-3 * thrust::get<1>(t)) + (0.05 * thrust::get<3>(t)) + (2 * thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation12 : public FlowEquation
{
	ThreeLevelSystemFlowEquation12(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[6].begin(), variables[12].begin(), variables[4].begin(), variables[0].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[6].end(), variables[12].end(), variables[4].end(), variables[0].end(), derivatives.end())), comp_func_three_level_system8());
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system9
{
	comp_func_three_level_system9()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<5>(t) = (-6 * thrust::get<3>(t)) + (-0.05 * thrust::get<2>(t)) + (-2 * thrust::get<4>(t)) + (0.05 * thrust::get<0>(t)) + (2 * thrust::get<1>(t));
	}
};

struct ThreeLevelSystemFlowEquation13 : public FlowEquation
{
	ThreeLevelSystemFlowEquation13(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[1].begin(), variables[7].begin(), variables[3].begin(), variables[13].begin(), variables[5].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[1].end(), variables[7].end(), variables[3].end(), variables[13].end(), variables[5].end(), derivatives.end())), comp_func_three_level_system9());
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system10
{
	comp_func_three_level_system10()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = 2 * ((-1 * thrust::get<2>(t)) + (-3 * thrust::get<1>(t)) + (0.025 * thrust::get<3>(t)) + thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation14 : public FlowEquation
{
	ThreeLevelSystemFlowEquation14(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[8].begin(), variables[14].begin(), variables[4].begin(), variables[2].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[8].end(), variables[14].end(), variables[4].end(), variables[2].end(), derivatives.end())), comp_func_three_level_system10());
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemFlowEquation15 : public FlowEquation
{
	ThreeLevelSystemFlowEquation15(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[3].begin(), variables[3].end(), variables[7].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return (2 * val1) + (-0.05 * val2); });
	}

private:
	const cudaT k;

};


struct comp_func_three_level_system11
{
	comp_func_three_level_system11()
	{}

	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{
		thrust::get<4>(t) = -2 * ((-1 * thrust::get<3>(t)) + (1.5 * thrust::get<1>(t)) + (0.025 * thrust::get<2>(t)) + thrust::get<0>(t));
	}
};

struct ThreeLevelSystemFlowEquation16 : public FlowEquation
{
	ThreeLevelSystemFlowEquation16(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(variables[8].begin(), variables[16].begin(), variables[6].begin(), variables[4].begin(), derivatives.begin())),thrust::make_zip_iterator(thrust::make_tuple(variables[8].end(), variables[16].end(), variables[6].end(), variables[4].end(), derivatives.end())), comp_func_three_level_system11());
	}

private:
	const cudaT k;

};

struct ThreeLevelSystemFlowEquation17 : public FlowEquation
{
	ThreeLevelSystemFlowEquation17(const cudaT k_) : k(k_)
	{}

	void operator() (DimensionIteratorC &derivatives, const DevDatC &variables) override
	{
		thrust::transform(variables[5].begin(), variables[5].end(), variables[7].begin(), derivatives.begin(), [] __host__ __device__ (const cudaT &val1, const cudaT &val2) { return 2 * (val1 + (-1 * val2)); });
	}

private:
	const cudaT k;

};

class ThreeLevelSystemFlowEquations : public FlowEquationsWrapper
{
public:
	ThreeLevelSystemFlowEquations(const cudaT k_) : k(k_)
	{
		flow_equations = std::vector< FlowEquation* > {
			new ThreeLevelSystemFlowEquation0(k),
			new ThreeLevelSystemFlowEquation1(k),
			new ThreeLevelSystemFlowEquation2(k),
			new ThreeLevelSystemFlowEquation3(k),
			new ThreeLevelSystemFlowEquation4(k),
			new ThreeLevelSystemFlowEquation5(k),
			new ThreeLevelSystemFlowEquation6(k),
			new ThreeLevelSystemFlowEquation7(k),
			new ThreeLevelSystemFlowEquation8(k),
			new ThreeLevelSystemFlowEquation9(k),
			new ThreeLevelSystemFlowEquation10(k),
			new ThreeLevelSystemFlowEquation11(k),
			new ThreeLevelSystemFlowEquation12(k),
			new ThreeLevelSystemFlowEquation13(k),
			new ThreeLevelSystemFlowEquation14(k),
			new ThreeLevelSystemFlowEquation15(k),
			new ThreeLevelSystemFlowEquation16(k),
			new ThreeLevelSystemFlowEquation17(k)
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
		return "three_level_system";
	}

	const static uint8_t dim = 18;

private:
	const cudaT k;
	std::vector < FlowEquation* > flow_equations;
};

# endif //PROJECT_THREELEVELSYSTEMFLOWEQUATION_HPP
