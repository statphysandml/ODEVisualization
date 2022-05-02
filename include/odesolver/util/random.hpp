//
// Created by kades on 5/22/19.
//

#ifndef PROJECT_RANDOM_HPP
#define PROJECT_RANDOM_HPP

#include <random>
#include <thrust/random.h>


static std::random_device rd; // random device engine, usually based on /dev/random on UNIX-like systems
// initialize Mersennes' twister using rd to generate the seed
static std::mt19937 gen(rd());

// https://stackoverflow.com/questions/12614164/generating-random-numbers-with-uniform-distribution-using-thrust/12614606#12614606
struct RandomNormalGenerator
{
    __host__ __device__
    float operator () (int idx)
    {
        thrust::default_random_engine randEng;
        thrust::random::normal_distribution<cudaT> dist(0.0f, 1.0f);
        randEng.discard(idx);
        return dist(randEng);
    }
};

/*

// https://stackoverflow.com/questions/40142493/using-curand-inside-a-thrust-functor

// Functor for initializing the random generators on the GPU
struct curand_setup
{
    const unsigned long long offset;

    curand_setup(const unsigned long long offset) :
            offset(offset)
    {}

    using init_tuple = thrust::tuple<int, curandState&>;
    __device__
    void operator()(init_tuple t){
        curandState s;
        unsigned int seed = thrust::get<0>(t);
        // seed a random number generator
        curand_init(seed, 0, offset, &s);
        thrust::get<1>(t) = s;
    }
};


// Functor for initializing the lattice sites with random Gaussian noise
struct initialization
{
    const double sqrt2epsilon;

    initialization(const double epsilon) :
            sqrt2epsilon(std::sqrt(2.0 * epsilon))
    {}

    using Tuple = thrust::tuple<curandState&, cudaT&>;
    __device__
    void operator()(Tuple t) {
        thrust::get<1>(t) = sqrt2epsilon * curand_normal(&thrust::get<0>(t));
    }
};

void initialize_helper(std::string starting_mode, dev_vec &lattice, thrust::device_vector<curandState> &s,
                       unsigned long long &rnd_offset, const double epsilon)
{
    if(starting_mode == "hot") {
        s = thrust::device_vector<curandState>(lattice.size());

        // Initialize the random generators
        thrust::for_each_n(thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0), s.begin())),
                           lattice.size(), curand_setup(rnd_offset));

        // Initialize each lattice site with a random Gaussian number, where 2 * epsilon refers to the standard deviation
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(s.begin(), lattice.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(s.end(), lattice.end())),
                         initialization(epsilon));
        rnd_offset += 1;
    }
    else
        thrust::fill(lattice.begin(), lattice.end(), 0.0);
    // print_range("Initialized Lattice", lattice.begin(), lattice.end());
} */

#endif //PROJECT_RANDOM_HPP
