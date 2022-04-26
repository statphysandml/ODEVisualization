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

#endif //PROJECT_RANDOM_HPP
