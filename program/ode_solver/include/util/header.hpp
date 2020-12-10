//
// Created by lukas on 10.03.19.
//

#ifndef PROJECT_HEADER_HPP
#define PROJECT_HEADER_HPP

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "fileos.hpp"

using namespace thrust::placeholders;

typedef double cudaT;

#ifndef GPU

typedef thrust::host_vector<cudaT> dev_vec;
typedef thrust::host_vector<int> dev_vec_int;
typedef thrust::host_vector<bool> dev_vec_bool;
typedef thrust::host_vector< dev_vec_int * > dev_ptrvec_vec_int;

typedef thrust::host_vector<cudaT>::iterator dev_iterator;
typedef thrust::host_vector<int>::iterator dev_iterator_int;
typedef thrust::host_vector<bool>::iterator dev_iterator_bool;

typedef thrust::host_vector<cudaT>::const_iterator const_dev_iterator;
typedef thrust::host_vector<int>::const_iterator const_dev_iterator_int;
typedef thrust::host_vector<bool>::const_iterator const_dev_iterator_bool;

typedef thrust::host_vector< dev_iterator_int* > dev_iter_vec_int;

#else

typedef thrust::device_vector<cudaT> dev_vec;
typedef thrust::device_vector<int> dev_vec_int;
typedef thrust::device_vector<bool> dev_vec_bool;
typedef thrust::host_vector< dev_vec_int * > dev_ptrvec_vec_int;

typedef thrust::device_vector<cudaT>::iterator dev_iterator;
typedef thrust::device_vector<int>::iterator dev_iterator_int;
typedef thrust::device_vector<bool>::iterator dev_iterator_bool;

typedef thrust::device_vector<cudaT>::const_iterator const_dev_iterator;
typedef thrust::device_vector<int>::const_iterator const_dev_iterator_int;
typedef thrust::device_vector<bool>::const_iterator const_dev_iterator_bool;

typedef thrust::device_vector< dev_iterator_int* > dev_iter_vec_int;

#endif

#endif //PROJECT_HEADER_HPP
