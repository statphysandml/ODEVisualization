//
// Created by lukas on 31.07.19.
//

#ifndef PROGRAM_DEV_DAT_HPP
#define PROGRAM_DEV_DAT_HPP

#include "header.hpp"
#include "../extern/thrust_functors.hpp"

#include <utility>
#include <numeric>
#include <fstream>
#include <algorithm>

namespace odesolver {
    template <typename Iterator>
    class DimensionIterator
    {
    public:
        DimensionIterator(const Iterator begin_iterator, const Iterator end_iterator):
            begin_iterator_(begin_iterator), end_iterator_(end_iterator), N_(end_iterator - begin_iterator)
        {}

        const Iterator begin() const
        {
            return begin_iterator_;
        }

        const Iterator end() const
        {
            return end_iterator_;
        }

        size_t size() const
        {
            return N_;
        }

    private:
        Iterator begin_iterator_;
        Iterator end_iterator_;
        size_t N_;
    };

    template<typename Vec, typename VecIterator, typename ConstVecIterator>
    class DevDat : public Vec
    {
    public:
        /** @brief Default constructor */
        DevDat() : DevDat(0, 0)
        {
            initialize_dimension_iterators();
        }

        /** @brief Constructor for DevDat
         * 
         * @param dim Number of dimensions
         * @param N Number of entries per dimension
         * @param init_val Default value for each entry
         */
        DevDat(const size_t dim, const size_t N, const cudaT init_val = 0) : Vec(dim * N, init_val), dim_(dim), N_(N)
        {
            initialize_dimension_iterators();
        }

        /** @brief Constructor for generating a DevDat from a flattened Vec vector
         * 
         * @param device_data Flattened Vec vector
         * @param dim Number of dimensions
         */

        DevDat(const Vec device_data, const size_t dim) : Vec(device_data), dim_(dim), N_(device_data.size() / dim)
        {
            initialize_dimension_iterators();
        }
    
        /** @brief Constructor for generating a DevDat based on a vector of elements -> len(elements) = dim and len(vector) = N
         * 
         * Can be reverted by transpose_device_data()
         * 
         * @param data Vector of elements
         */
        DevDat(std::vector<std::vector<double>> data) : DevDat(data[0].size(), data.size())
        {
            // Fill iterators with data
            for(auto j = 0; j < dim_; j++) {
                dev_iterator it = (*this)[j].begin();
                for (auto i = 0; i < N_; i++) {
                    *it = data[i][j];
                    it++;
                }
            }
        }

        // Copy constructor
        DevDat(const DevDat& other) : Vec(other), dim_(other.dim_), N_(other.N_)
        {
            // std::cout << "Copy constructor is called" << std::endl;
            initialize_dimension_iterators();
        }

        // Assignment - https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
        DevDat& operator=(DevDat other) {
            /* std::cout << "Assignment operator is called" << std::endl;
            print_range("This", this->begin(), this->end());
            print_range("Other", other.begin(), other.end()); */
            swapp(*this, other);
            return *this;
        }

        // Move constructor
        DevDat(DevDat&& other) noexcept : DevDat() // initialize via default constructor, C++11 only
        {
            /* std::cout << "&& Move operator is called" << std::endl;
            print_range("This", this->begin(), this->end());
            print_range("Other", other.begin(), other.end()); */
            swapp(*this, other);
        }

        friend void swapp(DevDat& first, DevDat& second) // nothrow
        {
            // enable ADL (not necessary in our case, but good practice)
            using std::swap;

            // by swapping the members of two objects,
            // the two objects are effectively swapped
            thrust::swap(static_cast<Vec&>(first), static_cast<Vec&>(second));
            swap(first.dim_, second.dim_);
            swap(first.N_, second.N_);
            swap(first.dimension_iterators_, second.dimension_iterators_);
            swap(first.const_dimension_iterators_, second.const_dimension_iterators_);
        }
        
        /** @brief Returns entries in i-th dimension of the DevDat */
        const DimensionIterator<ConstVecIterator>& operator[] (int i) const
        {
            return const_dimension_iterators_[i];
        }

        /** @brief Returns entries in i-th dimension of the DevDat */
        DimensionIterator<VecIterator>& operator[] (int i)
        {
            return dimension_iterators_[i];
        }

        /** @brief Returns the number of dimensions */
        size_t dim_size() const
        {
            return dim_;
        }

        /** @brief Returns the number of elements */
        size_t n_elems() const
        {
            return N_;
        }

        /** @brief Assign a flattened vector to the underlying vector representation. */
        template<typename VecB>
        void fill_by_vec(VecB &other)
        {
            Vec::operator=(other);
        }

        /** @brief Returns the n-th elem of the DevDat */
        std::vector<double> get_nth_element(const int n) const
        {
            std::vector<double> ith_element(dim_, 0);
            auto iterator = this->begin();
            // Jump to ith element in zeroth dimension
            thrust::advance(iterator, n);
            ith_element[0] = *iterator;
            // Fill further dimensions
            for(auto j = 1; j < dim_; j++)
            {
                thrust::advance(iterator, N_);
                ith_element[j] = *iterator;
            }
            return ith_element;
        }

        void set_dim(const size_t dim)
        {
            dim_ = dim;
        }

        void set_N(const size_t N)
        {
            N_ = N;
        }

        void initialize_dimension_iterators()
        {
            dimension_iterators_.clear();
            const_dimension_iterators_.clear();

            VecIterator begin = this->begin();
            VecIterator end = this->begin();
            thrust::advance(end, N_);
            dimension_iterators_.reserve(dim_);
            for(auto i = 0; i < dim_; i++)
            {
                dimension_iterators_.push_back(DimensionIterator<VecIterator> (begin, end));
                const_dimension_iterators_.push_back(DimensionIterator<ConstVecIterator> (begin, end));
                thrust::advance(begin, N_);
                thrust::advance(end, N_);
            }
        }

        /** @brief Converts the given DevDat to a vector of N elements with each of size dim */
        std::vector<std::vector<double>> transpose_device_data() const
        {
            // dim x total_number_of_coordinates (len = total_number_of_coordinates)
            // vs. total_number_of_coordinates x dim (len = dim)
            thrust::host_vector<cudaT> host_device_data(*this);

            std::vector<std::vector<double >> transposed_device_data(N_, std::vector<double> (dim_, 0));
            for(auto j = 0; j < dim_; j++) {
                for (auto i = 0; i < N_; i++) {
                    transposed_device_data[i][j] = host_device_data[j * N_ + i];
                }
            }
            return transposed_device_data;
        }

        /** @brief Print dimension by dimension */
        void print_dim_by_dim() const
        {
            for(auto dim_index = 0; dim_index < dim_; dim_index++)
                print_range("Entries in dimension " + std::to_string(dim_index + 1), (*this)[dim_index].begin(), (*this)[dim_index].end());
        }

        /** @brief Print element by element */
        void print_elem_by_elem() const
        {
            for(auto n = 0; n < N_; n++)
            {
                dev_vec nth_elem(get_nth_element(n));
                print_range("Elem " + std::to_string(n), nth_elem.begin(), nth_elem.end());
            }
        }
            
    private:
        size_t dim_;
        size_t N_;
        std::vector< DimensionIterator<VecIterator> > dimension_iterators_;
        std::vector < DimensionIterator<ConstVecIterator> > const_dimension_iterators_;
    };


    typedef DevDat<dev_vec, dev_iterator, const_dev_iterator> DevDatC;
    typedef DevDat<dev_vec_int, dev_iterator_int, const_dev_iterator_int > DevDatInt;

    typedef DimensionIterator<dev_iterator>DimensionIteratorC;
    typedef DimensionIterator<const_dev_iterator> ConstDimensionIteratorC;

    typedef DimensionIterator<dev_iterator_int> DimensionIteratorInt;
    typedef DimensionIterator<const_dev_iterator_int> ConstDimensionIteratorInt;

    typedef DevDat<dev_vec_bool, dev_iterator_bool, const_dev_iterator_bool> DevDatBool;

    void write_data_to_ofstream(const odesolver::DevDatC &data, std::ofstream &os, std::vector<int> skip_iterators_in_dimensions = std::vector<int>{}, std::vector< dev_iterator > end_iterators = std::vector< dev_iterator > {});
}

#endif //PROGRAM_DEV_DAT_HPP
