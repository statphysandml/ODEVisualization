//
// Created by lukas on 31.07.19.
//

#ifndef PROGRAM_DEV_DAT_HPP
#define PROGRAM_DEV_DAT_HPP

#include <odesolver/header.hpp>
#include <odesolver/util/thrust_functors.hpp>
#include <odesolver/util/json_conversions.hpp>

#include <utility>
#include <numeric>
#include <fstream>
#include <algorithm>

#include <param_helper/filesystem.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace odesolver {

    struct get_nth_element_idx
    {
        get_nth_element_idx(const int n, const int N) :
            n_(n), N_(N)
        {}

        __host__ __device__
        int operator()(const int &idx)
        {
            return idx * N_ + n_;
        }

        const int n_;
        const int N_;
    };

    struct get_transposed_element_idx
    {
        get_transposed_element_idx(const int dim, const int N) :
            dim_(dim), N_(N)
        {}

        __host__ __device__
        int operator()(const int &idx)
        {
            return idx / dim_ + (idx % dim_) * N_;
        }

        const int dim_;
        const int N_;
    };

    template <typename Iterator>
    class DimensionIterator
    {
    public:
        DimensionIterator(const Iterator begin_iterator, size_t N):
            begin_iterator_(begin_iterator), N_(N)
        {}

        const Iterator begin() const
        {
            return begin_iterator_;
        }

        const Iterator end() const
        {
            return begin_iterator_ + N_;
        }

        size_t size() const
        {
            return N_;
        }

        void set_N(const size_t N)
        {
            N_ = N;
        }

    private:
        Iterator begin_iterator_;
        size_t N_;
    };

    template<typename Vec, typename VecIterator, typename ConstVecIterator>
    class DevDat : public Vec
    {
    public:
        /** @brief Default constructor */
        DevDat() : DevDat(0, 0)
        {}

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
        DevDat(std::vector<std::vector<double>> data) : DevDat(data.size(), data[0].size())
        {
            for(auto n = 0; n < dim_; n++)
            {
                thrust::copy(data[n].begin(), data[n].end(), (*this)[n].begin());
            }
            transpose();
        }

        // Copy constructor
        DevDat(const DevDat& other) : Vec(other), dim_(other.dim_), N_(other.N_)
        {
            // std::cout << "Copy constructor is called with other.N=" << other.N_ << ", other.dim_" << other.dim_ << ", other vec size" << other.size() << std::endl;
            initialize_dimension_iterators();
        }

        // Assignment - https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
        DevDat& operator=(DevDat other) {
            // std::cout << "Assignment operator is called" << std::endl;
            /* print_range("This", this->begin(), this->end());
            print_range("Other", other.begin(), other.end()); */
            swapp(*this, other);
            return *this;
        }

        // Move constructor
        DevDat(DevDat&& other) noexcept : DevDat() // initialize via default constructor, C++11 only
        {
            // std::cout << "&& Move operator is called" << std::endl;
            /* print_range("This", this->begin(), this->end());
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
        
        Vec data() const
        {
            return static_cast<Vec>(*this);
        }

        /** @brief Returns entries in i-th dimension of DevDat */
        const DimensionIterator<ConstVecIterator>& operator[] (int i) const
        {
            return const_dimension_iterators_[i];
        }

        /** @brief Returns entries in i-th dimension of DevDat */
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

        /** @brief Returns the n-th elem of DevDat */
        std::vector<double> get_nth_element(const int n, const int start_idx=0, const int end_idx=-1) const
        {
            int n_dims;
            if(end_idx == -1)
                n_dims = dim_ - start_idx;
            else
                n_dims = end_idx - start_idx;
            if(n_dims < 0)
            {
                std::cerr << "error in get_nth_element: end_idx needs to be bigger or equal to start_idx" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else if(end_idx > int(dim_))
            {
                std::cout << "End idx: " << end_idx << " dim_: " << dim_ << std::endl;
                std::cerr << "error in get_nth_element: end_idx bigger than dim_size()" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            
            // map = {start_idx * N_ + n, (start_idx + 1) * N_ n, ....(start_idx + n_dims) * N_ + n}
            Vec nth_element_cuda(n_dims);
            thrust::gather(
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(start_idx), get_nth_element_idx(n, N_)),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(start_idx + n_dims), get_nth_element_idx(n, N_)),
                this->begin(),
                nth_element_cuda.begin()
            );

            std::vector<double> nth_element(n_dims, 0);
            thrust::copy(nth_element_cuda.begin(), nth_element_cuda.end(), nth_element.begin());
            return nth_element;
        }

        /** @brief Sets the n-th elem of DevDat */
        void set_nth_element(const int n, std::vector<double> nth_element, const int start_idx=0)
        {
            if(int(nth_element.size()) - start_idx > int(dim_))
            {
                std::cerr << "error in set_nth_element: nth_element.size() - start_idx bigger than dim_size()" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            Vec nth_element_cuda(nth_element.size());
            thrust::copy(nth_element.begin(), nth_element.end(), nth_element_cuda.begin());
            thrust::scatter(
                nth_element_cuda.begin(),
                nth_element_cuda.end(),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(start_idx), get_nth_element_idx(n, N_)),
                this->begin()
            ); 
        }

        void set_N(const size_t N)
        {
            N_ = N;
            for(auto i = 0; i < dim_; i++)
            {
                dimension_iterators_[i].set_N(N);
                const_dimension_iterators_[i].set_N(N);
            }
        }

        void set_dim(const size_t dim)
        {
            dim_ = dim;
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
                dimension_iterators_.push_back(DimensionIterator<VecIterator> (begin, N_));
                const_dimension_iterators_.push_back(DimensionIterator<ConstVecIterator> (begin, N_));
                thrust::advance(begin, N_);
                thrust::advance(end, N_);
            }
        }

        void reshape(const size_t dim, const size_t N)
        {
            if(dim * N != dim_ * N_)
            {
                std::cerr << "error in reshape: cannot reshape DevDat of shape " << dim_ << " x " << N_ << " into DevDat of shape " << dim << " x " << N << std::endl;
                std::exit(EXIT_FAILURE);
            }
            dim_ = dim;
            N_ = N;
            initialize_dimension_iterators();
        }

        void resize(const size_t dim, const size_t N)
        {
            Vec::resize(dim * N);
            dim_ = dim;
            N_ = N;
            initialize_dimension_iterators();
        }

        /** @brief Converts the given DevDat to a vector of N elements with each of size dim */
        std::vector<std::vector<double>> to_vec_vec() const
        {
            auto transposed_device_data = transposed();
            std::vector<std::vector<double>> data(N_, std::vector<double> (dim_, 0));
            for(auto n = 0; n < n_elems(); n++)
            {
                thrust::copy(transposed_device_data[n].begin(), transposed_device_data[n].end(), data[n].begin());
            }
            return data;
        }

        DevDat transposed() const
        {
            // map = {start_idx * N_ + n, (start_idx + 1) * N_ n, ....(start_idx + n_dims) * N_ + n}
            Vec transposed_device_data(N_ * dim_);
            thrust::gather(
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(0), get_transposed_element_idx(dim_, N_)),
                thrust::make_transform_iterator(
                    thrust::make_counting_iterator(int(N_ * dim_)), get_transposed_element_idx(dim_, N_)),
                this->begin(),
                transposed_device_data.begin()
            );
            return DevDat(transposed_device_data, N_);
        }

        void transpose()
        {
            *this = transposed();
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
                Vec nth_elem(get_nth_element(n));
                print_range("Elem " + std::to_string(n), nth_elem.begin(), nth_elem.end());
            }
        }

        std::string to_string() const
        {
            std::string s = "";
            for(auto dim_index = 0; dim_index < dim_; dim_index++)
            {
                print_range_in_string((*this)[dim_index].begin(), (*this)[dim_index].end(), s);
                s += "\n";
            }
            s.pop_back();
            return s;
        }

        void write_to_file(std::string rel_dir, std::string filename)
        {
            write_devdat_to_file(*this, rel_dir, filename);
        }
            
    private:
        size_t dim_;
        size_t N_;
        std::vector<DimensionIterator<VecIterator>> dimension_iterators_;
        std::vector<DimensionIterator<ConstVecIterator>> const_dimension_iterators_;
    };


    typedef DevDat<dev_vec, dev_iterator, const_dev_iterator> DevDatC;
    typedef DevDat<dev_vec_int, dev_iterator_int, const_dev_iterator_int > DevDatInt;

    typedef DimensionIterator<dev_iterator>DimensionIteratorC;
    typedef DimensionIterator<const_dev_iterator> ConstDimensionIteratorC;

    typedef DimensionIterator<dev_iterator_int> DimensionIteratorInt;
    typedef DimensionIterator<const_dev_iterator_int> ConstDimensionIteratorInt;

    typedef DevDat<dev_vec_bool, dev_iterator_bool, const_dev_iterator_bool> DevDatBool;

    void write_data_to_ofstream(const odesolver::DevDatC &data, std::ofstream &os, std::vector<int> skip_iterators_in_dimensions = std::vector<int>{}, std::vector< dev_iterator > end_iterators = std::vector< dev_iterator > {});

    void write_devdat_to_file(DevDatC &data, std::string rel_dir, std::string filename);

    DevDatC load_devdat(std::string rel_dir, std::string filename);
}

#endif //PROGRAM_DEV_DAT_HPP
