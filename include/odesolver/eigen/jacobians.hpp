#ifndef PROGRAM_JACOBIANS_HPP
#define PROGRAM_JACOBIANS_HPP

#include <vector>
#include <numeric>
#include <Eigen/Dense>

#include <odesolver/header.hpp>

#include <param_helper/json.hpp>
using json = nlohmann::json;


namespace odesolver {
    namespace eigen {
        struct Jacobians
        {
            typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXd;
            typedef Eigen::Map<MatXd, Eigen::Unaligned> MatXdMap;

            Jacobians();

            /*
            param: jacobian_elements: List of flattend row-major jacobian matrices
            */
            Jacobians(std::vector<std::vector<double>> jacobian_elements);

            Jacobians(std::vector<double> jacobian_elements, uint dim);

            size_t size() const;

            void compute_characteristics();

            MatXd get_jacobian(const int i) const;

            Eigen::EigenSolver<MatXd>::EigenvectorsType get_eigenvector(const int i) const;

            Eigen::EigenSolver<MatXd>::EigenvalueType get_eigenvalue(const int i) const;

            std::vector<std::vector<std::vector<cudaT>>> get_real_parts_of_eigenvectors() const;

            std::vector<std::vector<cudaT>> get_real_parts_of_eigenvalues() const;

            std::vector<int> get_indices_with_saddle_point_characteristics() const;

            json jacobians_to_json() const;

            json eigenvectors_to_json() const;

            json eigenvalues_to_json() const;

            int sign(const double val) const;

            std::vector<double> jacobian_elements_;

            std::vector<MatXdMap> jacobians_;
            
            std::vector<Eigen::EigenSolver<MatXd>::EigenvectorsType> eigenvectors_;
            std::vector<Eigen::EigenSolver<MatXd>::EigenvalueType> eigenvalues_;
        };
    }
}

#endif //PROGRAM_JACOBIANS_HPP