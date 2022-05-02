#include "../../include/odesolver/util/jacobians.hpp"
    
using json = nlohmann::json;

Jacobians::Jacobians()
{}

Jacobians::Jacobians(std::vector<std::vector<double>> jacobian_elements):
    jacobian_elements_(jacobian_elements)
{
    auto dim = int(std::sqrt(jacobian_elements_[0].size()));

    for(auto& jacobian_element: jacobian_elements_)
    {
        jacobians_.push_back(MatXdMap(jacobian_element.data(), dim, dim));
    }
}

size_t Jacobians::size() const
{
    return jacobians_.size();
}

void Jacobians::compute_characteristics()
{
    eigenvectors_.clear();
    eigenvalues_.clear();
    for(auto &jacobian : jacobians_) {
        // std::cout << "The jacobian matrix is:\n" << jacobian << std::endl;
        Eigen::EigenSolver<MatXd> eigensolver(jacobian);
        if(eigensolver.info() != Eigen::Success)
            abort();
        /* std::cout << "The eigenvalues of the jacobian matrix are:\n" << eigensolver.eigenvalues() << std::endl;
        std::cout << "Here's a matrix whose columns are eigenvectors of the jacobian\n"
                << "corresponding to these eigenvalues:\n"
                << eigensolver.eigenvectors() << "\n" << std::endl; */
        eigenvectors_.push_back(eigensolver.eigenvectors());
        eigenvalues_.push_back(eigensolver.eigenvalues());
    }
}

Jacobians::MatXd Jacobians::get_jacobian(const int i) const
{
    return jacobians_[i];
}

Eigen::EigenSolver<Jacobians::MatXd>::EigenvectorsType Jacobians::get_eigenvector(const int i) const
{
    return eigenvectors_[i];
}

Eigen::EigenSolver<Jacobians::MatXd>::EigenvalueType Jacobians::get_eigenvalue(const int i) const
{
    return eigenvalues_[i];
}

std::vector<std::vector< std::vector<cudaT>>> Jacobians::get_real_parts_of_eigenvectors() const
{
    std::vector<std::vector<std::vector<cudaT>>> real_eigenvectors;
    for(auto &eigenvector : eigenvectors_)
    {
        std::vector<std::vector<cudaT>> real_eigenvecs;
        for(auto eigen_vec_i = 0; eigen_vec_i < eigenvector.cols(); eigen_vec_i++)
        {
            auto eigen_vec = eigenvector.col(eigen_vec_i);
            std::vector<cudaT> realeigen_vec;
            for(auto i = 0; i < eigen_vec.size(); i++)
                realeigen_vec.push_back(eigen_vec[i].real()) ;
            real_eigenvecs.push_back(realeigen_vec);
        }
        real_eigenvectors.push_back(real_eigenvecs);
    }
    return real_eigenvectors;
}
std::vector<std::vector<cudaT>> Jacobians::get_real_parts_of_eigenvalues() const
{
    std::vector<std::vector<cudaT>> eigenvalues_vec;
    for(auto &eigenvalue : eigenvalues_) {
        std::vector<cudaT> eigenvals_vec;
        for(auto i = 0; i < eigenvalue.size(); i++)
            eigenvals_vec.push_back(eigenvalue[i].real());
        eigenvalues_vec.push_back(eigenvals_vec);
    }
    return eigenvalues_vec;
}

std::vector<int> Jacobians::get_indices_with_saddle_point_characteristics() const
{
    std::vector<int> saddle_point_indices{};
    for(auto j = 0; j < eigenvalues_.size(); j++)
    {
        int eigenvalue_sign = sign(eigenvalues_[j][0].real());
        for(auto i = 1; i < eigenvalues_[j].size(); i++)
        {
            if(sign(eigenvalues_[j][i].real()) != eigenvalue_sign) {
                saddle_point_indices.push_back(j);
                break;
            }
        }
    }
    return saddle_point_indices;
}

json Jacobians::jacobians_to_json() const
{
    json j;
    for(auto &jacobian : jacobians_)
    {
        json jac_json;
        for(auto row_i = 0; row_i < jacobian.rows(); row_i++)
        {
            auto row = jacobian.row(row_i);
            std::vector<double> vec(row.size());
            Eigen::RowVectorXd::Map(&vec[0], row.size()) = row;
            jac_json.push_back(vec);
        }
        j.push_back(jac_json);
    }
    return j;
}

json Jacobians::eigenvectors_to_json() const
{
    json j;
    for(auto &eigenvector : eigenvectors_)
    {
        json eigen_json;
        for(auto eigen_vec_i = 0; eigen_vec_i < eigenvector.cols(); eigen_vec_i++)
        {
            auto eigen_vec = eigenvector.col(eigen_vec_i);
            json json_vec;
            for(auto i = 0; i < eigen_vec.size(); i++)
                json_vec.push_back(json{eigen_vec[i].real(), eigen_vec[i].imag()}) ;
            eigen_json.push_back(json_vec);
        }
        j.push_back(eigen_json);
    }
    return j;
}

json Jacobians::eigenvalues_to_json() const
{
    json j;
    for(auto &eigenvalue : eigenvalues_) {
        json json_vec;
        for(auto i = 0; i < eigenvalue.size(); i++)
            json_vec.push_back(json{eigenvalue[i].real(), eigenvalue[i].imag()});
        j.push_back(json_vec);
    }
    return j;
}

int Jacobians::sign(const double val) const
{
    if(val > 0)
        return 1;
    else
        return -1;
}
