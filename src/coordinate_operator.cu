#include "../include/coordinate_operator.hpp"

// ToDo: Move??
std::vector< Eigen::MatrixXd* > compute_jacobian_elements(const odesolver::DevDatC &coordinates, JacobianWrapper * jacobian_equations)
{
    const uint dim = coordinates.dim_size();
    auto number_of_coordinates = coordinates[0].size();
    auto jacobians = std::vector< Eigen::MatrixXd* > (number_of_coordinates);

    for(auto coor_idx = 0; coor_idx < number_of_coordinates; coor_idx ++)
        jacobians[coor_idx] = new Eigen::MatrixXd(dim, dim);

    // Evaluate jacobian for each element separately
    for(auto matrix_idx = 0; matrix_idx < pow(dim, 2); matrix_idx ++)
    {
        odesolver::DevDatC jacobian_element_wrapper(1, number_of_coordinates);
        DimensionIteratorC &jacobian_element = jacobian_element_wrapper[0];
        // dev_vec jacobian_element(number_of_coordinates);
        (*jacobian_equations)(jacobian_element, coordinates, matrix_idx);
        thrust::host_vector<cudaT> host_jacobian_element(jacobian_element.begin(), jacobian_element.end());
        for(auto coor_idx = 0; coor_idx < number_of_coordinates; coor_idx ++) {
            (*jacobians[coor_idx])(matrix_idx) = host_jacobian_element[coor_idx];
        }
    }
    for(auto coor_idx = 0; coor_idx < number_of_coordinates; coor_idx++) {
        (*jacobians[coor_idx]).transposeInPlace();
    }
    return jacobians;
}


CoordinateOperatorParameters::CoordinateOperatorParameters(const json params_, const PathParameters path_parameters_) : FRGVisualizationParameters(params_, path_parameters_),
                                                            dim(get_value_by_key<int>("dim")),
                                                            k(get_value_by_key<cudaT>("k"))
{
    auto coordinates_ = get_value_by_key<json>("initial_coordinates");
    std::transform(coordinates_.begin(), coordinates_.end(), std::back_inserter(initial_coordinates),
                   [] (json &dat) { return dat.get< std::vector<double> >(); });

    std::string theory = path_parameters.theory;
    flow_equations = FlowEquationsWrapper::make_flow_equation(theory);
    jacobian_equations = JacobianWrapper::make_jacobian(theory);
}

CoordinateOperatorParameters CoordinateOperatorParameters::from_file(
        const std::string theory,
        const std::string mode_type,
        const std::string dir,
        const std::string root_dir,
        const bool relative_path)
{
    return CoordinateOperatorParameters(
            Parameters::read_parameter_file(root_dir + "/" + theory + "/" + dir + "/", "config", relative_path),
            PathParameters(theory, mode_type, root_dir, relative_path));
}

CoordinateOperatorParameters CoordinateOperatorParameters::from_parameters(
        const std::string theory,
        const std::vector <std::vector<double> > initial_coordinates,
        const std::string mode_,
        const std::string root_dir,
        const bool relative_path)
{
    return CoordinateOperatorParameters(
            json{{"initial_coordinates", initial_coordinates},
                 {"mode",                mode_}},
            PathParameters(theory, mode_, root_dir, relative_path));
}

/* template<typename ConditionalObserverParameters>
CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters>::EvolveOnConditionParameters(const json params_) :
    Parameters(params_),
    observe_every_nth_step(get_value_by_key<cudaT>("observe_every_nth_step")),
    maximum_total_number_of_steps(get_value_by_key<cudaT>("maximum_total_number_of_steps")),
    results_dir(get_value_by_key<std::string>("results_dir")),
    conditional_oberserver_name(get_value_by_key<std::string>("conditional_observer_name"))
{}

template<typename ConditionalObserverParameters>
CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalObserverParameters>::EvolveOnConditionParameters(
    const uint observe_every_nth_step_,
    const uint maximum_total_number_of_steps_,
    const std::string results_dir_) :
    EvolveOnConditionParameters(
            json {{"observe_every_nth_step", observe_every_nth_step_},
                  {"maximum_total_number_of_steps", maximum_total_number_of_steps_},
                  {"results_dir", results_dir_},
                  {"conditional_observer_name", ConditionalObserverParameters::name()}}
    )
{} */

//[ Public functions of CoordinateOperator

// Velocities
void CoordinateOperator::compute_velocities()
{
    raw_velocities = compute_vertex_velocities(raw_coordinates, ep.flow_equations);
}

// Jacobians and eigendata
void CoordinateOperator::compute_jacobians_and_eigendata() {
    std::cout << std::endl;

    jacobians = compute_jacobian_elements(raw_coordinates, ep.jacobian_equations);
    for(auto &jacobian : jacobians) {
        std::cout << "The jacobian matrix is:" << *jacobian << std::endl;
        Eigen::EigenSolver<Eigen::MatrixXd> eigensolver(*jacobian);
        if (eigensolver.info() != Eigen::Success) abort();
        std::cout << "The eigenvalues of the jacobian matrix are:\n" << eigensolver.eigenvalues() << std::endl;
        std::cout << "Here's a matrix whose columns are eigenvectors of the jacobian\n"
                  << "corresponding to these eigenvalues:\n"
                  << eigensolver.eigenvectors() << "\n" << std::endl;
        eigenvectors.push_back(eigensolver.eigenvectors());
        eigenvalues.push_back(eigensolver.eigenvalues());
    }
}

void CoordinateOperator::write_characteristics_to_file(const std::string dir) const
{
    std::vector < std::vector<double> > velocities = raw_velocities.transpose_device_data();
    std::vector < std::vector<double> > coordinates = raw_coordinates.transpose_device_data();

    json j;
    j["coordinates"] = vec_vec_to_json(coordinates);
    if(velocities.size() > 0)
    {
        auto velocities_json = vec_vec_to_json(velocities);
        j["numer_of_velocities"] = velocities.size();
        j["velocities"] = velocities_json;
    }
    if(jacobians.size() > 0)
    {
        auto jacobians_json = jacobians_to_json();
        j["jacobian"] = jacobians_json;
        auto eigenvectors_json = eigenvectors_to_json();
        j["eigenvectors"] = eigenvectors_json;
        auto eigenvalues_json = eigenvalues_to_json();
        j["eigenvalues"] = eigenvalues_json;
    }
    Parameters::write_parameter_file(j, ep.path_parameters.get_base_path() + dir + "/", "characteristics", ep.path_parameters.relative_path);
}

std::vector< std::vector<double> > CoordinateOperator::get_initial_coordinates() const
{
    return ep.initial_coordinates;
}

void CoordinateOperator::set_raw_coordinates(const odesolver::DevDatC coordinates)
{
    raw_coordinates = coordinates;
}

odesolver::DevDatC CoordinateOperator::get_raw_coordinates() const
{
    return raw_coordinates;
}

odesolver::DevDatC CoordinateOperator::get_raw_velocities() const
{
    return raw_velocities;
}

std::vector< Eigen::MatrixXd* > CoordinateOperator::get_jacobians() const
{
    return jacobians;
};

std::vector<std::vector< std::vector<cudaT>>> CoordinateOperator::get_real_parts_of_eigenvectors() const
{
    std::vector<std::vector<std::vector<cudaT>>> real_eigenvectors;
    for(auto &eigenvector : eigenvectors)
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
};

std::vector<std::vector<cudaT>> CoordinateOperator::get_real_parts_of_eigenvalues() const
{
    std::vector<std::vector<cudaT>> eigenvalues_vec;
    for(auto &eigenvalue : eigenvalues) {
        std::vector<cudaT> eigenvals_vec;
        for(auto i = 0; i < eigenvalue.size(); i++)
            eigenvals_vec.push_back(eigenvalue[i].real());
        eigenvalues_vec.push_back(eigenvals_vec);
    }
    return eigenvalues_vec;
}

Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType CoordinateOperator::get_eigenvector(const int i) const
{
    return eigenvectors[i];
}

Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType CoordinateOperator::get_eigenvalue(const int i) const
{
    return eigenvalues[i];
}

std::vector<int> CoordinateOperator::get_indices_with_saddle_point_characteristics() const
{
    std::vector<int> saddle_point_indices{};
    for(auto j = 0; j < eigenvalues.size(); j++)
    {
        int eigenvalue_sign = sign(eigenvalues[j][0].real());
        for (auto i = 1; i < eigenvalues[j].size(); i++)
        {
            if (sign(eigenvalues[j][i].real()) != eigenvalue_sign) {
                saddle_point_indices.push_back(j);
                break;
            }
        }
    }
    return saddle_point_indices;
}

//]

//[ Private functions of CoordinateOperator

json CoordinateOperator::vec_vec_to_json(const std::vector < std::vector<double> > data) const
{
    // Convert to json and write to file
    json j;
    for(const auto &dat : data)
    {
        std::vector <double> vec(dat.size());
        thrust::copy(dat.begin(), dat.end(), vec.begin());
        j.push_back(vec);
    }
    return j;

}

json CoordinateOperator::jacobians_to_json() const
{
    json j;
    for(auto &jacobian : jacobians)
    {
        json jac_json;
        for(auto row_i = 0; row_i < jacobian->rows(); row_i++)
        {
            auto row = jacobian->row(row_i);
            std::vector<double> vec(row.size());
            Eigen::RowVectorXd::Map(&vec[0], row.size()) = row;
            jac_json.push_back(vec);
        }
        j.push_back(jac_json);
    }
    return j;
}

json CoordinateOperator::eigenvectors_to_json() const
{
    json j;
    for(auto &eigenvector : eigenvectors)
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

json CoordinateOperator::eigenvalues_to_json() const
{
    json j;
    for(auto &eigenvalue : eigenvalues) {
        json json_vec;
        for(auto i = 0; i < eigenvalue.size(); i++)
            json_vec.push_back(json{eigenvalue[i].real(), eigenvalue[i].imag()});
        j.push_back(json_vec);
    }
    return j;
}

//]

/* template class CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalRangeObserverParameters>::EvolveOnConditionParameters;
template class CoordinateOperatorParameters::EvolveOnConditionParameters<ConditionalIntersectionObserverParameters>::EvolveOnConditionParameters; */