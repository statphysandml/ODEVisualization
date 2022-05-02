#include "../include/flow_equations_t.hpp"

odesolver::DevDatC get_fixed_points()
{
    // double sigma = 10.0;
    double beta = 8.0/3.0;
    double rho = 28.0;
    odesolver::DevDatC fixed_points(std::vector<std::vector<double>>{
        {0.0, 0.0, 0.0},
        {std::sqrt(beta * (rho - 1)), std::sqrt(beta * (rho - 1)), rho - 1},
        {-std::sqrt(beta * (rho - 1)), -std::sqrt(beta * (rho - 1)), rho - 1}
    });
    return fixed_points;
}

void compute_vertex_velocities_t()
{
    auto flow_equations_ptr = generate_flow_equations<LorentzAttractorFlowEquations>(0);
    auto fixed_points = get_fixed_points();
    fixed_points.print_dim_by_dim();

    auto vertex_velocities = compute_vertex_velocities(fixed_points, flow_equations_ptr.get());

    vertex_velocities.print_dim_by_dim();
}

odesolver::DevDatC compute_jacobians()
{

    auto jacobian_equations_ptr = generate_jacobian_equations<LorentzAttractorJacobianEquations>(0);

    auto fixed_points = get_fixed_points();
    fixed_points.print_elem_by_elem();

    auto jacobian_elements = compute_jacobian_elements(fixed_points, jacobian_equations_ptr.get());

    jacobian_elements.print_elem_by_elem();

    return std::move(jacobian_elements);
}

std::vector<std::vector<double>> compute_jacobian_elements_t()
{
    auto jacobian_elements = compute_jacobians();
    return jacobian_elements.transpose_device_data();
}