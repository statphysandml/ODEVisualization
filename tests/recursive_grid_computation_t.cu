void func()
{
    DynamicRecursiveGridComputation dynamic_recursive_grid_computation(
        computation_parameters_.number_of_cubes_per_gpu_call_,
        computation_parameters_.maximum_number_of_gpu_calls_
    );

    odesolver::DevDatC reference_vertices;
    odesolver::DevDatC reference_vertex_velocities;

    PartialRanges partial_ranges(n_branches_, partial_lambda_ranges_, fixed_lambdas_);

    for(auto i = 0; i < partial_ranges.size(); i++)
    {
        auto lambda_ranges = partial_ranges[i];

        dynamic_recursive_grid_computation.initialize(
            std::vector<std::vector<int>> {n_branches_},
            lambda_ranges,
            DynamicRecursiveGridComputation::ReferenceVertices
        );
    
        while(!dynamic_recursive_grid_computation.finished())
        {
            // Compute vertices
            dynamic_recursive_grid_computation.next(vertices);
        
            // Compute vertex velocities
            reference_vertex_velocities = odesolver::DevDatC(reference_vertices.dim_size(), reference_vertices.n_elems());
            compute_vertex_velocities(reference_vertices, reference_vertex_velocities, flow_equations_ptr_.get());
    
            <-> compare
        }
    }
}