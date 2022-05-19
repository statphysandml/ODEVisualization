
// Trying to get the system running with variants

/* class observer_generic
        : public boost::static_visitor<>
{
    observer_variants::variant_type operator(
}; */

#include "boost/variant.hpp"

template<typename ...Ts>
struct Tfs
{
    using variant_type = boost::variant<Ts...>;
};

template<typename Derived>
struct Model
{
    bool callable(std::string observer_name) const
    {
        if(Derived::Observer::name() == observer_name)
            return true;
        else
            return false;
    }
};

struct PseudoObserverWrapper : public Model<PseudoObserverWrapper>
{
    PseudoObserverWrapper() = default;
    typedef PseudoObserver Observer;

    PseudoObserverWrapper(json params)
    {

    }
};

struct TrackingObserverWrapper : public Model<TrackingObserverWrapper>
{
    TrackingObserverWrapper() = default;
    typedef TrackingObserver Observer;

    TrackingObserverWrapper(json params)
    {

    }
};

typedef Tfs<PseudoObserverWrapper, TrackingObserverWrapper> observer_variants;

struct if_correct_observer: public boost::static_visitor<bool> {
    if_correct_observer(const std::string observer_name_) : observer_name(observer_name_)
    {}

    template <class T>
    bool operator()(T v) const {
        return v.callable(observer_name);
    }

    const std::string observer_name;
};

void maain()
{
    // Try to decode to each type of the variant. Short-circuit evaluation
    // prevents decoding attempts to following types on success.

    /* assert(boost::apply_visitor(if_visitor(), v0, v1, v2));


    observer_variants::variant_type aux_variant = PseudoObserver();
    std::string observer_name = "TrackingObserver";
    aux_variant::name(); */

    /* auto vart = boost::apply_visitor([](const auto& arg) -> VariantType {
        if constexpr (IsInsideVariantType<std::decay_t<decltype(arg)>>) {
        return VariantType{arg};
    } else {
        // This branch should never be reached
        throw std::runtime_error("can not convert ExtendedVariantType to VarintType inside CodableVariant");
    }
    }, aux_variant */

    std::string observer_name ="TrackingObserver";

    std::vector<observer_variants::variant_type> aux_variant {PseudoObserverWrapper(), TrackingObserverWrapper()};
    observer_variants::variant_type resulting_observer_wrapper;
    for(auto &variant: aux_variant)
    {
        if(boost::apply_visitor(if_correct_observer(observer_name), variant)) {
            resulting_observer_wrapper = variant;
            decltype(resulting_observer_wrapper) actual_observer_wrapper(json {});
            break;
        }
    }

/*    Fileos fileos("test");
    resulting_observer_wrapper::Observer test (fileos.get()); */

    // bool success = aux_try_decode(observer_name, PseudoObserver());

    /* boost::apply_visitor([observer_name](const auto& arg) -> observer_variants::variant_type {
        if(arg::name() == observer_name) {
        return observer_variants::variant_type{arg};
    } else {
        // This branch should never be reached
        throw std::runtime_error("can not convert ExtendedVariantType to VarintType inside CodableVariant");
    }
    }, aux_variant); */
    /* bool success = (aux_try_decode<PseudoObserver, TrackingObserver, ConditionalRangeObserver, ConditionalIntersectionObserver>(observer_name, aux_variant) || ...);

    // If decoding failed on all types, throw exception
    if (!success) {
        std::cout << "Observer not defined" << std::endl;
    } */
}



void hyperbolic_system()
{
    const std::string theory = "hyperbolic_system";

    //[ Fixed point search
    /* const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {20, 20}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0)};

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search"); */
    //]

    //[ Visualization
    const std::vector<int> n_branches {200, 200};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0)};

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {2.99999847412109,-1.99999847412109}});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 20;
    const std::vector<double> shift_per_dim {0.001, 0.001};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-10, 1e-10};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    ConditionalRangeObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization");
    //]

    // Single evaluation
    /* std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {0, 0}};

    CoordinateOperatorParameters evaluator_parameters(theory, coordinates, "/data/", true, "jacobian");
    evaluator_parameters.write_to_file("jacobian"); */

}


void three_d_hyperbolic_system()
{
    const std::string theory = "3D_hyperbolic_system";

    //[ Fixed point search
    /* const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0), std::pair<cudaT, cudaT> (-3.0, 3.0)};

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search"); */
    //]

    //[ Visualization
    const std::vector<int> n_branches {200, 200,  1};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0)};
    const std::vector <std::vector <cudaT> > fixed_variables = std::vector< std::vector<cudaT> > { std::vector<cudaT> {-0.01, 0.0, 0.01} };

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fixed_variables);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {2.99999847412109,-1.99999847412109, 0}});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 20;
    const std::vector<double> shift_per_dim {0.001, 0.001, 0.001};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0), std::pair<cudaT, cudaT> (-100000.0, 100000.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-10, 1e-10, 1e-10};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_distances = std::vector< cudaT > {1e-3};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val,vicinity_distances);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization");
    //]

    //[ Visualization y z
    /* const std::vector<int> n_branches {1, 200, 200};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-5.0, 3.0), std::pair<cudaT, cudaT> (-3.0, 3.0)};
    const std::vector <std::vector <cudaT> > fixed_variables = std::vector< std::vector<cudaT> > { std::vector<cudaT> {2.0, 3.0, 3.8} };

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fixed_variables);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {2.99999847412109,-1.99999847412109, 0}});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 20;
    const std::vector<double> shift_per_dim {0.001, 0.001, 0.001};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-2.0, 6.0), std::pair<cudaT, cudaT> (-5.0, 3.0), std::pair<cudaT, cudaT> (-3.0, 3.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-10, 1e-10, 1e-10};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_distances = std::vector< cudaT > {1e-2};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val,vicinity_distances);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization_yz"); */
    //]


    // Single evaluation
    /* std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {0, 0}};

    CoordinateOperatorParameters evaluator_parameters(theory, coordinates, "/data/", true, "jacobian");
    evaluator_parameters.write_to_file("jacobian"); */

}


void three_point_system()
{
    const std::string theory = "three_point_system";

    //[ Fixed point search
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    // mu, lambda, g
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
        std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search");
    //]
    //[ Visualization
    /* const std::vector<int> n_branches {1, 80, 80};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {0, 0.0187308, -0.164471} };

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fix_lambdas);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {
        std::vector<cudaT> {0.0, -1.71661376890953e-07, 6.29425048748189e-08 },
        std::vector<cudaT> {0.0187292264855426, -0.50875129285066, 0.299244802474976},
        std::vector<cudaT> {-0.164468709884151, -0.164090993942753, 0.570244262264621 }
    });
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 10;
    const std::vector<double> shift_per_dim {0.001, 0.001, 0.001};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-12, 1e-12, 1e-12};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_distances = std::vector< cudaT > {1e-5};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val,vicinity_distances);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 100000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization"); */
    //]

    // Single evaluation
    /* std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {-0.164467294607589, -0.164092870399133, 0.570242801808599}};

    CoordinateOperatorParameters evaluator_parameters(theory, coordinates, "/data/", true, "jacobian");
    evaluator_parameters.write_to_file("jacobian"); */
    //]
}

void four_point_system()
{
    const std::string theory = "four_point_system";

    //[ Fixed point search
    /* const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    // mu, lambda, g
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search"); */
    //]
    //[ Visualization

    const std::vector<int> n_branches {200, 200, 1, 1, 1};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-3.0, 2.0), std::pair<cudaT, cudaT> (-2.0, 1.0)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {-0.1082}, std::vector<cudaT> {0.63998}, std::vector<cudaT> {0.553988}};

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fix_lambdas);

    // {-0.226228, 0.63998, 0.553988, -0.0603059, -0.1082});

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {
            std::vector<cudaT> {-0.226227508697766,-0.0603059942535251,-0.108200256143866,0.639979298994415,0.553988357314688}
    });
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    /* const uint N_per_eigen_dim = 10;
    const std::vector<double> shift_per_dim {0.001, 0.001, 0.001};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.61, 1.0)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-12, 1e-12, 1e-12};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_distances = std::vector< cudaT > {1e-5};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val,vicinity_distances);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 100000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);*/

    visualization_parameters.write_to_file("visualization");
    //]

    // Single evaluation
    /* std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {-0.164467294607589, -0.164092870399133, 0.570242801808599}};

    CoordinateOperatorParameters evaluator_parameters(theory, coordinates, "/data/", true, "jacobian");
    evaluator_parameters.write_to_file("jacobian"); */
    //]
}

void custom_main()
{
    // search_fixed_points();
    // search_fixed_point_with_parameters();
    evaluate();

    // hyperbolic_system();
    // three_point_system();
    // three_d_hyperbolic_system();
    // four_point_system();

    // FRGVisualizationParameters::initialize_theory_folder("three_point_system");

    /* Writing parameter files */

    //[ Identity

    // Visualization

    /* const std::string theory = "identity";
    const std::vector<int> n_branches{1, 1, 80, 100, 1};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (-2.0, 2.0)};
    // const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > {
    //        std::vector<cudaT> {-1, -0.2, 0.2, 1}, std::vector<cudaT> {-5, 5}, std::vector<cudaT> {1}};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > {
            std::vector<cudaT> {0.5}, std::vector<cudaT> {-0.2}, std::vector<cudaT> {0.3}};

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fix_lambdas);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {0, 0 ,0 ,0 ,0}});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 10;
    const std::vector<double> shift_per_dim {0.02, 0.1, 0.01, 0.1, 0.1};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
        std::pair<cudaT, cudaT> (-1.2, 1.2), std::pair<cudaT, cudaT> (-1.2, 1.2), std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (-2.0, 2.0), std::pair<cudaT, cudaT> (-1.2, 1.2)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {0.01, 0.01, 0.01, 0.01, 0.01};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e10;
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 10;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization");*/


    // Fix point search
    /*const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {3, 3, 3, 3, 3}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.1, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)}; */

    /* const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {3, 3}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.1, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)}; */

    /*const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {10, 10, 10, 10, 10, 10, 10, 10}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};

    const int maximum_recursion_depth = 18;

    const std::string theory = "identity";

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search"); */

    //[ Identity 2D

    // Visualization

    /*const std::string theory = "identity";
    const std::vector<int> n_branches{80, 100};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (-2.0, 2.0)};
    const std::vector <std::vector <cudaT> > fix_lambdas {};

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fix_lambdas);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {0, 0 ,0 ,0 ,0}});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 10;
    const std::vector<double> shift_per_dim {0.02, 0.1};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {0.01, 0.01};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e10;
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 10;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization");*7

    // Fix point search -> ToDo

    /* const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {3, 3}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}, std::vector<int> {2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.1, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};*/

    /* const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {10, 10, 10, 10, 10, 10, 10, 10}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2, 2, 2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};*/

    /* const int maximum_recursion_depth = 18;

    const std::string theory = "identity";

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search");*/



    // Evolution
    /* const std::string theory = "identity";
    std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {1, 1}, std::vector<cudaT> {0.5, -1}, std::vector<cudaT> {-0.3, -0.7}};

    CoordinateOperatorParameters evaluator_parameters(theory, coordinates, "/data/", true, "evolve");
    evaluator_parameters.write_to_file("evolve"); */

    // Conditional Evolution
    /* const std::string theory = "identity";
    std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {1, 1}, std::vector<cudaT> {0.5, -1}, std::vector<cudaT> {-0.3, -0.7}};
    CoordinateOperatorParameters evaluator_parameters(theory, coordinates, "/data/", true, "evolve_with_observer");

    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {0.01, 0.01};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e10;
    ConditionalIntersectionObserverParameters conditional_observer_parameters(lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val);
    evaluator_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step=10;
    const uint maximum_total_number_of_steps=1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    evaluator_parameters.append_parameters(evolve_with_observer_parameters);

    evaluator_parameters.write_to_file("evolve_with_observer"); */

    // ToDo: First!! With command ./FRGVisualization three_point_system fixed_point_search
    /* const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {20, 20, 20}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.61, 1.0), std::pair<cudaT, cudaT> (-1.8, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::string theory = "three_point_system";

    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file("fixed_point_search"); */


    // Single evaluation
    /* const std::string theory = "identity";
    std::vector < std::vector<cudaT> > coordinates {std::vector<cudaT> {1, 1, 1, 1, 1}};

    CoordinateOperatorParameters evaluator_parameters(theory, coordinates);
    evaluator_parameters.write_to_file("evaluate");*/

    //]

    //[ FourPointSystem


    // Visualization

    /* const std::string theory = "four_point_system";
    const std::vector<int> n_branches {8, 10, 1, 1, 1};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-3.0, 2.0), std::pair<cudaT, cudaT> (-2.0, 1.0)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {-0.1082}, std::vector<cudaT> {0.1, 0.2, 0.3, 0.63998, 1.0}, std::vector<cudaT> {0.553988, 1.0, 2.0}};
    VisualizationParameters visualization_parameters(theory, n_branches, lambda_ranges, fix_lambdas);

    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {0, 0 ,0 ,0 ,0}});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    visualization_parameters.write_to_file("visualization"); */

    //[ ThreePointSystem

    // Visualization

    // Todo: Second!! ./FRGVisualization three_point_system visualization

    /* const std::string theory = "three_point_system";
    const std::vector<int> n_branches {20, 20, 1};
    const std::vector <std::pair<cudaT, cudaT> > partial_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.3, 0.9), std::pair<cudaT, cudaT> (-1.0, 0.4)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {0.0187308, 0, -0.164471} };

    VisualizationParameters visualization_parameters(theory, n_branches, partial_lambda_ranges, fix_lambdas);

    visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {std::vector<cudaT> {0.299241730826242, -0.508756268365042, 0.0187309537615095}, std::vector<cudaT> {0.570243002084585, -0.164092840047983, -0.164467312739446}, std::vector<cudaT> {6.29425e-08, -1.71661e-07, 3.43323e-07 }});
    // visualization_parameters.append_fixed_point_parameters(std::vector < std::vector<cudaT> > {}, "/data/", true, "fixed_point_search");

    const bool skip_fix_lambdas = false;
    const bool with_vertices = true;
    VisualizationParameters::ComputeVertexVelocitiesParameters compute_vertex_velocities_parameters(skip_fix_lambdas, with_vertices);
    visualization_parameters.append_parameters(compute_vertex_velocities_parameters);

    const uint N_per_eigen_dim = 20;
    const std::vector<double> shift_per_dim {0.001, 0.001, 0.01};
    VisualizationParameters::ComputeSeparatrizesParameters compute_separatrizes_parameters(N_per_eigen_dim, shift_per_dim);
    visualization_parameters.append_parameters(compute_separatrizes_parameters);

    const std::vector <std::pair<cudaT, cudaT> > boundary_lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
            std::pair<cudaT, cudaT> (-0.3, 0.9), std::pair<cudaT, cudaT> (-1.0, 0.4), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::vector <cudaT > minimum_change_of_state = std::vector< cudaT > {1e-10, 1e-10, 1e-10};
    const cudaT minimum_delta_t = 0.000001;
    const cudaT maximum_flow_val = 1e8;
    const std::vector <cudaT > vicinity_dimensions = std::vector< cudaT > {1e-2};
    ConditionalIntersectionObserverParameters conditional_observer_parameters(boundary_lambda_ranges, minimum_change_of_state, minimum_delta_t, maximum_flow_val, vicinity_dimensions);
    visualization_parameters.append_parameters(conditional_observer_parameters);

    const uint observe_every_nth_step = 200;
    const uint maximum_total_number_of_steps = 1000000;
    CoordinateOperatorParameters::EvolveOnConditionParameters evolve_with_observer_parameters(observe_every_nth_step, maximum_total_number_of_steps);
    visualization_parameters.append_parameters(evolve_with_observer_parameters);

    visualization_parameters.write_to_file("visualization"); */
    //]




// Visualization

/* const std::string theory = "identity";

const std::vector<int> n_branches{1, 1, 8, 8, 1};
const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-1.0, 1.0), std::pair<cudaT, cudaT> (-2.0, 2.0)};
const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > {
        std::vector<cudaT> {-1, -0.2, 0.2, 1}, std::vector<cudaT> {-5, 5}, std::vector<cudaT> {1}};

VisualizationParameters visualization_parameters(theory, n_branches, lambda_ranges, fix_lambdas);
visualization_parameters.write_to_file("visualization");

Visualization visualization(visualization_parameters);

visualization.compute_vertex_velocities(number_of_cubes_per_gpu_call, false, true); */



    //[ Visualization of the three point system

    /* const uint dim = 3; // gn, lam3, mh2 -> g, lambda, mu
    const cudaT k = 1;
    const std::vector<int> n_branches {100, 100, 1};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.2, 0.7), std::pair<cudaT, cudaT> (-0.8, 0.4)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {0.0187308, -0.164471} };
    const std::string theory = "three_point_system";

    VisualizationParameters visualization_parameters(dim, k, theory, n_branches, lambda_ranges, fix_lambdas);
    Visualization visualization(visualization_parameters);

    visualization.compute_vertex_velocities(number_of_cubes_per_gpu_call, false, true);

    visualization_parameters.write_to_file("./dat/", "");*/

    //]

    //[ Visualization of the four point system

    /* const uint dim = 5;
    const cudaT k = 1;
    const std::vector<int> n_branches {200, 200, 1, 1, 1};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-3.0, 2.0), std::pair<cudaT, cudaT> (-2.0, 1.0)};
    const std::vector <std::vector <cudaT> > fix_lambdas = std::vector< std::vector<cudaT> > { std::vector<cudaT> {-0.1082}, std::vector<cudaT> {0.63998}, std::vector<cudaT> {0.553988}};
    const std::string theory = "four_point_system";

    VisualizationParameters visualization_parameters(dim, k, theory, n_branches, lambda_ranges, fix_lambdas);
    Visualization visualization(visualization_parameters);

    visualization.compute_vertex_velocities(number_of_cubes_per_gpu_call, false, true);

    visualization_parameters.write_to_file("./dat/", ""); */

    //]



    // auto * jacobian_equations = new IdentityJacobianEquations(dim);
    // auto * jacobian_equations = new ThreePointSystemJacobianEquations(k);


    //[ Identity

    /* const uint dim = 3;
    const cudaT k = 1;
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {6, 6, 6}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};
    // const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (0.1, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.1, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::string theory = "identity"; */

    //]

    //[ Three point system

    /* const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {10, 10, 10}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}, std::vector<int> {2, 2, 2}};

    // g, lambda, mu
    const std::vector < std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-1.0001, 3.0), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::string theory = "three_point_system"; */

    //]

    //[ Four point system

    /* const uint dim = 5;
    const cudaT k = 1;
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {6, 6, 6, 6, 6}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}};
    // const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (0.1, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9), std::pair<cudaT, cudaT> (-0.9, 0.9)};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {std::pair<cudaT, cudaT> (-0.101, 0.901), std::pair<cudaT, cudaT> (-0.901, 0.801), std::pair<cudaT, cudaT> (-0.901, 0.801), std::pair<cudaT, cudaT> (0.01, 2.0), std::pair<cudaT, cudaT> (0.01, 2.0)};
    const std::string theory = "four_point_system"; */

    //]


    /* const uint dim = 5;
    const cudaT k = 1;
    const int maximum_recursion_depth = 18;
    const std::vector< std::vector<int> > n_branches_per_depth = std::vector< std::vector<int> > {std::vector<int> {3, 4, 5, 6, 7}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}, std::vector<int> {2, 2, 2, 2, 2}};
    const std::vector <std::pair<cudaT, cudaT> > lambda_ranges = std::vector <std::pair<cudaT, cudaT> > {
        std::pair<cudaT, cudaT> (-0.3, 0.1),
        std::pair<cudaT, cudaT> (0.5, 0.7),
        std::pair<cudaT, cudaT> (0.5, 0.6),
        std::pair<cudaT, cudaT> (0.04, 0.1),
        std::pair<cudaT, cudaT> (-0.05, -0.15)};
    auto * flow_equations = new FourPointSystemFlowEquations(k);
    auto * jacobian_equations = new FourPointSystemJacobianEquations(k); */

    // Simple Test for the evaluation of the flow equation for a single vertex

    /* auto * flow_equations = new FourPointSystemFlowEquations(k);
    // auto * flow_equations = new ThreePointSystemFlowEquations(k);

    FlowEquationEvaluator flow_equation_evaluator(flow_equations);

    // dev_vec vertex(std::vector< cudaT >{-0.226228, 0.63998, 0.553988, -0.0603059, -0.1082});
    dev_vec vertex(std::vector< cudaT >{0.1, 0.2, 0.3 , 0.4, 0.5}); // mu, Lam3, Lam4, g3, g4
    dev_vec vertex_velocity = flow_equation_evaluator(vertex);

    print_range("Vertex", vertex_velocity.begin(), vertex_velocity.end()); */

    // Computation of the fix points

    /* std::string dir = "test";
    FixedPointSearchParameters fixed_point_search_parameters(theory, maximum_recursion_depth, n_branches_per_depth, lambda_ranges);
    fixed_point_search_parameters.write_to_file(dir);

    FixedPointSearch fixed_point_search(fixed_point_search_parameters);

    //std::cout << 100*100*100*100.0/(40*40*40*40) << std::endl;

    fixed_point_search.find_fixed_point_solutions();

    Counter<Collection>::print_statistics();

    // Just for testing issues
    std::vector<Leaf> solutions = fixed_point_search.get_solutions();

    fixed_point_search.write_solutions_to_file(dir);
    fixed_point_search.load_solutions_from_file(dir);

    // Consider solutions
    fixed_point_search.cluster_solutions(dir); */

    // auto * jacobian_equations = new IdentityJacobianEquations(k);


    /* for(auto dim_index = 0; dim_index < actual_fixed_points.size(); dim_index++)
        print_range("Found fix points in dimnension " + std::to_string(dim_index + 1), actual_fixed_points[dim_index]->begin(), actual_fixed_points[dim_index]->end()); */

    /* dev_vec* aligned_device_data_ptr = align_device_data(actual_fixed_points);
    auto N = aligned_device_data_ptr->size();
    aligned_device_data_ptr->resize(2 * N);
    thrust::fill( aligned_device_data_ptr->begin() + N , aligned_device_data_ptr->begin() + 2 * N , 1.0);

    print_range("Aligned device data", aligned_device_data_ptr->begin(), aligned_device_data_ptr->end());*/


    // Compute Jacobians and Eigenvectors/Eigenvalues

    /* Jacobians jacobians(actual_fixed_points);
    jacobians.compute_jacobians_and_eigendata(jacobian_equations);

    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType eigenvector;
    Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType eigenvalue;
    jacobians.get_eigenvalue_and_eigenvector(eigenvalue, eigenvector, 0);
    dev_vec initial_point = jacobians.get_coordinate(0);
    std::cout << "Size" << initial_point.size() << std::endl;

    std::cout << eigenvector.col(0) << std::endl;

    Eigen::VectorXcd first_eigenvector = eigenvector.col(0);

    std::vector< std::complex<double> > first_eigenvec(first_eigenvector.data(), first_eigenvector.data() + first_eigenvector.size());

    for(auto x : first_eigenvec)
        std::cout << x << std::endl;

    print_range("Vertex", initial_point.begin(), initial_point.end());

    double eps = 0.1;
    thrust::transform(initial_point.begin(), initial_point.end(), first_eigenvec.begin(), initial_point.begin(), [eps] __host__ __device__  (const cudaT &val1, const std::complex<double> &val2) { return val1 + eps*val2.real(); });

    print_range("Vertex", initial_point.begin(), initial_point.end());

    int driver_version , runtime_version;
    cudaDriverGetVersion( &driver_version );
    cudaRuntimeGetVersion ( &runtime_version );
    std::cout << driver_version << "\t" << runtime_version << std::endl;

    // create error stepper, can be used with make_controlled or make_dense_output
    // typedef boost::numeric::odeint::runge_kutta_dopri5< std::vector<dev_vec* >, cudaT , std::vector<dev_vec* >, cudaT > StepperType;
    typedef boost::numeric::odeint::runge_kutta4< dev_vec, cudaT , dev_vec, cudaT > StepperType;*/


    // dev_vec results(actual_fixed_points.size()*actual_fixed_points[0]->size());

    /*for(auto dim_index = 0; dim_index < dim; dim_index++)
        results[dim_index] = new dev_vec(actual_fixed_points[0]->size(), 0);*/

    /* cudaT t = 0.0;

    syst system(flow_equations, dim);
    StepperType rk;
    rk.do_step(system, *aligned_device_data_ptr, t, t + 0.1);

    print_range("Aligned device data after first step", aligned_device_data_ptr->begin(), aligned_device_data_ptr->end());

    rk.do_step(system, *aligned_device_data_ptr, t, t + 0.1);

    print_range("Aligned device data after first step", aligned_device_data_ptr->begin(), aligned_device_data_ptr->end());*/

    // calculate the Lyapunov exponents -- the main loop
    /*double t = 0.0;
    while( t < 10000.0 )
    {
        boost::numeric::odeint::integrate_adaptive( make_controlled( 1.0e-6 , 1.0e-6 , StepperType() ) , lorenz ,  , t , t + 1.0 , 0.1 );
        t += 1.0;
        obs( x , t );
    }*/
}

// Vertex: -5.46776 -1.61825 6.79708

/*
 * 	Depth: 17
	Cube indices: 551 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
Fixed point: 3.7998e-07 7.15256e-07 7.15256e-07
 */