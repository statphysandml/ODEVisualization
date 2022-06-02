#include <odesolver/evolution/evolution_observer.hpp>

namespace odesolver {
    namespace evolution {
    
        FlowObserver::FlowObserver(const json params) : Parameters(params),
            valid_coordinates_mask_ptr_(std::make_shared<dev_vec_bool>())
        {}

        void FlowObserver::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {}
        
        void FlowObserver::initialize(const odesolver::DevDatC &coordinates, cudaT t)
        {
            this->valid_coordinates_mask_ptr_->resize(coordinates.n_elems());
            thrust::fill(this->valid_coordinates_mask_ptr_->begin(), this->valid_coordinates_mask_ptr_->end(), true);
        }

        std::string FlowObserver::name()
        {
            return "flow_observer";
        }

        dev_vec_bool& FlowObserver::valid_coordinates_mask()
        {
            return *valid_coordinates_mask_ptr_;
        }

        const dev_vec_bool FlowObserver::valid_coordinates_mask() const
        {
            return *valid_coordinates_mask_ptr_;
        }

        int FlowObserver::n_valid_coordinates() const
        {
            return thrust::count(valid_coordinates_mask_ptr_->begin(), valid_coordinates_mask_ptr_->end(), true);
        }

        bool FlowObserver::valid_coordinates() const
        {
            if(n_valid_coordinates() == 0)
                return false;
            else
                return true;
        }

        // Changes conditional_mask to false if conditions is false
        void FlowObserver::update_valid_coordinates(dev_vec_bool &conditions)
        {
            thrust::transform_if(conditions.begin(), conditions.end(), valid_coordinates_mask_ptr_->begin(), valid_coordinates_mask_ptr_->begin(),
            [] __host__ __device__ (const bool &status) { return status; }, thrust::identity<bool>());
        }

        DivergentFlow::DivergentFlow(
            const json params, std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr
        ) : FlowObserver(params),
            flow_equations_ptr_(flow_equations_ptr),
            maximum_abs_flow_val_(get_entry<cudaT>("maximum_abs_flow_val"))
        {}
        
        DivergentFlow DivergentFlow::generate(
                std::shared_ptr<odesolver::flowequations::FlowEquationsWrapper> flow_equations_ptr,
                const cudaT maximum_abs_flow_val
        )
        {
            return DivergentFlow(
                json {{"maximum_abs_flow_val", maximum_abs_flow_val}},
                flow_equations_ptr
            );
        }

        void DivergentFlow::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            // Check if evaluated flow is too large - makes use of maximum_abs_flow_val
            auto flow = compute_flow(coordinates, flow_equations_ptr_.get());
            auto maximum_abs_flow_val = maximum_abs_flow_val_;
            for(auto dim_index = 0; dim_index < coordinates.dim_size(); dim_index++) {
                thrust::transform_if(
                    flow[dim_index].begin(),
                    flow[dim_index].end(),
                    this->valid_coordinates_mask_ptr_->begin(),
                    [] __host__ __device__(const cudaT &val) { return false; },
                    [maximum_abs_flow_val] __host__ __device__(const cudaT &val) { return abs(val) > maximum_abs_flow_val; });
            }
        }

        NoChange::NoChange(
            const json params
        ) : FlowObserver(params),
            minimum_change_of_state_(get_entry<std::vector<cudaT>>("minimum_change_of_state"))
        {}
        
        NoChange NoChange::generate(
            const std::vector<cudaT> minimum_change_of_state
        )
        {
            return NoChange(
                json {{"minimum_change_of_state", minimum_change_of_state}}
            );
        }

        struct compare_to_previous_change
        {
            compare_to_previous_change(const cudaT minimum_change_of_state_in_dim) :
                minimum_change_of_state_in_dim_(minimum_change_of_state_in_dim)
            {}

            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                cudaT current_coor = thrust::get<0>(t);
                cudaT prev_coor = thrust::get<1>(t);
                bool status  = thrust::get<2>(t);
                // Check if coor remains valid; last condition is to prevent a stopping before the first step with the boost integrate functions
                if(status or abs(current_coor - prev_coor) > minimum_change_of_state_in_dim_ or current_coor == prev_coor)
                    thrust::get<3>(t) = true;
                else
                    thrust::get<3>(t) = false;
            }

            const cudaT minimum_change_of_state_in_dim_;
        };

        void NoChange::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            detected_changes_.resize(coordinates.n_elems());
            
            // Reset detected changes to false
            thrust::fill(detected_changes_.begin(), detected_changes_.end(), false);

            // Check if change in variable is large enough - makes use of minimum_change_in_state
            for(auto dim_index = 0; dim_index < minimum_change_of_state_.size(); dim_index++) {
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        coordinates[dim_index].begin(),
                        previous_coordinates_[dim_index].begin(),
                        detected_changes_.begin(),
                        detected_changes_.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        coordinates[dim_index].end(),
                        previous_coordinates_[dim_index].end(),
                        detected_changes_.end(),
                        detected_changes_.end())),
                    compare_to_previous_change(minimum_change_of_state_[dim_index])
                );
            }

            // Detected changes contains true for valid coordinates and false for coordinates with no change
            this->update_valid_coordinates(detected_changes_);

            previous_coordinates_ = coordinates;
        }

        void NoChange::initialize(const odesolver::DevDatC &coordinates, cudaT t)
        {
            FlowObserver::initialize(coordinates, t);
            previous_coordinates_ = coordinates;
        }

        OutOfRangeCondition::OutOfRangeCondition(
            const json params
        ) : FlowObserver(params),
            variable_ranges_(odesolver::util::json_to_vec_pair(get_entry<json>("variable_ranges"))),
            observed_dimension_indices_(get_entry<std::vector<int>>("observed_dimension_indices"))
        {
            if(observed_dimension_indices_.size() == 0)
            {
                observed_dimension_indices_ = std::vector<int>(variable_ranges_.size(), 0);
                std::generate(observed_dimension_indices_.begin(), observed_dimension_indices_.end(), [n = 0] () mutable { return n++; });
            }
            if(observed_dimension_indices_.size() != variable_ranges_.size())
            {
                std::cerr << "error: observed_dimension_indices must be either empty or to be of the same size as variable_ranges" << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        
        OutOfRangeCondition OutOfRangeCondition::generate(
            const std::vector<std::pair<cudaT, cudaT>> variable_ranges,
            const std::vector<int> observed_dimension_indices
        )
        {
            return OutOfRangeCondition(
                json {
                    {"variable_ranges", variable_ranges},
                    {"observed_dimension_indices", observed_dimension_indices}
                }
            );
        }

        void OutOfRangeCondition::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            out_of_range_.resize(coordinates.n_elems());
            
            // Reset out_of_range to true
            thrust::fill(out_of_range_.begin(), out_of_range_.end(), true);

            // Check if coordinates are in variable range -> change to false, if not
            for(auto i = 0; i < observed_dimension_indices_.size(); i++) {
                auto variable_range = variable_ranges_[i];
                thrust::transform_if(
                    out_of_range_.begin(),
                    out_of_range_.end(),
                    coordinates[observed_dimension_indices_[i]].begin(),
                    out_of_range_.begin(),
                    [] __host__ __device__ (const bool &status) { return false; },
                    [variable_range] __host__ __device__ (const cudaT &val) {
                        return val < variable_range.first or val > variable_range.second;
                    }
                );
            }
            
            // True for valid coordinates and false for coordinates with out of range
            this->update_valid_coordinates(out_of_range_);
        }


        Intersection::Intersection(
            const json params
        ) : FlowObserver(params),
            vicinity_distances_(get_entry<std::vector<cudaT>>("vicinity_distances")),
            fixed_variables_(get_entry<std::vector<cudaT>>("fixed_variables")),
            fixed_variable_indices_(get_entry<std::vector<int>>("fixed_variable_indices")),
            remember_intersections_(get_entry<bool>("remember_intersections")),
            intersection_counter_ptr_(std::make_shared<dev_vec_int>()),
            detected_intersections_ptr_(std::make_shared<std::vector<std::vector<cudaT>>>()),
            detected_intersections_types_ptr_(std::make_shared<std::vector<bool>>())
        {}
        
        Intersection Intersection::generate(
            const std::vector<cudaT> vicinity_distances,
            const std::vector<cudaT> fixed_variables,
            const std::vector<int> fixed_variable_indices,
            const bool remember_intersections
        )
        {
            return Intersection(
                json {
                    {"vicinity_distances", vicinity_distances},
                    {"fixed_variables", fixed_variables},
                    {"fixed_variable_indices", fixed_variable_indices},
                    {"remember_intersections", remember_intersections}
                }
            );
        }
        
        struct check_for_intersection
        {
            check_for_intersection(const cudaT fixed_variable, const cudaT vicinity_distance) :
                    fixed_variable_(fixed_variable), vicinity_distance_(vicinity_distance)
            {}
        
            template <typename Tuple>
            __host__ __device__
            void operator()(Tuple t)
            {
                cudaT current_coor = thrust::get<0>(t);
                cudaT prev_coor = thrust::get<1>(t);
                
                bool intersection = ((prev_coor < fixed_variable_) and (fixed_variable_ < current_coor)) or ((current_coor < fixed_variable_) and (fixed_variable_ < prev_coor));
                bool vicinity = (abs(current_coor - fixed_variable_) < vicinity_distance_) and (current_coor != prev_coor);

                if(!intersection)
                    thrust::get<2>(t) = false; // no intersection
                if(!vicinity)
                    thrust::get<3>(t) = false; // not in vicinity
            }
        
            const cudaT fixed_variable_;
            const cudaT vicinity_distance_;
        };

        void Intersection::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            // Reset true
            thrust::fill(intersections_.begin(), intersections_.end(), true);
            thrust::fill(vicinities_.begin(), vicinities_.end(), true);
            thrust::fill(intersections_and_vicinities_.begin(), intersections_and_vicinities_.end(), true);

            // Check wether intersection criteria hold in each dimension (resulting in true and, otherwise, false)
            for(auto i = 0; i < fixed_variable_indices_.size(); i++) {
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(
                        coordinates[fixed_variable_indices_[i]].begin(),
                        previous_coordinates_[fixed_variable_indices_[i]].begin(),
                        intersections_.begin(),
                        vicinities_.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        coordinates[fixed_variable_indices_[i]].end(),
                        previous_coordinates_[fixed_variable_indices_[i]].end(),
                        intersections_.end(),
                        vicinities_.end())),
                    check_for_intersection(fixed_variables_[i], vicinity_distances_[i])
                );
            }
            
            thrust::transform(intersections_.begin(), intersections_.end(), vicinities_.begin(), intersections_and_vicinities_.begin(),
                [] __host__ __device__ (const bool &intersection, const bool &vicinity) { return intersection or vicinity; });

            int n_intersections_and_vicinities = thrust::count(intersections_and_vicinities_.begin(), intersections_and_vicinities_.end(), true);

            // Store intersections
            if(remember_intersections_ and n_intersections_and_vicinities > 0)
            {
                // Copy intersection types to DevDat
                intersection_type_.resize(coordinates.n_elems());

                thrust::copy_if(intersections_.begin(), intersections_.end(), intersections_and_vicinities_.begin(), intersection_type_.begin(), thrust::identity<bool>()); // true = actual intersection, false = no actual intersection, but vicinity

                auto n_total = detected_intersections_types_ptr_->size();

                // Copy intersection types to std::vector
                detected_intersections_types_ptr_->resize(n_total + n_intersections_and_vicinities);
                thrust::copy(intersection_type_.begin(), intersection_type_.end(), detected_intersections_types_ptr_->begin() + n_total);

                // Copy intersection coordinates to dev_vec
                flattened_intersection_coordinates_.resize(n_intersections_and_vicinities * coordinates.dim_size());

                for(auto dim_index = 0; dim_index < coordinates.dim_size(); dim_index++) {
                    thrust::copy_if(coordinates[dim_index].begin(), coordinates[dim_index].end(), intersections_and_vicinities_.begin(), flattened_intersection_coordinates_.begin() + dim_index * n_intersections_and_vicinities, thrust::identity<bool>());
                }

                // Copy intersection coordinates to std::vector (by transposing and conversion to std::vector)
                DevDatC intersection_coordinates(flattened_intersection_coordinates_, coordinates.dim_size());
                auto transposed_intersection_coordinates = intersection_coordinates.transposed();

                detected_intersections_ptr_->resize(n_total + n_intersections_and_vicinities, std::vector<cudaT>(coordinates.dim_size()));

                for(auto n = 0; n < n_intersections_and_vicinities; n++)
                {
                    thrust::copy(transposed_intersection_coordinates[n].begin(), transposed_intersection_coordinates[n].end(), (*detected_intersections_ptr_)[n_total + n].begin());
                }
            }

            // Update intersection counter
            intersection_counter_ptr_->resize(coordinates.n_elems(), 0);
            thrust::transform_if(intersection_counter_ptr_->begin(), intersection_counter_ptr_->end(), intersections_.begin(), intersection_counter_ptr_->begin(), [] __host__ __device__ (const int &counter) { return counter + 1; }, thrust::identity<bool>());

            previous_coordinates_ = coordinates;
        }
        
        void Intersection::initialize(const odesolver::DevDatC &coordinates, cudaT t)
        {
            FlowObserver::initialize(coordinates, t);

            intersections_.resize(coordinates.n_elems());
            vicinities_.resize(coordinates.n_elems());
            intersections_and_vicinities_.resize(coordinates.n_elems());

            previous_coordinates_ = coordinates;
        }

        bool Intersection::valid_coordinates() const
        {
            return true;
        }
        
        dev_vec_int Intersection::intersection_counter() const
        {
            return *intersection_counter_ptr_;
        }

        std::vector<std::vector<cudaT>> Intersection::detected_intersections() const
        {
            return *detected_intersections_ptr_;
        }

        std::vector<bool> Intersection::detected_intersection_types() const
        {
            return *detected_intersections_types_ptr_;
        }


        TrajectoryObserver::TrajectoryObserver(
            const json params
        ) : FlowObserver(params),
            os_(get_entry<std::string>("file"))
        {}
        
        TrajectoryObserver TrajectoryObserver::generate(
            std::string file
        )
        {
            return TrajectoryObserver(
                json {{"file", file}}
            );
        }

        void TrajectoryObserver::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            os_ << t << " ";
            print_range_in_os(coordinates.begin(), coordinates.end(), os_);
            os_ << std::endl;
        }
        
        bool TrajectoryObserver::valid_coordinates() const
        {
            return true;
        }


        EvolutionObserver::EvolutionObserver(
            const json params,
            const std::vector<std::shared_ptr<FlowObserver>> observers
        ) : FlowObserver(params),
            observers_(observers)
        {}

        EvolutionObserver EvolutionObserver::generate(
            const std::vector<std::shared_ptr<FlowObserver>> observers
        )
        {
            return EvolutionObserver(
                json {},
                observers
            );
        }

        void EvolutionObserver::operator() (const odesolver::DevDatC &coordinates, cudaT t)
        {
            for(auto observer : observers_)
            {
                (*observer)(coordinates, t);
                auto conditions = observer->valid_coordinates_mask();
                this->update_valid_coordinates(conditions);
                if(!this->valid_coordinates())
                    throw StopEvolutionException(t, observer->name());
            }
        }

        void EvolutionObserver::initialize(const odesolver::DevDatC &coordinates, cudaT t)
        {
            FlowObserver::initialize(coordinates, t);
            for(auto observer : observers_)
            {
                observer->initialize(coordinates, t);
            }
        }
    }
}