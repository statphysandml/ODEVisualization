#include <odesolver/modes/jacobians.hpp>
    
using json = nlohmann::json;


namespace odesolver {
    namespace modes {
        
        Jacobians Jacobians::from_devdat(devdat::DevDatC jacobian_elements)
        {
            auto transposed_elements = jacobian_elements.transposed();
            std::vector<double> jacobians(jacobian_elements.size());
            thrust::copy(transposed_elements.begin(), transposed_elements.end(), jacobians.begin());
            return Jacobians(jacobians, int(std::sqrt(jacobian_elements.dim_size())));
        }
    }
}
