#include <odesolver/util/random.hpp>


namespace odesolver {
    namespace util { 
        std::random_device g_rd;

        std::mt19937 g_gen(g_rd());

        void set_random_seed(uint32_t seed_val)
        {
            std::cout << "Setting random seed" << std::endl;
            g_gen.seed(seed_val);
        }
    }
}