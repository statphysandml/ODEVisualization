#include <odesolver/evolution/stepper.hpp>


namespace odesolver {
    namespace evolution {
        namespace stepper {

            RungaKutta4::RungaKutta4() : stepper_(stepper_type())
            {}

            SymplecticRKNSB3McLachlan::SymplecticRKNSB3McLachlan() : stepper_(stepper_type())
            {}

            RungaKuttaDopri5::RungaKuttaDopri5() : stepper_(stepper_type())
            {}
        }
    }
}