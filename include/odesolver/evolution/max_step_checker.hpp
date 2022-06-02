
namespace odesolver {
    namespace maxstepchecker {
        class MaxStepChecker {
        public:
            // construct/copy/destruct
            MaxStepChecker(const int = 500);

            // public member functions
            void reset();
            void operator()(void);
        };
    }
}
