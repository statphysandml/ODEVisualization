#include "../../include/observers/evolution.hpp"

const std::map<std::string, typename EvolutionParameters::StepSizeType > EvolutionParameters::mode_resolver = {
        {"constant", constant},
        {"adaptive", adpative},
};