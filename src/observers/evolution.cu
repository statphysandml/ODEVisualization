#include "../../include/observers/evolution.hpp"

const std::map<std::string, typename Evolution::StepSizeType > Evolution::mode_resolver = {
        {"constant", constant},
        {"adaptive", adpative},
};