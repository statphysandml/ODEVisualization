#ifndef PROGRAM_JSON_CONVERSION_HPP
#define PROGRAM_JSON_CONVERSION_HPP

#include <vector>
#include <algorithm>

#include <param_helper/json.hpp>

using json = nlohmann::json;


std::vector<std::vector<double>> json_to_vec_vec(const json j);

json vec_vec_to_json(const std::vector <std::vector<double>> data);

#endif //PROGRAM_JSON_CONVERSION_HPP