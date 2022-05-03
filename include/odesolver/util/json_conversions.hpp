#ifndef PROGRAM_JSON_CONVERSION_HPP
#define PROGRAM_JSON_CONVERSION_HPP

#include <vector>
#include <algorithm>

#include <param_helper/json.hpp>

using json = nlohmann::json;


template<typename T=double>
std::vector<std::vector<T>> json_to_vec_vec(const json j)
{
    std::vector< std::vector<T> > data;
    data.reserve(j.size());
    std::transform(j.begin(), j.end(), std::back_inserter(data),
        [] (const json &dat) { return dat.get<std::vector<T>>(); });
    return data;
}

template<typename T>
json vec_vec_to_json(const std::vector<std::vector<T>> data)
{
    json j;
    for(const auto &dat : data)
    {
/*         std::vector<double> vec(dat.size());
        std::copy(dat.begin(), dat.end(), vec.begin()); */
        j.push_back(dat);
    }
    return j;
}

#endif //PROGRAM_JSON_CONVERSION_HPP