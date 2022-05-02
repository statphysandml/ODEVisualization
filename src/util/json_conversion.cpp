#include "../../include/odesolver/util/json_conversions.hpp"

std::vector<std::vector<double>> json_to_vec_vec(const json j)
{
    std::vector< std::vector<double> > data;
    data.reserve(j.size());
    std::transform(j.begin(), j.end(), std::back_inserter(data),
        [] (const json &dat) { return dat.get<std::vector<double>>(); });
    return data;
}

json vec_vec_to_json(const std::vector <std::vector<double>> data)
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