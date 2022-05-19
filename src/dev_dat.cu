#include <odesolver/dev_dat.hpp>


namespace odesolver {
    void write_data_to_ofstream(const DevDatC &data, std::ofstream &os, std::vector<int> skip_iterators_in_dimensions, std::vector< dev_iterator > end_iterators)
    {
        std::vector< const_dev_iterator > data_iterators;
        for(auto i = 0; i < data.dim_size(); i++)
        {
            if(std::find(skip_iterators_in_dimensions.begin(), skip_iterators_in_dimensions.end(), i) == skip_iterators_in_dimensions.end())
                data_iterators.push_back(data[i].begin());
        }

        auto data_num = data.n_elems();
        if(end_iterators.size() > 0)
            data_num = end_iterators[0] - data[0].begin();

        // Data is written transposed to file by incrementing the data iterators in each step
        for(auto i = 0; i < data_num; i++)
        {
            for(auto &data_iter: data_iterators)
            {
                os << *data_iter << " ";
                data_iter++;
            }
            os << std::endl;
        }
    }

    void write_devdat_to_file(DevDatC &data, std::string rel_dir, std::string filename)
    {
        auto transposed_data = data.transpose_device_data();
        param_helper::fs::write_parameter_file(
            json {odesolver::util::vec_vec_to_json(transposed_data)},
            param_helper::proj::project_root() + "/" + rel_dir + "/", filename,
            false
        );
    }

    DevDatC load_devdat(std::string rel_dir, std::string filename)
    {
        json j = param_helper::fs::read_parameter_file(param_helper::proj::project_root() + rel_dir + "/", filename, false);
        return DevDatC(odesolver::util::json_to_vec_vec(j.get<std::vector<std::vector<cudaT>>>()));
    }
}