#include "../include/dev_dat_python.hpp"


namespace odesolver {
    namespace pybind {

        void init_devdat(py::module &m)
        {
            py::class_<DevDatC>(m, "DevDat")
                .def(py::init<size_t, size_t, cudaT>(), "dim"_a=3, "N"_a=1000, "init_val"_a=0.0)
                .def(py::init<std::vector<cudaT>, size_t>(), "data"_a, "dim"_a)
                .def(py::init<std::vector<std::vector<cudaT>>>(), "data"_a)
                .def("dim_size", &DevDatC::dim_size)
                .def("n_elems", &DevDatC::n_elems)
                .def("set_nth_element", &DevDatC::set_nth_element)
                .def("get_nth_element", &DevDatC::get_nth_element)
                .def("fill_dim", [] (DevDatC &devdat, unsigned dim, std::vector<cudaT> data, const int start_idx=0) { 
                    if(int(data.size()) - start_idx > int(devdat.n_elems()))
                    {
                        std::cerr << "error in fill_dim: data.size() - start_idx bigger than n_elems" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    thrust::copy(data.begin(), data.end(), devdat[dim].begin() + start_idx);
                })
                .def("data_in_dim", [] (DevDatC &devdat, unsigned dim, const int start_idx=0, const int end_idx=-1) {
                    int n_elems;
                    if(end_idx == -1)
                        n_elems = devdat.n_elems() - start_idx;
                    else
                        n_elems = end_idx - start_idx;
                    if(n_elems < 0)
                    {
                        std::cerr << "error in data_in_dim: end_idx needs to be bigger or equal to start_idx" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    else if(end_idx > int(devdat.n_elems()))
                    {
                        std::cerr << "error in data_in_dim: end_idx bigger than n_elems()" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    std::vector<cudaT> data(n_elems);
                    thrust::copy(devdat[dim].begin() + start_idx, devdat[dim].begin() + start_idx + n_elems, data.begin());
                    return data;
                })
                .def("reshape", &DevDatC::reshape)
                .def("resize", &DevDatC::resize)
                .def("to_vec_vec", &DevDatC::to_vec_vec)
                .def("transposed", &DevDatC::transposed)
                .def("transpose", &DevDatC::transpose)
                .def("write_to_file", &DevDatC::write_to_file);
        }
    }
}