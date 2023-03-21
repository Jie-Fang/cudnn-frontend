#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace cudnn_frontend;

// pybinds for pygraph class
void init_pygraph_submodule(py::module_ &);

// pybinds for all properties and helpers
void init_properties(py::module_ &);

PYBIND11_MODULE(pycudnn, m)
{
  init_pygraph_submodule(m);
  init_properties(m);

  py::class_<cuDNNFEContext>(m, "cuDNNFEContext")
    .def(py::init<>())
    .def("set_intermediate_data_type", &cuDNNFEContext::set_intermediate_data_type)
    .def("get_intermediate_data_type", &cuDNNFEContext::get_intermediate_data_type_string)
    .def("set_tensor_data_type", &cuDNNFEContext::set_tensor_data_type)
    .def("get_tensor_data_type", &cuDNNFEContext::get_tensor_data_type_string)
    .def("set_compute_type", &cuDNNFEContext::set_compute_type)
    .def("get_compute_type", &cuDNNFEContext::get_compute_type_string)
    .def("set_tensor_dims", &cuDNNFEContext::set_tensor_dims)
    .def("get_tensor_dims", &cuDNNFEContext::get_tensor_dims)
    .def("set_layout", &cuDNNFEContext::set_layout)
    .def("get_layout", &cuDNNFEContext::get_layout_string)
    .def("set_spatial_dims", &cuDNNFEContext::set_spatial_dims)
    .def("get_spatial_dims", &cuDNNFEContext::get_spatial_dims)
    .def("__repr__",      &cuDNNFEContext::describe);

  py::class_<reduction_properties> reduction_properties(m, "reduction_properties");
  reduction_properties.def(py::init<std::string const &>())
    .def("get_mode", &reduction_properties::get_mode)
    .def("set_mode", &reduction_properties::set_mode)
    .def("get_tensor_data_type",  &reduction_properties::get_tensor_data_type)
    .def("set_tensor_data_type",  &reduction_properties::set_tensor_data_type)
    .def("get_port_name",       &reduction_properties::get_port_name)
    .def("map_port_to_tensor",       &reduction_properties::map_port_to_tensor)
    ;


  py::enum_<reduction_properties::PORTS>(m, "reduction_ports")
        .value("X", reduction_properties::PORTS::X)
        .value("Y", reduction_properties::PORTS::Y)
        .export_values();
}