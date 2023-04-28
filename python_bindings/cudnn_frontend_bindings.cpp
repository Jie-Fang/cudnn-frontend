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

bool
is_cudnn_supported() {
    return cudnnGetVersion() >= 8500;
}

PYBIND11_MODULE(pycudnn, m)
{
  init_pygraph_submodule(m);
  init_properties(m);

  m.def("is_cudnn_supported", &is_cudnn_supported);

  py::class_<cuDNNFEContext>(m, "cuDNNFEContext")
    .def(py::init<>())
    .def("set_intermediate_data_type", &cuDNNFEContext::set_intermediate_data_type)
    .def("set_tensor_data_type", &cuDNNFEContext::set_tensor_data_type)
    .def("get_tensor_data_type", &cuDNNFEContext::get_tensor_data_type)
    .def("set_compute_type", &cuDNNFEContext::set_compute_type)
    .def("get_compute_type", &cuDNNFEContext::get_compute_type)
    .def("set_tensor_dims", &cuDNNFEContext::set_tensor_dims)
    .def("get_tensor_dims", &cuDNNFEContext::get_tensor_dims)
    .def("set_layout", &cuDNNFEContext::set_layout)
    .def("get_layout", &cuDNNFEContext::get_layout_string)
    .def("set_spatial_dims", &cuDNNFEContext::set_spatial_dims)
    .def("get_spatial_dims", &cuDNNFEContext::get_spatial_dims);
}
