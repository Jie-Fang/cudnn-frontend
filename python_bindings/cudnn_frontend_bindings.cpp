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

  auto execute_wrapper = [](Graph * graph, std::unordered_map<std::string, int64_t> var_pack) -> cudnn_frontend::error_t {
      std::unordered_map<std::string, void *> var_pack_;
      for (auto item : var_pack) {
          var_pack_.insert(std::make_pair(item.first, (void *)item.second));
      }
      graph->execute(var_pack_);
      return cudnn_frontend::error_t::OK;
  };

  py::class_<Graph>(m, "Graph")
    .def(py::init<std::string const &, cuDNNFEContext const &>())
    .def("add_tensor", static_cast<cudnn_frontend::error_t (Graph::*)(tensor_properties const &)>(&Graph::add_tensor))
    .def("add_node", static_cast<cudnn_frontend::error_t (Graph::*)(convolution_properties const &)>(&Graph::add_node))
    .def("add_node", static_cast<cudnn_frontend::error_t (Graph::*)(pointwise_properties const &)>(&Graph::add_node))
    .def("infer_shapes", &Graph::infer_shapes)
    .def("is_valid_tensor", &Graph::is_valid_tensor)
    .def("tensor_at", &Graph::tensor_at, py::return_value_policy::reference)
    .def("build", &Graph::build)
    .def("execute", execute_wrapper)
    ;

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
    .def("set_port_names",       &reduction_properties::set_port_names)
    ;


  py::enum_<reduction_properties::PORTS>(m, "reduction_ports")
        .value("X", reduction_properties::PORTS::X)
        .value("Y", reduction_properties::PORTS::Y)
        .export_values();
}