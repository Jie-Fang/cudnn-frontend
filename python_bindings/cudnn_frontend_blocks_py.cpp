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

PYBIND11_MODULE(cudnn_frontend_blocks, m)
{
  init_pygraph_submodule(m);
  init_properties(m);

  auto execute_wrapper = [](Graph * graph, std::unordered_map<std::string, int64_t> var_pack) -> cudnn_frontend_error_t {
      std::unordered_map<std::string, void *> var_pack_;
      for (auto item : var_pack) {
          var_pack_.insert(std::make_pair(item.first, (void *)item.second));
      }
      graph->execute(var_pack_);
      return cudnn_frontend_error_t::OK;
  };

  py::class_<Graph>(m, "Graph")
    .def(py::init<std::string const &, cuDNNFEContext const &>())
    .def("add_tensor", static_cast<cudnn_frontend_error_t (Graph::*)(tensor_properties const &)>(&Graph::add_tensor))
    .def("add_node", static_cast<cudnn_frontend_error_t (Graph::*)(convolution_node const &)>(&Graph::add_node))
    .def("add_node", static_cast<cudnn_frontend_error_t (Graph::*)(pointwise_node const &)>(&Graph::add_node))
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

  py::class_<convolution_node> convolution_node(m, "convolution_node");
  convolution_node.def(py::init<std::string const &>())
    .def("get_padding",  &convolution_node::get_padding)
    .def("set_padding",  &convolution_node::set_padding)
    .def("get_stride",   &convolution_node::get_stride)
    .def("set_stride",   &convolution_node::set_stride)
    .def("get_dilation", &convolution_node::get_dilation)
    .def("set_dilation", &convolution_node::set_dilation)
    .def("get_tensor_data_type", &Node::get_tensor_data_type)
    .def("set_tensor_data_type", &Node::set_tensor_data_type)
    .def("get_compute_type",     &Node::get_compute_type)
    .def("set_compute_type",     &Node::set_compute_type)
    .def("get_port_name",       &convolution_node::get_port_name)
    .def("set_port_names",       &convolution_node::set_port_names)
    ;

  py::enum_<convolution_node::PORTS>(m, "convolution_ports")
        .value("X", convolution_node::PORTS::X)
        .value("W", convolution_node::PORTS::W)
        .value("Y", convolution_node::PORTS::Y)
        .export_values();


  py::class_<pointwise_node> pointwise_node(m, "pointwise_node");
  pointwise_node.def(py::init<std::string const &>())
    .def("get_mode", &pointwise_node::get_mode)
    .def("set_mode", static_cast<int (pointwise_node::*)(std::string)>(&pointwise_node::set_mode))
    .def("get_tensor_data_type",  &Node::get_tensor_data_type)
    .def("set_tensor_data_type",  &Node::set_tensor_data_type)
    .def("get_compute_type",      &Node::get_compute_type)
    .def("set_compute_type",      &Node::set_compute_type)
    .def("get_port_name",       &pointwise_node::get_port_name)
    .def("set_port_names",       &pointwise_node::set_port_names)
    ;


  py::enum_<pointwise_node::PORTS>(m, "pointwise_ports")
        .value("X", pointwise_node::PORTS::X)
        .value("B", pointwise_node::PORTS::B)
        .value("Y", pointwise_node::PORTS::Y)
        .export_values();



  py::enum_<cudnn_frontend_error_t>(m, "cudnn_frontend_error")
    .value("OK", cudnn_frontend_error_t::OK)
    .value("TENSOR_DIMENSIONS_NOT_SET", cudnn_frontend_error_t::TENSOR_DIMENSIONS_NOT_SET)
    .value("POINTWISE_MODE_NOT_SET", cudnn_frontend_error_t::POINTWISE_MODE_NOT_SET)
    .value("SHAPE_DEDUCTION_FAILED", cudnn_frontend_error_t::SHAPE_DEDUCTION_FAILED)
    .value("OUTPUT_TENSOR_NODE_NOT_FOUND", cudnn_frontend_error_t::OUTPUT_TENSOR_NODE_NOT_FOUND)
    .value("UNKNOWN_TENSOR_NAME", cudnn_frontend_error_t::UNKNOWN_TENSOR_NAME)
    .value("INPUT_PORT_COUNT_MISMATCH", cudnn_frontend_error_t::INPUT_PORT_COUNT_MISMATCH)
    ;

  py::class_<reduction_node> reduction_node(m, "reduction_node");
  reduction_node.def(py::init<std::string const &>())
    .def("get_mode", &reduction_node::get_mode)
    .def("set_mode", &reduction_node::set_mode)
    .def("get_tensor_data_type",  &reduction_node::get_tensor_data_type)
    .def("set_tensor_data_type",  &reduction_node::set_tensor_data_type)
    .def("get_port_name",       &reduction_node::get_port_name)
    .def("set_port_names",       &reduction_node::set_port_names)
    ;


  py::enum_<reduction_node::PORTS>(m, "reduction_ports")
        .value("X", reduction_node::PORTS::X)
        .value("Y", reduction_node::PORTS::Y)
        .export_values();
}