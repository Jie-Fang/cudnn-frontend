#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

void init_properties(py::module_ &m) {
    py::class_<cudnn_frontend::tensor_properties, std::shared_ptr<cudnn_frontend::tensor_properties>>(m, "tensor_properties")
        .def(py::init<std::string const &>())
        .def("get_data_type", &cudnn_frontend::tensor_properties::get_data_type)
        .def("set_data_type", &cudnn_frontend::tensor_properties::set_data_type)
        .def("get_dim", &cudnn_frontend::tensor_properties::get_dim)
        .def("set_dim", &cudnn_frontend::tensor_properties::set_dim)
        .def("get_stride", &cudnn_frontend::tensor_properties::get_stride)
        .def("set_stride", &cudnn_frontend::tensor_properties::set_stride)
        .def("get_is_virtual", &cudnn_frontend::tensor_properties::get_is_virtual)
        .def("set_is_virtual", &cudnn_frontend::tensor_properties::set_is_virtual)
        .def("get_is_pass_by_value", &cudnn_frontend::tensor_properties::get_is_pass_by_value)
        .def("set_is_pass_by_value", &cudnn_frontend::tensor_properties::set_is_pass_by_value)
        .def("get_uid", &cudnn_frontend::tensor_properties::get_uid)
        .def("set_uid", &cudnn_frontend::tensor_properties::set_uid)
        .def("get_size", &cudnn_frontend::tensor_properties::get_size)
        .def("__repr__", [](cudnn_frontend::tensor_properties const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });
    
    py::class_<cudnn_frontend::convolution_node> convolution_node(m, "convolution_node");
    convolution_node.def(py::init<std::string const &>())
        .def("get_padding",  &cudnn_frontend::convolution_node::get_padding)
        .def("set_padding",  &cudnn_frontend::convolution_node::set_padding)
        .def("get_stride",   &cudnn_frontend::convolution_node::get_stride)
        .def("set_stride",   &cudnn_frontend::convolution_node::set_stride)
        .def("get_dilation", &cudnn_frontend::convolution_node::get_dilation)
        .def("set_dilation", &cudnn_frontend::convolution_node::set_dilation)
        .def("get_tensor_data_type", &cudnn_frontend::Node::get_tensor_data_type)
        .def("set_tensor_data_type", &cudnn_frontend::Node::set_tensor_data_type)
        .def("get_compute_type",     &cudnn_frontend::Node::get_compute_type)
        .def("set_compute_type",     &cudnn_frontend::Node::set_compute_type)
        .def("get_port_name",       &cudnn_frontend::convolution_node::get_port_name)
        .def("set_port_names",       &cudnn_frontend::convolution_node::set_port_names)
        .def("__repr__", [](cudnn_frontend::convolution_node const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });;

    py::enum_<cudnn_frontend::convolution_node::PORTS>(convolution_node, "ports")
        .value("X", cudnn_frontend::convolution_node::PORTS::X)
        .value("W", cudnn_frontend::convolution_node::PORTS::W)
        .value("Y", cudnn_frontend::convolution_node::PORTS::Y)
        .export_values();
}