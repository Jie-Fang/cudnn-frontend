#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

void init_properties(py::module_ &m) {
    py::class_<cudnn_frontend::graph::Tensor, std::shared_ptr<cudnn_frontend::graph::Tensor>>(m, "tensor")
        .def(py::init<std::string const &>())
        .def("get_name", &cudnn_frontend::graph::Tensor::get_name)
        .def("get_data_type", &cudnn_frontend::graph::Tensor::get_data_type)
        .def("set_data_type", &cudnn_frontend::graph::Tensor::set_data_type)
        .def("get_dim", &cudnn_frontend::graph::Tensor::get_dim)
        .def("set_dim", &cudnn_frontend::graph::Tensor::set_dim)
        .def("get_stride", &cudnn_frontend::graph::Tensor::get_stride)
        .def("set_stride", &cudnn_frontend::graph::Tensor::set_stride)
        .def("get_is_virtual", &cudnn_frontend::graph::Tensor::get_is_virtual)
        .def("set_is_virtual", &cudnn_frontend::graph::Tensor::set_is_virtual)
        .def("get_is_pass_by_value", &cudnn_frontend::graph::Tensor::get_is_pass_by_value)
        .def("set_is_pass_by_value", &cudnn_frontend::graph::Tensor::set_is_pass_by_value)
        .def("get_uid", &cudnn_frontend::graph::Tensor::get_uid)
        .def("set_uid", &cudnn_frontend::graph::Tensor::set_uid)
        .def("__repr__", [](cudnn_frontend::graph::Tensor const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });
    
    py::class_<cudnn_frontend::graph::Convolution> convolution(m, "convolution");
    convolution.def(py::init<std::string const &>())
        .def("get_padding",  &cudnn_frontend::graph::Convolution::get_padding)
        .def("set_padding",  &cudnn_frontend::graph::Convolution::set_padding)
        .def("get_stride",   &cudnn_frontend::graph::Convolution::get_stride)
        .def("set_stride",   &cudnn_frontend::graph::Convolution::set_stride)
        .def("get_dilation", &cudnn_frontend::graph::Convolution::get_dilation)
        .def("set_dilation", &cudnn_frontend::graph::Convolution::set_dilation)
        .def("get_compute_data_type",     &cudnn_frontend::graph::Convolution::get_compute_data_type)
        .def("set_compute_data_type",     &cudnn_frontend::graph::Convolution::set_compute_data_type)
        .def("get_tensor_at_port",       &cudnn_frontend::graph::Convolution::get_tensor_at_port)
        .def("map_port_to_tensor",       &cudnn_frontend::graph::Convolution::map_port_to_tensor)
        .def("__repr__", [](cudnn_frontend::graph::Convolution const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });;

    py::enum_<cudnn_frontend::graph::Convolution::PORTS>(convolution, "ports")
        .value("X", cudnn_frontend::graph::Convolution::PORTS::X)
        .value("W", cudnn_frontend::graph::Convolution::PORTS::W)
        .value("Y", cudnn_frontend::graph::Convolution::PORTS::Y)
        .export_values();
    
    py::class_<cudnn_frontend::graph::Matmul> matmul(m, "matmul");
    matmul.def(py::init<std::string const &>())
        .def("get_compute_data_type",     &cudnn_frontend::graph::Matmul::get_compute_data_type)
        .def("set_compute_data_type",     &cudnn_frontend::graph::Matmul::set_compute_data_type)
        .def("get_tensor_at_port",       &cudnn_frontend::graph::Matmul::get_tensor_at_port)
        .def("map_port_to_tensor",       &cudnn_frontend::graph::Matmul::map_port_to_tensor)
        .def("__repr__", [](cudnn_frontend::graph::Matmul const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });;

    py::enum_<cudnn_frontend::graph::Matmul::PORTS>(matmul, "ports")
        .value("A", cudnn_frontend::graph::Matmul::PORTS::A)
        .value("B", cudnn_frontend::graph::Matmul::PORTS::B)
        .value("C", cudnn_frontend::graph::Matmul::PORTS::C)
        .export_values();

    py::class_<cudnn_frontend::graph::Pointwise> pointwise(m, "pointwise");
    pointwise.def(py::init<std::string const &>())
        .def("get_mode", &cudnn_frontend::graph::Pointwise::get_mode)
        .def("set_mode", &cudnn_frontend::graph::Pointwise::set_mode)
        .def("get_compute_data_type",      &cudnn_frontend::graph::Pointwise::get_compute_data_type)
        .def("set_compute_data_type",      &cudnn_frontend::graph::Pointwise::set_compute_data_type)
        .def("get_tensor_at_port",       &cudnn_frontend::graph::Pointwise::get_tensor_at_port)
        .def("map_port_to_tensor",       &cudnn_frontend::graph::Pointwise::map_port_to_tensor);

    py::enum_<cudnn_frontend::graph::Pointwise::PORTS>(pointwise, "ports")
        .value("X", cudnn_frontend::graph::Pointwise::PORTS::X)
        .value("B", cudnn_frontend::graph::Pointwise::PORTS::B)
        .value("Y", cudnn_frontend::graph::Pointwise::PORTS::Y)
        .export_values();
}