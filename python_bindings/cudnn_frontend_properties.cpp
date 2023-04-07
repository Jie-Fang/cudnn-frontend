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
        .def("get_dim", static_cast< std::vector<int64_t>& (cudnn_frontend::graph::Tensor::*)()>((&cudnn_frontend::graph::Tensor::get_dim)))
        .def("set_dim", &cudnn_frontend::graph::Tensor::set_dim)
        .def("get_stride", &cudnn_frontend::graph::Tensor::get_stride)
        .def("set_stride", &cudnn_frontend::graph::Tensor::set_stride)
        .def("get_is_virtual", &cudnn_frontend::graph::Tensor::get_is_virtual)
        .def("set_is_virtual", &cudnn_frontend::graph::Tensor::set_is_virtual)
        .def("get_is_pass_by_value", &cudnn_frontend::graph::Tensor::get_is_pass_by_value)
        .def("set_is_pass_by_value", &cudnn_frontend::graph::Tensor::set_is_pass_by_value)
        .def("get_uid", &cudnn_frontend::graph::Tensor::get_uid)
        .def("set_uid", &cudnn_frontend::graph::Tensor::set_uid)
        .def("get_size", &cudnn_frontend::graph::Tensor::get_size)
        .def("__repr__", [](cudnn_frontend::graph::Tensor const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });
    
    py::class_<cudnn_frontend::graph::convolution> convolution(m, "convolution");
    convolution.def(py::init<std::string const &>())
        .def("get_padding",  &cudnn_frontend::graph::convolution::get_padding)
        .def("set_padding",  &cudnn_frontend::graph::convolution::set_padding)
        .def("get_stride",   &cudnn_frontend::graph::convolution::get_stride)
        .def("set_stride",   &cudnn_frontend::graph::convolution::set_stride)
        .def("get_dilation", &cudnn_frontend::graph::convolution::get_dilation)
        .def("set_dilation", &cudnn_frontend::graph::convolution::set_dilation)
        .def("get_tensor_data_type", &cudnn_frontend::graph::convolution::get_tensor_data_type)
        .def("set_tensor_data_type", &cudnn_frontend::graph::convolution::set_tensor_data_type)
        .def("get_compute_type",     &cudnn_frontend::graph::convolution::get_compute_type)
        .def("set_compute_type",     &cudnn_frontend::graph::convolution::set_compute_type)
        .def("get_port_name",       &cudnn_frontend::graph::convolution::get_port_name)
        .def("map_port_to_tensor",       &cudnn_frontend::graph::convolution::map_port_to_tensor)
        .def("__repr__", [](cudnn_frontend::graph::convolution const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });;

    py::enum_<cudnn_frontend::graph::convolution::PORTS>(convolution, "ports")
        .value("X", cudnn_frontend::graph::convolution::PORTS::X)
        .value("W", cudnn_frontend::graph::convolution::PORTS::W)
        .value("Y", cudnn_frontend::graph::convolution::PORTS::Y)
        .export_values();
    
    py::class_<cudnn_frontend::graph::matmul> matmul(m, "matmul");
    matmul.def(py::init<std::string const &>())
        .def("get_tensor_data_type", &cudnn_frontend::graph::convolution::get_tensor_data_type)
        .def("set_tensor_data_type", &cudnn_frontend::graph::convolution::set_tensor_data_type)
        .def("get_compute_type",     &cudnn_frontend::graph::convolution::get_compute_type)
        .def("set_compute_type",     &cudnn_frontend::graph::convolution::set_compute_type)
        .def("get_port_name",       &cudnn_frontend::graph::matmul::get_port_name)
        .def("map_port_to_tensor",       &cudnn_frontend::graph::matmul::map_port_to_tensor)
        .def("__repr__", [](cudnn_frontend::graph::matmul const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });;

    py::enum_<cudnn_frontend::graph::matmul::PORTS>(matmul, "ports")
        .value("X", cudnn_frontend::graph::matmul::PORTS::X)
        .value("W", cudnn_frontend::graph::matmul::PORTS::W)
        .value("Y", cudnn_frontend::graph::matmul::PORTS::Y)
        .export_values();

    py::class_<cudnn_frontend::graph::pointwise> pointwise(m, "pointwise");
    pointwise.def(py::init<std::string const &>())
        .def("get_mode", &cudnn_frontend::graph::pointwise::get_mode)
        .def("set_mode", &cudnn_frontend::graph::pointwise::set_mode)
        .def("get_tensor_data_type",  &cudnn_frontend::graph::convolution::get_tensor_data_type)
        .def("set_tensor_data_type",  &cudnn_frontend::graph::convolution::set_tensor_data_type)
        .def("get_compute_type",      &cudnn_frontend::graph::convolution::get_compute_type)
        .def("set_compute_type",      &cudnn_frontend::graph::convolution::set_compute_type)
        .def("get_port_name",       &cudnn_frontend::graph::pointwise::get_port_name)
        .def("map_port_to_tensor",       &cudnn_frontend::graph::pointwise::map_port_to_tensor);

    py::enum_<cudnn_frontend::graph::pointwise::PORTS>(pointwise, "ports")
        .value("X", cudnn_frontend::graph::pointwise::PORTS::X)
        .value("B", cudnn_frontend::graph::pointwise::PORTS::B)
        .value("Y", cudnn_frontend::graph::pointwise::PORTS::Y)
        .export_values();
}