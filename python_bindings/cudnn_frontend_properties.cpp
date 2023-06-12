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
        .def("set_output", [](cudnn_frontend::graph::Tensor& self, bool const is_output){
            self.set_is_virtual(!is_output);
            return self;
        }) // NOTICE THATS ITS JUST ANOTHER NAME FOR SET_IS_VIRTUAL
        .def("get_is_pass_by_value", &cudnn_frontend::graph::Tensor::get_is_pass_by_value)
        .def("set_is_pass_by_value", &cudnn_frontend::graph::Tensor::set_is_pass_by_value)
        .def("get_uid", &cudnn_frontend::graph::Tensor::get_uid)
        .def("set_uid", &cudnn_frontend::graph::Tensor::set_uid)
        .def("__repr__", [](cudnn_frontend::graph::Tensor const& props){
            std::ostringstream out;
            out << props;
            return out.str();
        });
}