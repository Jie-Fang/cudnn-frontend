#include <utility>
#include <unordered_map>

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

#include "cudnn_frontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

// Raise C++ exceptions corresponding to C++ FE error codes.
// Pybinds will automatically convert C++ exceptions to pythpn exceptions.
void throw_if(bool const cond, cudnn_frontend::cudnn_frontend_error_t const error_code, std::string const& error_msg) {
    if(cond == false)
        return;

    switch(error_code) {
        case cudnn_frontend::cudnn_frontend_error_t::OK:
            return;
        case cudnn_frontend::cudnn_frontend_error_t::ATTRIBUTE_NOT_SET:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::cudnn_frontend_error_t::SHAPE_DEDUCTION_FAILED:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::cudnn_frontend_error_t::INVALID_TENSOR_NAME:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::cudnn_frontend_error_t::INVALID_VARIANT_PACK:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::cudnn_frontend_error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::cudnn_frontend_error_t::GRAPH_EXECUTION_FAILED:
            throw std::runtime_error(error_msg);
    }
}

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
    // This Graph class is the sole structure which implicitly makes PyGraph own all tensors, nodes, and cudnn descriptors.
    cudnn_frontend::Graph graph;

public:
    // TODO: only uses context as implementation underneath requires it.
    // Later will be removed as it will passed on to functions like a handle.
    PyGraph(std::string const &name ) : graph(name, cudnn_frontend::cuDNNFEContext()) {}

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    std::shared_ptr<cudnn_frontend::tensor_properties> 
    add_tensor(
        std::string const& name
        , std::string const& data_type
        , std::vector<int64_t> const& dim
        , std::vector<int64_t> const& stride
        , bool const& isVirtual
        , bool const& isByValue
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::tensor_properties>(name);
        props_ptr->set_data_type(CUDNN_DATA_HALF);
        props_ptr->set_dim(dim);
        props_ptr->set_stride(stride);
        props_ptr->set_is_virtual(isVirtual);
        props_ptr->set_is_pass_by_value(isByValue);

        auto status = graph.add_tensor(props_ptr);
        throw_if(status != cudnn_frontend::cudnn_frontend_error_t::OK, status, "Adding tensor " + name + " failed.");

        return props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::tensor_properties> 
    add_conv(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::tensor_properties>& image_props_ptr
        , std::shared_ptr<cudnn_frontend::tensor_properties>& weight_props_ptr
        , std::string const& compute_type
        , std::vector<int64_t> const& padding
        , std::vector<int64_t> const& stride
        , std::vector<int64_t> const& dilation
    ) {
        cudnn_frontend::convolution_node props(name);
        props.set_compute_type(CUDNN_DATA_FLOAT);
        props.set_padding(padding);
        props.set_stride(stride);
        props.set_dilation(dilation);

        // TODO: Check whether image and weight already exist.
        props.set_port_names({{cudnn_frontend::convolution_node::PORTS::X, image_props_ptr->get_name()}, {cudnn_frontend::convolution_node::PORTS::W, weight_props_ptr->get_name()}});

        auto status = graph.add_node(props);
        throw_if(status != cudnn_frontend::cudnn_frontend_error_t::OK, status, "Adding node " + name + " failed.");

        return graph.get_tensor(props.get_port_name(cudnn_frontend::convolution_node::PORTS::Y));
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::tensor_properties> 
    add_bias(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::tensor_properties>& input_props_ptr
        , std::shared_ptr<cudnn_frontend::tensor_properties>& bias_props_ptr
        , std::string const& compute_type
    ) {
        cudnn_frontend::pointwise_node props(name);
        props.set_compute_type(CUDNN_DATA_FLOAT);
        props.set_mode("Add");

        // TODO: Check whether image and weight already exist.
        props.set_port_names({{cudnn_frontend::pointwise_node::PORTS::X, input_props_ptr->get_name()}, {cudnn_frontend::pointwise_node::PORTS::B, bias_props_ptr->get_name()}});

        auto status = graph.add_node(props);
        throw_if(status != cudnn_frontend::cudnn_frontend_error_t::OK, status, "Adding node " + name + " failed.");

        return graph.get_tensor(props.get_port_name(cudnn_frontend::pointwise_node::PORTS::Y));
    }

    void build() {
        auto status = graph.infer_shapes();
        throw_if(status != cudnn_frontend::cudnn_frontend_error_t::OK, status, "Property inferencing failed.");

        status = graph.build();
        throw_if(status != cudnn_frontend::cudnn_frontend_error_t::OK, status, "Backend graph building failed.");
    }

    void execute(std::unordered_map<std::string, int64_t> var_pack) {
        std::unordered_map<std::string, void *> var_pack_;
        for (auto item : var_pack) {
            var_pack_.insert(std::make_pair(item.first, (void *)item.second));
        }
        // TODO: Probably concatenate in a macro?
        auto status = graph.execute(var_pack_);
        throw_if(status != cudnn_frontend::cudnn_frontend_error_t::OK, status, "Graph execution failed");
        return;
    }

    friend std::ostream& operator<<(std::ostream& os, const PyGraph& props);
};

inline std::ostream& operator<<(std::ostream& os, const PyGraph& props) {
    os << props.graph;
    return os;
}

std::vector<int64_t>
default_vector(void) {
    return {};
}

void init_pygraph_submodule(py::module_ &m) {
  py::class_<PyGraph>(m, "pygraph")
        .def(py::init<std::string const &>())
        .def("add_tensor", &PyGraph::add_tensor, 
             py::arg_v("name", "test_tensor_name"),
             py::arg_v("data_type", "half"),
             py::arg_v{"dim", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false}
        )
        .def("add_conv", &PyGraph::add_conv, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_type", "float"),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("add_bias", &PyGraph::add_bias, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("bias"),
             py::arg_v("compute_type", "float")
        )
        .def("build", &PyGraph::build)
        .def("execute", &PyGraph::execute)
        .def("__repr__", [](PyGraph const& graph){
            std::ostringstream out;
            out << graph;
            return out.str();
        });
}