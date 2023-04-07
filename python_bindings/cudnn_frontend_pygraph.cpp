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
void throw_if(bool const cond, cudnn_frontend::error_t const error_code, std::string const& error_msg) {
    if(cond == false)
        return;

    switch(error_code) {
        case cudnn_frontend::error_t::OK:
            return;
        case cudnn_frontend::error_t::ATTRIBUTE_NOT_SET:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_t::SHAPE_DEDUCTION_FAILED:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_t::INVALID_TENSOR_NAME:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_t::INVALID_VARIANT_PACK:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_t::GRAPH_EXECUTION_FAILED:
            throw std::runtime_error(error_msg);
    }
}

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
    // This Graph class is the sole structure which implicitly makes PyGraph own all tensors, nodes, and cudnn descriptors.
    cudnn_frontend::graph::Graph graph;

public:
    // TODO: only uses context as implementation underneath requires it.
    // Later will be removed as it will passed on to functions like a handle.
    PyGraph(std::string const &name ) : graph(name, cudnn_frontend::cuDNNFEContext()) {}

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_tensor(
        std::string const& name
        , cudnn_frontend::DataType_t const& data_type
        , std::vector<int64_t> const& dim
        , std::vector<int64_t> const& stride
        , bool const& isVirtual
        , bool const& isByValue
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(name);
        props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        props_ptr->set_dim(dim);
        props_ptr->set_stride(stride);
        props_ptr->set_is_virtual(isVirtual);
        props_ptr->set_is_pass_by_value(isByValue);

        auto status = graph.insert_tensor(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding tensor " + name + " failed.");

        return props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_conv(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& image_props_ptr
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& weight_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
        , std::vector<int64_t> const& padding
        , std::vector<int64_t> const& stride
        , std::vector<int64_t> const& dilation
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::convolution>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        props_ptr->set_padding(padding);
        props_ptr->set_stride(stride);
        props_ptr->set_dilation(dilation);

        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::convolution::PORTS::X, image_props_ptr->get_name()}, {cudnn_frontend::graph::convolution::PORTS::W, weight_props_ptr->get_name()}});
        
        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::convolution::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add conv node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_matmul(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& image_props_ptr
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& weight_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::matmul>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        
        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::matmul::PORTS::X, image_props_ptr->get_name()}, {cudnn_frontend::graph::matmul::PORTS::W, weight_props_ptr->get_name()}});

        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::matmul::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add matmul node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_bias(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& bias_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::pointwise>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        props_ptr->set_mode(cudnn_frontend::PointwiseMode_t::ADD);

        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::pointwise::PORTS::X, input_props_ptr->get_name()}, {cudnn_frontend::graph::pointwise::PORTS::B, bias_props_ptr->get_name()}});

        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add pointwise node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_scale(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& scale_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::pointwise>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        props_ptr->set_mode(cudnn_frontend::PointwiseMode_t::MUL);

        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::pointwise::PORTS::X, input_props_ptr->get_name()}, {cudnn_frontend::graph::pointwise::PORTS::B, scale_props_ptr->get_name()}});

        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add pointwise node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_relu(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::pointwise>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        props_ptr->set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);

        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::pointwise::PORTS::X, input_props_ptr->get_name()}});

        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add pointwise node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_elu(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::pointwise>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        props_ptr->set_mode(cudnn_frontend::PointwiseMode_t::ELU_FWD);

        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::pointwise::PORTS::X, input_props_ptr->get_name()}});

        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add pointwise node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor> 
    insert_gelu(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr
        , cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props_ptr = std::make_shared<cudnn_frontend::graph::pointwise>(name);
        props_ptr->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
        props_ptr->set_mode(cudnn_frontend::PointwiseMode_t::GELU_FWD);

        // TODO: Check whether image and weight already exist.
        props_ptr->map_port_to_tensor({{cudnn_frontend::graph::pointwise::PORTS::X, input_props_ptr->get_name()}});

        // Add output tensor to graph
        auto output_props_ptr = std::make_shared<cudnn_frontend::graph::Tensor>(props_ptr->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y));
        output_props_ptr->set_data_type(cudnn_frontend::DataType_t::HALF);
        auto status = graph.insert_tensor(output_props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding output tensor to node " + name + " failed.");

        // Add pointwise node to graph
        status = graph.insert_node(props_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Adding node " + name + " failed.");

        return output_props_ptr;
    }

    void build() {
        auto status = graph.build();
        throw_if(status != cudnn_frontend::error_t::OK, status, "Backend graph building failed.");

        return;
    }

    void execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor>, int64_t> var_pack) {
        std::unordered_map<std::string, void *> var_pack_;
        for (auto item : var_pack) {
            var_pack_.insert(std::make_pair(item.first->get_name(), (void *)item.second));
        }
        // TODO: Probably concatenate in a macro?
        auto status = graph.execute(var_pack_);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Graph execution failed");
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
        .def("insert_tensor", &PyGraph::insert_tensor, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("data_type"),
             py::arg_v{"dim", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false}
        )
        .def("insert_conv", &PyGraph::insert_conv, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg("compute_type"),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("insert_matmul", &PyGraph::insert_matmul, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg("compute_type")
        )
        .def("insert_bias", &PyGraph::insert_bias, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("bias"),
             py::arg("compute_type")
        )
        .def("insert_scale", &PyGraph::insert_scale, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("compute_type")
        )
        .def("insert_relu", &PyGraph::insert_relu, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("compute_type")
        )
        .def("insert_elu", &PyGraph::insert_elu, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("compute_type")
        )
        .def("insert_gelu", &PyGraph::insert_gelu, 
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("compute_type")
        )
        .def("build", &PyGraph::build)
        .def("execute", &PyGraph::execute)
        .def("__repr__", [](PyGraph const& graph){
            std::ostringstream out;
            out << graph;
            return out.str();
        });
}