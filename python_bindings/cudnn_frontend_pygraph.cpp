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
    PyGraph(std::string const &name,
            int64_t tensor_dims,
            int64_t spatial_dims,
            cudnn_frontend::DataType_t io_data_type,
            cudnn_frontend::DataType_t intermediate_data_type,
            cudnn_frontend::DataType_t compute_data_type) : graph(name) {
                graph.set_compute_type(compute_data_type)
                    .set_intermediate_data_type(intermediate_data_type)
                    .set_io_data_type(io_data_type)
                    .set_tensor_dims(tensor_dims)
                    .set_spatial_dims(spatial_dims);
            }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_tensor(
        std::string const& name,
        cudnn_frontend::DataType_t const& data_type,
        std::vector<int64_t> const& dim,
        std::vector<int64_t> const& stride,
        bool const& isVirtual,
        bool const& isByValue
    ) {
        auto props = cudnn_frontend::graph::Tensor(name);

        if(data_type != cudnn_frontend::DataType_t::NOT_SET)
            props.set_data_type(data_type);
        
        if(dim.size())
            props.set_dim(dim);
        
        if(stride.size())
            props.set_stride(stride);
        
        props.set_is_virtual(isVirtual).set_is_pass_by_value(isByValue);
        
        graph.insert_tensor(props);

        return graph.get_tensor(name);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_conv(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& image_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& weight_props_ptr,
        cudnn_frontend::DataType_t const& compute_type,
        std::vector<int64_t> const& padding,
        std::vector<int64_t> const& stride,
        std::vector<int64_t> const& dilation
    ) {
        auto props = cudnn_frontend::graph::Convolution(name)
                        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_padding(padding)
                        .set_stride(stride)
                        .set_dilation(dilation)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Convolution::PORTS::X, image_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Convolution::PORTS::W, weight_props_ptr->get_name()}
                        });
        
        // Add conv node to graph
        graph.insert_node(props);
        
        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_matmul(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& image_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& weight_props_ptr,
        cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props = cudnn_frontend::graph::Matmul(name)
        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
        .map_port_to_tensor({
            {cudnn_frontend::graph::Matmul::PORTS::X, image_props_ptr->get_name()},
            {cudnn_frontend::graph::Matmul::PORTS::W, weight_props_ptr->get_name()}
        });

        // Add matmul node to graph
        graph.insert_node(props);

        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Matmul::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_bias(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& bias_props_ptr,
        cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name)
                        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, input_props_ptr->get_name()},
                            {cudnn_frontend::graph::Pointwise::PORTS::B, bias_props_ptr->get_name()}
                        });

        // Add pointwise node to graph
        graph.insert_node(props);

        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_scale(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& scale_props_ptr,
        cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name)
                        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, input_props_ptr->get_name()},
                            {cudnn_frontend::graph::Pointwise::PORTS::B, scale_props_ptr->get_name()}
                        });

        // Add pointwise node to graph
        graph.insert_node(props);

        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_relu(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name)
                        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, input_props_ptr->get_name()}
                        });

        // Add pointwise node to graph
        graph.insert_node(props);

        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_elu(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name)
                        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_mode(cudnn_frontend::PointwiseMode_t::ELU_FWD)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, input_props_ptr->get_name()}
                        });

        // Add pointwise node to graph
        graph.insert_node(props);

        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
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
        auto props = cudnn_frontend::graph::Pointwise(name)
                        .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                        .set_mode(cudnn_frontend::PointwiseMode_t::GELU_FWD)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, input_props_ptr->get_name()}
                        });

        // Add pointwise node to graph
        graph.insert_node(props);

        auto output_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y);
        auto output_tensor = cudnn_frontend::graph::Tensor(output_tensor_name);
        graph.insert_tensor(output_tensor);

        return graph.get_tensor(output_tensor_name);
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
        .def(py::init<std::string const &, int64_t, int64_t,
                cudnn_frontend::DataType_t,cudnn_frontend::DataType_t,cudnn_frontend::DataType_t>(),
             py::arg_v("name", "test_graph"),
             py::arg_v("tensor_dims", 4),
             py::arg_v("spatial_dims", 2),
             py::arg_v("io_data_type", cudnn_frontend::DataType_t::HALF),
             py::arg_v("intermediate_data_type", cudnn_frontend::DataType_t::FLOAT),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::FLOAT)
        )
        .def("tensor", &PyGraph::insert_tensor,
             py::arg_v("name", "test_tensor_name"),
             py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"dim", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false}
        )
        .def("conv", &PyGraph::insert_conv,
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg("compute_type"),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("matmul", &PyGraph::insert_matmul,
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg("compute_type")
        )
        .def("bias", &PyGraph::insert_bias,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("bias"),
             py::arg("compute_type")
        )
        .def("scale", &PyGraph::insert_scale,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("compute_type")
        )
        .def("relu", &PyGraph::insert_relu,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("compute_type")
        )
        .def("elu", &PyGraph::insert_elu,
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