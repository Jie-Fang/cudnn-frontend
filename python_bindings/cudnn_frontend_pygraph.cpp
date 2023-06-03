#include <utility>
#include <unordered_map>

#include "dlpack/dlpack.h"

// Part of the Array API specification.
#define CUDNN_FRONTEND_DLPACK_CAPSULE_NAME "dltensor"
#define CUDNN_FRONTEND_DLPACK_USED_CAPSULE_NAME "used_dltensor"

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
        case cudnn_frontend::error_t::HEURISTIC_QUERY_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_t::INVALID_CUDA_DEVICE:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_t::UNSUPPORTED_GRAPH_FORMAT:
            throw std::runtime_error(error_msg);
    }
}

char* extract_data_pointer(py::object obj) {
    throw_if(!py::hasattr(obj, "__dlpack__"), cudnn_frontend::error_t::INVALID_VARIANT_PACK, "Object does not have the __dlpack__() method");

    py::capsule capsule = obj.attr("__dlpack__")();
    throw_if(capsule.is_none(), cudnn_frontend::error_t::INVALID_VARIANT_PACK, "Failed to retrieve the DLPack capsule.");

    DLManagedTensor *managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged, cudnn_frontend::error_t::INVALID_VARIANT_PACK, "Invalid device type.");

    return (char *)managed->dl_tensor.data + managed->dl_tensor.byte_offset;
}

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
    // This Graph class is the sole structure which implicitly makes PyGraph own all tensors, nodes, and cudnn descriptors.
    cudnn_frontend::graph::Graph graph;

public:
    PyGraph(std::string const &name,
            cudnn_frontend::DataType_t io_data_type,
            cudnn_frontend::DataType_t intermediate_data_type,
            cudnn_frontend::DataType_t compute_data_type) : graph(name) {
                graph.set_compute_data_type(compute_data_type)
                     .set_intermediate_data_type(intermediate_data_type)
                     .set_io_data_type(io_data_type);
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
    // Takes all tensor properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor>>
    insert_batchnorm(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& X_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& scale_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& bias_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& in_running_mean_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& in_running_var_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& epsilon_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& exp_avg_factor_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Batchnorm(name)
                        .set_compute_data_type(compute_data_type)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Batchnorm::PORTS::X, X_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_mean, in_running_mean_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_var, in_running_var_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Batchnorm::PORTS::Scale, scale_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Batchnorm::PORTS::Bias, bias_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Batchnorm::PORTS::EPS, epsilon_props_ptr->get_name()}
                            , {cudnn_frontend::graph::Batchnorm::PORTS::EXP_AVG, exp_avg_factor_props_ptr->get_name()}
                        });        
        graph.insert_node(props);
        
        auto Y_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Y);
        auto Y_tensor = cudnn_frontend::graph::Tensor(Y_tensor_name);
        graph.insert_tensor(Y_tensor);
        
        auto Mean_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Mean);
        auto Mean_tensor = cudnn_frontend::graph::Tensor(Mean_tensor_name);
        graph.insert_tensor(Mean_tensor);
        
        auto Var_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Var);
        auto Var_tensor = cudnn_frontend::graph::Tensor(Var_tensor_name);
        graph.insert_tensor(Var_tensor);
        
        auto Next_running_mean_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Next_running_mean);
        auto Next_running_mean_tensor = cudnn_frontend::graph::Tensor(Next_running_mean_tensor_name);
        graph.insert_tensor(Next_running_mean_tensor);
        
        auto Next_running_var_tensor_name = props.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Next_running_var);
        auto Next_running_var_tensor = cudnn_frontend::graph::Tensor(Next_running_var_tensor_name);
        graph.insert_tensor(Next_running_var_tensor);

        return {graph.get_tensor(Y_tensor_name), graph.get_tensor(Mean_tensor_name), graph.get_tensor(Var_tensor_name), graph.get_tensor(Next_running_mean_tensor_name), graph.get_tensor(Next_running_var_tensor_name)};
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
        cudnn_frontend::DataType_t const& compute_data_type,
        std::vector<int64_t> const& padding,
        std::vector<int64_t> const& stride,
        std::vector<int64_t> const& dilation
    ) {
        auto props = cudnn_frontend::graph::Convolution(name)
                        .set_compute_data_type(compute_data_type)
                        .set_padding(padding)
                        .set_stride(stride)
                        .set_dilation(dilation);
        props.inputs.X = image_props_ptr;
        props.inputs.W = weight_props_ptr;
        auto outputs = graph.conv(props.inputs, props);

        return outputs.Y;        
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
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Matmul(name).set_compute_data_type(compute_data_type);
        props.inputs.A = image_props_ptr;
        props.inputs.B = weight_props_ptr;
        auto outputs = graph.matmul(props.inputs, props);

        return outputs.C;
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
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::ADD);
        props.inputs.IN_0 = input_props_ptr;
        props.inputs.IN_1 = bias_props_ptr;
        auto outputs = graph.pointwise(props.inputs, props);

        return outputs.OUT_0;
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
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::MUL);
        props.inputs.IN_0 = input_props_ptr;
        props.inputs.IN_1 = scale_props_ptr;
        auto outputs = graph.pointwise(props.inputs, props);

        return outputs.OUT_0;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_relu(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);
        props.inputs.IN_0 = input_props_ptr;
        auto outputs = graph.pointwise(props.inputs, props);

        return outputs.OUT_0;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_elu(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::ELU_FWD);
        props.inputs.IN_0 = input_props_ptr;
        auto outputs = graph.pointwise(props.inputs, props);

        return outputs.OUT_0;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes input properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_gelu(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr
        , cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::GELU_FWD);
        props.inputs.IN_0 = input_props_ptr;
        auto outputs = graph.pointwise(props.inputs, props);

        return outputs.OUT_0;
    }

    void build() {
        cudnnHandle_t handle;
        cudnnCreate(&handle);
        auto status = graph.build(handle);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Backend graph building failed.");

        auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

        status = graph.set_executor(plans);
        if (status != cudnn_frontend::error_t::OK) {
            auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK)
                        .build_plans(handle);

            status = graph.set_executor(plans);
            throw_if(status != cudnn_frontend::error_t::OK, status, "Backend Plan building failed.");
        }
        cudnnDestroy(handle);

        return;
    }

    int64_t get_workspace_size() {
        return graph.get_workspace_size();
    }

    void execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor>, py::object> var_pack) {
        cudnnHandle_t handle;
        cudnnCreate(&handle);

        std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor>, void *> var_pack_;
        for (auto item : var_pack) {
            var_pack_.insert(std::make_pair(item.first, extract_data_pointer(item.second)));
        }

        // TODO: Probably concatenate in a macro?
        auto status = graph.execute(handle, var_pack_);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Graph execution failed");
        
        cudnnDestroy(handle);
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
        .def(py::init<std::string const &, cudnn_frontend::DataType_t, cudnn_frontend::DataType_t, cudnn_frontend::DataType_t>(),
             py::arg_v("name", "test_graph"),
             py::arg_v("io_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("intermediate_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("tensor", &PyGraph::insert_tensor,
             py::arg_v("name", "test_tensor_name"),
             py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"dim", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false}
        )
        .def("batchnorm", &PyGraph::insert_batchnorm,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("in_running_mean"),
             py::arg("in_running_var"),
             py::arg("epsilon"),
             py::arg("exp_avg_factor"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("conv", &PyGraph::insert_conv,
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("matmul", &PyGraph::insert_matmul,
             py::arg_v("name", "test_tensor_name"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("bias", &PyGraph::insert_bias,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("bias"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("scale", &PyGraph::insert_scale,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg("scale"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("relu", &PyGraph::insert_relu,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("elu", &PyGraph::insert_elu,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("insert_gelu", &PyGraph::insert_gelu,
             py::arg_v("name", "test_tensor_name"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("build", &PyGraph::build)
        .def("get_workspace_size", &PyGraph::get_workspace_size)
        .def("execute", &PyGraph::execute)
        .def("__repr__", [](PyGraph const& graph){
            std::ostringstream out;
            out << graph;
            return out.str();
        });
}