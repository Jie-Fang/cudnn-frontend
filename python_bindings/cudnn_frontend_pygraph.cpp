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
    tensor(
        std::string const& name,
        std::vector<int64_t> const& dim,
        std::vector<int64_t> const& stride,
        cudnn_frontend::DataType_t const& data_type,
        bool const& is_virtual,
        bool const& is_by_value
    ) {
        auto props = cudnn_frontend::graph::Tensor(name)
                            .set_data_type(data_type)
                            .set_is_virtual(is_virtual)
                            .set_is_pass_by_value(is_by_value)
                            .set_dim(dim)
                            .set_stride(stride);
        
        return graph.tensor(props);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes all tensor properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor>>
    batchnorm(
        std::string const& name,
        cudnn_frontend::NormFwdPhase_t const forward_phase,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& X_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& scale_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& bias_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& in_running_mean_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& in_running_var_props_ptr,
        float const epsilon,
        float const momentum,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Batchnorm(name)
                        .set_forward_phase(forward_phase)
                        .set_compute_data_type(compute_data_type)
                        .set_epsilon(epsilon)
                        .set_momentum(momentum);
        props.inputs.X = X_props_ptr;
        props.inputs.SCALE = scale_props_ptr;
        props.inputs.BIAS = bias_props_ptr;
        props.inputs.PREV_RUNNING_MEAN = in_running_mean_props_ptr;
        props.inputs.PREV_RUNNING_VAR = in_running_var_props_ptr;

        auto [Y, mean, inv_var, next_running_mean, next_running_var] = graph.batchnorm(props.inputs, props);
        return {Y, mean, inv_var, next_running_mean, next_running_var};
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
        auto props = cudnn_frontend::graph::Conv_fprop(name)
                        .set_compute_data_type(compute_data_type)
                        .set_padding(padding)
                        .set_stride(stride)
                        .set_dilation(dilation);
        props.inputs.X = image_props_ptr;
        props.inputs.W = weight_props_ptr;
        auto [Y] = graph.conv_fprop(props.inputs, props);

        // Default virtualness in python is true
        Y->set_is_virtual(true);

        return Y;
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and loss properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor>
    insert_wgrad(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& image_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& loss_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type,
        std::vector<int64_t> const& padding,
        std::vector<int64_t> const& stride,
        std::vector<int64_t> const& dilation
    ) {
        auto props = cudnn_frontend::graph::Conv_wgrad(name)
                        .set_compute_data_type(compute_data_type)
                        .set_padding(padding)
                        .set_stride(stride)
                        .set_dilation(dilation);
        props.inputs.X = image_props_ptr;
        props.inputs.DY = loss_props_ptr;
        auto [DW] = graph.conv_wgrad(props.inputs, props);

        // Default virtualness in python is true
        DW->set_is_virtual(true);

        return DW;
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
        auto [C] = graph.matmul(props.inputs, props);

        // Default virtualness in python is true
        C->set_is_virtual(true);

        return C;
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
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
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
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
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
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor>, 2UL>
    genstats(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Genstats(name).set_compute_data_type(compute_data_type);
        props.inputs.X = input_props_ptr;
        auto [SUM, SQ_SUM] = graph.genstats(props.inputs.X, props);

        // Default virtualness in python is true
        SUM->set_is_virtual(true);
        SQ_SUM->set_is_virtual(true);

        return {SUM, SQ_SUM};
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
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
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
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor>, 2>
    scaled_dot_product_attention(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& q
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& k
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& v
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& seq_len_q
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& seq_len_k
        , bool const is_inference
        , float const scale_k
        , std::shared_ptr<cudnn_frontend::graph::Tensor>& bias
        , bool const use_padding_mask
        , bool const use_causal_mask
        , py::object const dropout
        , cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto scaled_dot_product_attention_options = cudnn_frontend::graph::Scaled_dot_product_attention("mha")
                                                    .set_is_inference(is_inference)
                                                    .set_scale_k(scale_k)
                                                    .set_compute_data_type(compute_data_type);

        if(use_padding_mask) {
            scaled_dot_product_attention_options.use_padding_mask();
        }

        if(use_causal_mask) {
            scaled_dot_product_attention_options.use_causal_mask();
        }

        if(bias) {
            scaled_dot_product_attention_options.set_bias(bias);
        }

        if (py::isinstance<py::tuple>(dropout)) {
            py::tuple dropout_tuple = dropout.cast<py::tuple>();
            if (dropout_tuple.size() != 2) {
                throw std::runtime_error("dropout must be a tuple of two floats or a tuple of cudnn tensor and a float");
            }

            if (py::isinstance<py::float_>(dropout_tuple[0]) && py::isinstance<py::int_>(dropout_tuple[1])) {
                auto const dropout_probability = dropout_tuple[0].cast<float>();
                auto const seed = dropout_tuple[1].cast<int32_t>();

                scaled_dot_product_attention_options.set_dropout(dropout_probability, seed);                
            } else if (py::isinstance<std::shared_ptr<cudnn_frontend::graph::Tensor>>(dropout_tuple[0]) && py::isinstance<py::float_>(dropout_tuple[1])) {
                auto const dropout_mask = dropout_tuple[0].cast<std::shared_ptr<cudnn_frontend::graph::Tensor>>();
                auto const dropout_scale = dropout_tuple[1].cast<float>();

                scaled_dot_product_attention_options.set_dropout(dropout_mask, dropout_scale);
            } else {
                throw std::runtime_error("dropout must be a tuple of two floats or two shared_ptr references");
            }
        }
        else if (dropout.is(py::none())) {
            // Still fine as user does not want any kind of dropout
        } else {
            throw std::runtime_error("dropout must be a tuple of two floats or a tuple of cudnn tensor and a float");
        }

        scaled_dot_product_attention_options.inputs.Q = q;
        scaled_dot_product_attention_options.inputs.K = k;
        scaled_dot_product_attention_options.inputs.V = v;
        scaled_dot_product_attention_options.inputs.SEQ_LEN_Q = seq_len_q;
        scaled_dot_product_attention_options.inputs.SEQ_LEN_K = seq_len_k;
        
        auto [S, O] = graph.scaled_dot_product_attention(scaled_dot_product_attention_options.inputs, scaled_dot_product_attention_options);

        // Default virtualness in python is true
        S->set_is_virtual(true);
        O->set_is_virtual(true);

        return {S, O};
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

    void execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor>, py::object> var_pack, py::object workspace) {
        cudnnHandle_t handle;
        cudnnCreate(&handle);

        std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor>, void *> var_pack_;
        for (auto const& [tensor, pyobject] : var_pack) {
            var_pack_.emplace(tensor, extract_data_pointer(pyobject));
        }

        void* workspace_ptr = extract_data_pointer(workspace);

        // TODO: Probably concatenate in a macro?
        auto status = graph.execute(handle, var_pack_, workspace_ptr);
        throw_if(status != cudnn_frontend::error_t::OK, status, "Graph execution failed");
        
        cudnnDestroy(handle);
        return;
    }
};

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
        .def("tensor", &PyGraph::tensor,
             py::arg_v("name", "test_tensor_name"),
             py::arg{"dim"},
             py::arg{"stride"},
             py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"is_virtual", false},
             py::arg_v{"is_pass_by_value", false}
        )
        .def("batchnorm", &PyGraph::batchnorm,
             py::arg_v("name", "batch_norm"),
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("in_running_mean"),
             py::arg("in_running_var"),
             py::arg_v("epsilon", 1.0e-5),
             py::arg_v("momentum", 0.1),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("genstats", &PyGraph::genstats,
             py::arg_v("name", "genstats"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("conv", &PyGraph::insert_conv,
             py::arg_v("name", "conv_fprop"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("wgrad", &PyGraph::insert_wgrad,
             py::arg_v("name", "conv_wgrad"),
             py::arg("image"),
             py::arg("loss"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("matmul", &PyGraph::insert_matmul,
             py::arg_v("name", "matmul"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("bias", &PyGraph::insert_bias,
             py::arg_v("name", "bias"),
             py::arg("input"),
             py::arg("bias"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("scale", &PyGraph::insert_scale,
             py::arg_v("name", "scale"),
             py::arg("input"),
             py::arg("scale"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("relu", &PyGraph::insert_relu,
             py::arg_v("name", "relu"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("elu", &PyGraph::insert_elu,
             py::arg_v("name", "elu"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("insert_gelu", &PyGraph::insert_gelu,
             py::arg_v("name", "gelu"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("scaled_dot_product_attention", &PyGraph::scaled_dot_product_attention,
             py::arg_v("name", "scaled_dot_product_attention"),
             py::arg("q"),
             py::arg("k"),
             py::arg("v"),
             py::arg("seq_len_q"),
             py::arg("seq_len_k"),
             py::arg("is_inference"),
             py::arg_v("scale_k", 1.f),
             py::arg_v("bias", nullptr),
             py::arg_v("use_padding_mask", false),
             py::arg_v("use_causal_mask", false),
             py::arg_v("dropout", py::none()),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("build", &PyGraph::build)
        .def("get_workspace_size", &PyGraph::get_workspace_size)
        .def("execute", &PyGraph::execute);


}