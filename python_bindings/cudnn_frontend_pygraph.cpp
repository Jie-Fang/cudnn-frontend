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
void throw_if(bool const cond, cudnn_frontend::error_code_t const error_code, std::string const& error_msg) {
    if(cond == false)
        return;

    switch(error_code) {
        case cudnn_frontend::error_code_t::OK:
            return;
        case cudnn_frontend::error_code_t::ATTRIBUTE_NOT_SET:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::SHAPE_DEDUCTION_FAILED:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_TENSOR_NAME:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::INVALID_VARIANT_PACK:
            throw std::invalid_argument(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::GRAPH_EXECUTION_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::HEURISTIC_QUERY_FAILED:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::INVALID_CUDA_DEVICE:
            throw std::runtime_error(error_msg);
        case cudnn_frontend::error_code_t::UNSUPPORTED_GRAPH_FORMAT:
            throw std::runtime_error(error_msg);
    }
}

char* extract_data_pointer(py::object obj) {
    throw_if(!py::hasattr(obj, "__dlpack__"), cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Object does not have the __dlpack__() method");

    py::capsule capsule = obj.attr("__dlpack__")();
    throw_if(capsule.is_none(), cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Failed to retrieve the DLPack capsule.");

    DLManagedTensor *managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule.ptr(), CUDNN_FRONTEND_DLPACK_CAPSULE_NAME));
    throw_if(managed == nullptr, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid DLPack capsule.");

    DLDeviceType device_type = managed->dl_tensor.device.device_type;
    throw_if(device_type != kDLCPU && device_type != kDLCUDAHost && device_type != kDLCUDA && device_type != kDLCUDAManaged, cudnn_frontend::error_code_t::INVALID_VARIANT_PACK, "Invalid device type.");

    return (char *)managed->dl_tensor.data + managed->dl_tensor.byte_offset;
}

// This class is only meant direct pythonic API calls to c++ Graph class.
class PyGraph {
public:
    // This Graph class is the sole structure which implicitly makes PyGraph own all tensors, nodes, and cudnn descriptors.
    cudnn_frontend::graph::Graph graph;
    cudnnHandle_t handle;
    bool is_built;


    PyGraph(std::string const &name,
            cudnn_frontend::DataType_t io_data_type,
            cudnn_frontend::DataType_t intermediate_data_type,
            cudnn_frontend::DataType_t compute_data_type) : graph(name) ,handle(nullptr), is_built(false) {
                graph.set_compute_data_type(compute_data_type)
                     .set_intermediate_data_type(intermediate_data_type)
                     .set_io_data_type(io_data_type);
        cudnnCreate(&handle);
    }
    
    ~PyGraph() {
        cudnnDestroy(handle);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    tensor(
        std::vector<int64_t> const& dim,
        std::vector<int64_t> const& stride,
        cudnn_frontend::DataType_t const& data_type,
        bool const& is_virtual,
        bool const& is_pass_by_value,
        py::object const& name
    ) {
        auto props = cudnn_frontend::graph::Tensor_attributes()
                            .set_data_type(data_type)
                            .set_is_virtual(is_virtual)
                            .set_is_pass_by_value(is_pass_by_value)
                            .set_dim(dim)
                            .set_stride(stride);
        
        if (!name.is_none()) {
            if(py::isinstance<py::str>(name)) {
                props.set_name(name.cast<std::string>());
            }
            else {
                throw std::invalid_argument("tensor name can only be str type.");
            }
        }
        
        return graph.tensor(props);
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes all tensor properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    batchnorm(
        std::string const& name,
        cudnn_frontend::NormFwdPhase_t const forward_phase,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& X_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_mean_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& in_running_var_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& epsilon,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& momentum,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Batchnorm_attributes(name)
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

    std::vector<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>
    batchnorm_backward(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& grad_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& mean_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& inv_variance_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::DBN_attributes(name)
                        .set_compute_data_type(compute_data_type);
        props.inputs.X = input_props_ptr;
        props.inputs.DY = grad_props_ptr;
        props.inputs.SCALE = scale_props_ptr;
        props.inputs.MEAN = mean_props_ptr;
        props.inputs.INV_VARIANCE = inv_variance_props_ptr;
        
        auto [DX, DScale, DBias] = graph.batchnorm_backward(props.inputs, props);
        return {DX, DScale, DBias};
    }

    // Returns a shared pointer as both this PyGraph class and the caller will own
    // the underlying object.
    // Takes image and weight properties by reference to shared pointer. This means this callee
    // does not own them and will not increse ref count.
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    conv(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type,
        std::vector<int64_t> const& padding,
        std::vector<int64_t> const& stride,
        std::vector<int64_t> const& dilation
    ) {
        auto props = cudnn_frontend::graph::Conv_fprop_attributes(name)
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    wgrad(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& loss_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type,
        std::vector<int64_t> const& padding,
        std::vector<int64_t> const& stride,
        std::vector<int64_t> const& dilation
    ) {
        auto props = cudnn_frontend::graph::Conv_wgrad_attributes(name)
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    matmul(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& image_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& weight_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Matmul_attributes(name).set_compute_data_type(compute_data_type);
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    bias(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise_attributes(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::ADD);
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    scale(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise_attributes(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::MUL);
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    relu(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise_attributes(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);
        props.inputs.IN_0 = input_props_ptr;
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2UL>
    genstats(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Genstats_attributes(name).set_compute_data_type(compute_data_type);
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    elu(
        std::string const& name,
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr,
        cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise_attributes(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::ELU_FWD);
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
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>
    gelu(
        std::string const& name
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_props_ptr
        , cudnn_frontend::DataType_t const& compute_data_type
    ) {
        auto props = cudnn_frontend::graph::Pointwise_attributes(name).set_compute_data_type(compute_data_type).set_mode(cudnn_frontend::PointwiseMode_t::GELU_FWD);
        props.inputs.IN_0 = input_props_ptr;
        auto [OUT_0] = graph.pointwise(props.inputs, props);

        // Default virtualness in python is true
        OUT_0->set_is_virtual(true);

        return OUT_0;
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
    scaled_dot_product_attention(
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_q
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_len_k
        , bool const is_inference
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_k
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias
        , bool const use_padding_mask
        , bool const use_causal_mask
        , py::object const& dropout
        , cudnn_frontend::DataType_t const& compute_data_type
        , py::object const& name
    ) {
        auto attributes = cudnn_frontend::graph::Scaled_dot_product_attention_attributes()
                                                    .set_is_inference(is_inference)
                                                    .set_scale_k(scale_k)
                                                    .set_bias(bias)
                                                    .set_padding_mask(use_padding_mask)
                                                    .set_causal_mask(use_causal_mask)
                                                    .set_compute_data_type(compute_data_type);
        
        if (!name.is_none()) {
            if(py::isinstance<py::str>(name)) {
                attributes.set_name(name.cast<std::string>());
            }
            else {
                throw std::invalid_argument("tensor name can only be str type.");
            }
        }

        if (!dropout.is_none()) {
            py::tuple dropout_tuple = dropout.cast<py::tuple>();
            if ((!dropout_tuple) || dropout_tuple.size() != 2) {
                throw std::runtime_error("dropout must be a tuple of (float probability, int seed) or a tuple of (mask tensor, scale tensor).");
            }

            if (py::isinstance<py::float_>(dropout_tuple[0]) && py::isinstance<py::int_>(dropout_tuple[1])) {
                auto const dropout_probability = dropout_tuple[0].cast<float>();
                auto const seed = dropout_tuple[1].cast<int32_t>();

                attributes.set_dropout(dropout_probability, seed);                
            } else {
                auto const dropout_mask = dropout_tuple[0].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if(!dropout_mask) {
                    throw std::runtime_error("dropout mask must be a cudnn_tensor.");
                }

                auto const dropout_scale = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if(!dropout_scale) {
                    throw std::runtime_error("dropout scale must be a cudnn_tensor.");
                }

                attributes.set_dropout(dropout_mask, dropout_scale);
            }
        }

        attributes.inputs.SEQ_LEN_Q = seq_len_q;
        attributes.inputs.SEQ_LEN_K = seq_len_k;
        
        auto [O, S] = graph.scaled_dot_product_attention(q, k, v, attributes);

        // Default virtualness in python is true
        S->set_is_virtual(true);
        O->set_is_virtual(true);

        return {O, S};
    }

    std::array<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, 2>
    scaled_dot_product_flash_attention(
        std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& q
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& k
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& v
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_q
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& seq_k
        , bool const is_inference
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& scale_k
        , std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& bias
        , bool const use_padding_mask
        , bool const use_alibi_mask
        , bool const use_causal_mask
        , py::object const& dropout
        , cudnn_frontend::DataType_t const& compute_data_type
        , py::object const& name
    ) {
        auto attributes = cudnn_frontend::graph::Scaled_dot_product_flash_attention_attributes()
                                                    .set_is_inference(is_inference)
                                                    .set_seq_len_q(seq_q)
                                                    .set_seq_len_k(seq_k)
                                                    .set_scale_k(scale_k)
                                                    .set_bias(bias)
                                                    .set_padding_mask(use_padding_mask)
                                                    .set_alibi_mask(use_alibi_mask)
                                                    .set_causal_mask(use_causal_mask)
                                                    .set_compute_data_type(compute_data_type);
        
        if (!name.is_none()) {
            if(py::isinstance<py::str>(name)) {
                attributes.set_name(name.cast<std::string>());
            }
            else {
                throw std::invalid_argument("tensor name can only be str type.");
            }
        }

        if (!dropout.is_none()) {
            py::tuple dropout_tuple = dropout.cast<py::tuple>();
            if ((!dropout_tuple) || (dropout_tuple.size() != 3 && dropout_tuple.size() != 2)) {
                throw std::runtime_error("dropout must be a tuple of (float probability, a seed tensor, and an offset tensor) or (mask tensor, scale tensor)");
            }
            if(py::isinstance<py::float_>(dropout_tuple[0])) {
                auto const probability = dropout_tuple[0].cast<float>();
                auto const seed = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if(!seed) {
                    throw std::runtime_error("dropout seed must be a cudnn_tensor.");
                }

                auto const offset = dropout_tuple[2].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if(!offset) {
                    throw std::runtime_error("dropout offset must be a cudnn_tensor.");
                }
    
                attributes.set_dropout(probability, seed, offset);
            }
            else {
                auto const mask = dropout_tuple[0].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if(!mask) {
                    throw std::runtime_error("dropout mask must be a cudnn_tensor.");
                }
                
                auto const scale = dropout_tuple[1].cast<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>>();
                if(!scale) {
                    throw std::runtime_error("dropout scale must be a cudnn_tensor.");
                }

                attributes.set_dropout(mask, scale);
            }
        }
        
        auto [O, Stats] = graph.scaled_dot_product_flash_attention(q, k, v, attributes);

        // Default virtualness in python is true
        Stats->set_is_virtual(true);
        O->set_is_virtual(true);

        return {O, Stats};
    }

    void check_support() {
        build();
    }

    void build() {
        if (is_built) {return;}
        
        is_built = true;
        
        auto status = graph.validate();
        throw_if(status.is_bad(), status.get_code(), status.get_message());

        status = graph.build_operation_graph(handle);
        throw_if(status.is_bad(), status.get_code(), status.get_message());

        auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A);

        status = plans.check_support(handle);
        if (status.is_bad()) {
            auto fallback_plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK);
            status = fallback_plans.check_support(handle);
            throw_if(status.is_bad(), status.get_code(), status.get_message());
            status = graph.set_execution_plans(fallback_plans);
        } else {
            status = graph.set_execution_plans(plans);
        }
        return;
    }

    int64_t get_workspace_size() {
        return graph.get_workspace_size();
    }

    void execute(std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, py::object> var_pack, py::object workspace) {

        std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void *> var_pack_;
        for (auto const& [tensor, pyobject] : var_pack) {
            var_pack_.emplace(tensor, extract_data_pointer(pyobject));
        }

        void* workspace_ptr = extract_data_pointer(workspace);

        // TODO: Probably concatenate in a macro?
        auto status = graph.execute(handle, var_pack_, workspace_ptr);
        throw_if(status.is_bad(), status.get_code(), status.get_message());
        
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
        .def("tensor", &PyGraph::tensor
             , py::arg{"dim"}
             , py::arg{"stride"}
             , py::arg_v("data_type", cudnn_frontend::DataType_t::NOT_SET)
             , py::arg_v{"is_virtual", false}
             , py::arg_v{"is_pass_by_value", false}
             , py::arg_v("name", py::none())
             , R"pbdoc(
                Create a tensor.

                Args:
                    dim (List[int]): The dimensions of the tensor.
                    stride (List[int]): The strides of the tensor.
                    data_type (pycudnn.data_type): The data type of the tensor. Default is pycudnn.data_type.NOT_SET.
                    is_virtual (bool): Flag indicating if the tensor is virtual. Default is False.
                    is_pass_by_value (bool): Flag indicating if the tensor is passed by value. Default is False.
                    name (Optional[str]): The name of the tensor.

                Returns:
                    cudnn_tensor: The created tensor.
            )pbdoc"
        )
        .def("batchnorm", &PyGraph::batchnorm,
             py::arg_v("name", "batch_norm"),
             py::arg("norm_forward_phase"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("bias"),
             py::arg("in_running_mean"),
             py::arg("in_running_var"),
             py::arg("epsilon"),
             py::arg("momentum"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("batchnorm_backward", &PyGraph::batchnorm_backward,
             py::arg_v("name", "batchnorm_backward"),
             py::arg("grad"),
             py::arg("input"),
             py::arg("scale"),
             py::arg("mean"),
             py::arg("inv_variance"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("genstats", &PyGraph::genstats,
             py::arg_v("name", "genstats"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("conv", &PyGraph::conv,
             py::arg_v("name", "conv_fprop"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("wgrad", &PyGraph::wgrad,
             py::arg_v("name", "conv_wgrad"),
             py::arg("image"),
             py::arg("loss"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET),
             py::arg_v{"padding", default_vector()},
             py::arg_v{"stride", default_vector()},
             py::arg_v{"dilation", default_vector()}
        )
        .def("matmul", &PyGraph::matmul,
             py::arg_v("name", "matmul"),
             py::arg("image"),
             py::arg("weight"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("bias", &PyGraph::bias,
             py::arg_v("name", "bias"),
             py::arg("input"),
             py::arg("bias"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("scale", &PyGraph::scale,
             py::arg_v("name", "scale"),
             py::arg("input"),
             py::arg("scale"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("relu", &PyGraph::relu,
             py::arg_v("name", "relu"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("elu", &PyGraph::elu,
             py::arg_v("name", "elu"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("gelu", &PyGraph::gelu,
             py::arg_v("name", "gelu"),
             py::arg("input"),
             py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
        )
        .def("scaled_dot_product_attention", &PyGraph::scaled_dot_product_attention
             , py::arg("q")
             , py::arg("k")
             , py::arg("v")
             , py::arg("seq_len_q")
             , py::arg("seq_len_k")
             , py::arg("is_inference")
             , py::arg_v("scale_k", nullptr)
             , py::arg_v("bias", nullptr)
             , py::arg_v("use_padding_mask", false)
             , py::arg_v("use_causal_mask", false)
             , py::arg_v("dropout", py::none())
             , py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
             , py::arg_v("name", py::none())
        )
        .def("scaled_dot_product_flash_attention", &PyGraph::scaled_dot_product_flash_attention
             , py::arg("q")
             , py::arg("k")
             , py::arg("v")
             , py::arg_v("seq_q", nullptr)
             , py::arg_v("seq_k", nullptr)
             , py::arg("is_inference")
             , py::arg_v("scale_k", nullptr)
             , py::arg_v("bias", nullptr)
             , py::arg_v("use_padding_mask", false)
             , py::arg_v("use_alibi_mask", false)
             , py::arg_v("use_causal_mask", false)
             , py::arg_v("dropout", py::none())
             , py::arg_v("compute_data_type", cudnn_frontend::DataType_t::NOT_SET)
             , py::arg_v("name", py::none())
        )
        .def("build", &PyGraph::build)
        .def("check_support", &PyGraph::check_support)
        .def("get_workspace_size", &PyGraph::get_workspace_size)
        .def("execute", &PyGraph::execute)
        .def("__repr__", [](PyGraph const& pygraph){
            std::stringstream ss;
            ss << json{pygraph.graph};
            return ss.str();
        });
}