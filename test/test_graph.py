import utils
import sys

utils.reportCurrentTime("import_cudnn")
import torch

utils.reportCurrentTime("import_torch")
from typing import Any
from dataclasses import dataclass, asdict, field
from data_types import DataType, convert_datatype, convert_to_torch_type_wrapper
import utils

utils.reportCurrentTime("import_test_graph_deps")


# Globally ensure cudnn is disabled for everything torch related
torch.backends.cudnn.enabled = False


# @brief: Reference code
# @details: the methods mirror cudnn.pygraph methods and class constructors(__init__)
# @note: we can easily replace PytorchReference by CustomReference to use a different reference framework (one LoC change in test_graph below)
class PytorchReference:
    # @brief: run convolution without bias
    # @param kwargs: these are the named parameters used in the associated cudnn.pygraph.conv function
    #   The only difference is that the input tensors are replaced by pytorch tensors
    # @param test_tensor_out_list: a list of test_tensor instances. Some reference functions may need this (e.g., reduction)
    # @details: all this function needs to do is unpack the cudnn.pygraph function arguments and pass them to the pytorch equivalent
    @staticmethod
    def conv_fprop(kwargs, test_tensor_out_list):
        # determine whether we need 2d or 3d convolution

        # Need WAR for int8 conv2d kernels. PyT does not supported them.
        # https://github.com/pytorch/pytorch/issues/63518
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))

        if len(kwargs["image"].shape) == 4:
            output = torch.nn.functional.conv2d(
                kwargs["image"],
                kwargs["weight"],
                bias=None,
                padding=kwargs["padding"],
                stride=kwargs["stride"],
                dilation=kwargs["dilation"],
            )
        elif len(kwargs["image"].shape) == 5:
            output = torch.nn.functional.conv3d(
                kwargs["image"],
                kwargs["weight"],
                bias=None,
                padding=kwargs["padding"],
                stride=kwargs["stride"],
                dilation=kwargs["dilation"],
            )
        else:
            assert False

        return [output.to(dtype=dtype)]

    @staticmethod
    def conv_dgrad(kwargs, test_tensor_out_list):
        input_size = test_tensor_out_list[0].dim
        dX = torch.nn.grad.conv2d_input(
            input_size,
            kwargs["filter"],
            kwargs["loss"],
            padding=kwargs["padding"],
            stride=kwargs["stride"],
            dilation=kwargs["dilation"],
        )
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [dX.to(dtype=dtype)]

    @staticmethod
    def conv_wgrad(kwargs, test_tensor_out_list):
        filter_dim_size = test_tensor_out_list[0].dim
        dW = torch.nn.grad.conv2d_weight(
            kwargs["image"],
            filter_dim_size,
            kwargs["loss"],
            kwargs["stride"],
            kwargs["padding"],
            kwargs["dilation"],
        )
        return [dW]

    @staticmethod
    def identity(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [kwargs["input"].to(dtype=dtype)]

    @staticmethod
    def reciprocal(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [torch.reciprocal(kwargs["input"]).to(dtype=dtype)]

    @staticmethod
    def atan2(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [torch.atan2(kwargs["a"], kwargs["b"]).to(dtype=dtype)]

    # @brief: run relu
    # @details: unpack the cudnn.pygraph.relu parameters and pass them to the pytorch equivalent
    @staticmethod
    def relu(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [torch.nn.functional.relu(kwargs["input"]).to(dtype=dtype)]

    @staticmethod
    def relu_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        input_tensor.requires_grad_(True)
        relu_output = torch.nn.functional.relu(input_tensor)
        relu_output.backward(loss_tensor)
        dX = input_tensor.grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def elu(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        input_tensor = kwargs["input"]
        alpha_tensor = kwargs["b"] if "b" in kwargs else 1.0
        if isinstance(alpha_tensor, torch.Tensor) and alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.item()
        elu_output = torch.nn.functional.elu(input_tensor, alpha=alpha_tensor)
        return [elu_output.to(dtype=dtype)]

    @staticmethod
    def elu_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        alpha_tensor = kwargs["b"] if "b" in kwargs else 1.0
        if isinstance(alpha_tensor, torch.Tensor) and alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.item()
        input_tensor.requires_grad_(True)
        elu_output = torch.nn.functional.elu(input_tensor, alpha=alpha_tensor)
        elu_output.backward(loss_tensor)
        dX = input_tensor.grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def gelu(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [torch.nn.functional.gelu(kwargs["input"]).to(dtype=dtype)]

    @staticmethod
    def gelu_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        input_tensor.requires_grad_(True)
        gelu_output = torch.nn.functional.gelu(input_tensor)
        gelu_output.backward(loss_tensor)
        dX = input_tensor.grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def gelu_approx_tanh(kwargs, test_tensor_out_list):
        return PytorchReference.gelu(kwargs, test_tensor_out_list)

    @staticmethod
    def gelu_approx_tanh_backward(kwargs, test_tensor_out_list):
        return PytorchReference.gelu_backward(kwargs, test_tensor_out_list)

    @staticmethod
    def sigmoid(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        return [torch.nn.functional.sigmoid(kwargs["input"]).to(dtype=dtype)]

    @staticmethod
    def sigmoid_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        input_tensor.requires_grad_(True)
        sigmoid_output = torch.nn.functional.sigmoid(input_tensor)
        sigmoid_output.backward(loss_tensor)
        dX = input_tensor.grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def swish(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        beta = kwargs["b"] if "b" in kwargs else 1.0
        input_tensor = kwargs["a"]
        swish_output = input_tensor * torch.sigmoid(beta * input_tensor)
        return [swish_output.to(dtype=dtype)]

    @staticmethod
    def swish_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        beta = kwargs["b"] if "b" in kwargs else 1.0
        sigmoid_beta_x = torch.sigmoid(beta * input_tensor)
        # Compute derivative of Swish: sigma(beta * x) + beta * x * sigma(beta * x) * (1 - sigma(beta * x))
        swish_grad = sigmoid_beta_x + beta * input_tensor * sigmoid_beta_x * (
            1 - sigmoid_beta_x
        )
        dX = loss_tensor * swish_grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def softplus(kwargs, test_tensor_out_list):
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        beta = kwargs["b"] if "b" in kwargs else 1.0
        input_tensor = kwargs["a"]
        # First compute log(1 + exp(β*x))
        softplus_output = torch.nn.functional.softplus(beta * input_tensor)
        # Then divide by β
        softplus_output = softplus_output / beta
        return [softplus_output.to(dtype=dtype)]

    @staticmethod
    def softplus_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        beta = kwargs["b"] if "b" in kwargs else 1.0
        input_tensor.requires_grad_(True)
        softplus_output = torch.nn.functional.softplus(beta * input_tensor) / beta
        softplus_output.backward(loss_tensor)
        dX = input_tensor.grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def batchnorm(kwargs, test_tensor_out_list):
        is_training = True
        momentum = kwargs["momentum"].item()
        epsilon = kwargs["epsilon"].item()
        # TODO(https://nvbugs/4272638): A bug with the cudnn backend disabled prevents correct behavior of batchnorm
        # As a WAR temporarily enable the cudnn backend.
        cudnn_enabled_before = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = True
        try:
            output = torch.nn.functional.batch_norm(
                kwargs["input"],
                kwargs["in_running_mean"],
                kwargs["in_running_var"],
                weight=kwargs["scale"],
                bias=kwargs["bias"],
                training=is_training,
                momentum=momentum,
                eps=epsilon,
            )
        except Exception as e:
            raise e
        finally:
            # Set back the backend to what it was before
            torch.backends.cudnn.enabled = cudnn_enabled_before

        output = [output]

        # torch's implementation only returns 1 output.
        # Filling out the others with an amount of None's and have the reference check deal with it
        output.extend([None] * 4)
        return output

    @staticmethod
    def matmul(kwargs, test_tensor_out_list):
        lsh_str = "A"
        rsh_str = "B"
        compute_type = eval(
            convert_to_torch_type_wrapper(
                kwargs["compute_data_type"]
                if "compute_data_type" in kwargs
                else test_tensor_out_list[0].data_type
            )
        )
        out_type = eval(
            convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type)
        )

        # Do computation in compute precision
        # except for fp8
        compute_type = (
            compute_type if not compute_type == torch.float8_e4m3fn else torch.float
        )
        compute_result = torch.bmm(
            kwargs[lsh_str].to(dtype=compute_type),
            kwargs[rsh_str].to(dtype=compute_type),
        )

        # Store both compute and output precision results
        test_tensor_out_list[0].compute_data = compute_result
        test_tensor_out_list[0].ref_data = compute_result.to(dtype=out_type)

        return [test_tensor_out_list[0].ref_data]

    @staticmethod
    def bias(kwargs, test_tensor_out_list):
        output = torch.add(kwargs["input"], kwargs["bias"])
        return [output]

    @staticmethod
    def add(kwargs, test_tensor_out_list):
        lsh_str = "a"
        rsh_str = "b"
        compute_type = eval(
            convert_to_torch_type_wrapper(
                kwargs["compute_data_type"]
                if "compute_data_type" in kwargs
                else test_tensor_out_list[0].data_type
            )
        )
        if kwargs[lsh_str].dtype != compute_type:
            kwargs[lsh_str] = kwargs[lsh_str].to(dtype=compute_type)
        if kwargs[rsh_str].dtype != compute_type:
            kwargs[rsh_str] = kwargs[rsh_str].to(dtype=compute_type)
        out_type = eval(
            convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type)
        )
        compute_result = torch.add(kwargs[lsh_str], kwargs[rsh_str]).to(
            dtype=compute_type
        )

        test_tensor_out_list[0].compute_data = compute_result
        test_tensor_out_list[0].ref_data = compute_result.to(dtype=out_type)
        return [test_tensor_out_list[0].ref_data]

    @staticmethod
    def sub(kwargs, test_tensor_out_list):
        output = torch.sub(kwargs["a"], kwargs["b"])
        return [output]

    @staticmethod
    def mul(kwargs, test_tensor_out_list):
        lsh_str = "a"
        rsh_str = "b"
        compute_type = eval(
            convert_to_torch_type_wrapper(
                kwargs["compute_data_type"]
                if "compute_data_type" in kwargs
                else test_tensor_out_list[0].data_type
            )
        )
        if kwargs[lsh_str].dtype != compute_type:
            kwargs[lsh_str] = kwargs[lsh_str].to(dtype=compute_type)
        if kwargs[rsh_str].dtype != compute_type:
            kwargs[rsh_str] = kwargs[rsh_str].to(dtype=compute_type)
        out_type = eval(
            convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type)
        )
        compute_result = torch.mul(kwargs[lsh_str], kwargs[rsh_str]).to(
            dtype=compute_type
        )

        test_tensor_out_list[0].compute_data = compute_result
        test_tensor_out_list[0].ref_data = compute_result.to(dtype=out_type)
        return [test_tensor_out_list[0].ref_data]

    @staticmethod
    def max(kwargs, test_tensor_out_list):
        output = torch.max(kwargs["input0"], kwargs["input1"])
        return [output]

    @staticmethod
    def min(kwargs, test_tensor_out_list):
        output = torch.min(kwargs["input0"], kwargs["input1"])
        return [output]

    @staticmethod
    def pow(kwargs, test_tensor_out_list):
        output = torch.pow(kwargs["input0"], kwargs["input1"])
        return [output]

    @staticmethod
    def mod(kwargs, test_tensor_out_list):
        output = torch.fmod(kwargs["input0"], kwargs["input1"])
        return [output]

    @staticmethod
    def div(kwargs, test_tensor_out_list):
        output = torch.div(kwargs["a"], kwargs["b"])
        return [output]

    @staticmethod
    def add_square(kwargs, test_tensor_out_list):
        output = torch.add(kwargs["a"], torch.square(kwargs["b"]))
        return [output]

    @staticmethod
    def tanh(kwargs, test_tensor_out_list):
        output = torch.tanh(kwargs["input"])
        return [output]

    @staticmethod
    def tanh_backward(kwargs, test_tensor_out_list):
        input_tensor = kwargs["input"]
        loss_tensor = kwargs["loss"]
        input_tensor.requires_grad_(True)
        tanh_output = torch.tanh(input_tensor)
        tanh_output.backward(loss_tensor)
        dX = input_tensor.grad
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]

    @staticmethod
    def abs(kwargs, test_tensor_out_list):
        output = torch.abs(kwargs["input"])
        return [output]

    @staticmethod
    def ceil(kwargs, test_tensor_out_list):
        output = torch.ceil(kwargs["input"])
        return [output]

    @staticmethod
    def floor(kwargs, test_tensor_out_list):
        output = torch.floor(kwargs["input"])
        return [output]

    @staticmethod
    def cos(kwargs, test_tensor_out_list):
        output = torch.cos(kwargs["input"])
        return [output]

    @staticmethod
    def sin(kwargs, test_tensor_out_list):
        output = torch.sin(kwargs["input"])
        return [output]

    @staticmethod
    def tan(kwargs, test_tensor_out_list):
        output = torch.tan(kwargs["input"])
        return [output]

    @staticmethod
    def exp(kwargs, test_tensor_out_list):
        output = torch.exp(kwargs["input"])
        return [output]

    @staticmethod
    def log(kwargs, test_tensor_out_list):
        output = torch.log(kwargs["input"])
        return [output]

    @staticmethod
    def neg(kwargs, test_tensor_out_list):
        output = torch.neg(kwargs["input"])
        return [output]

    @staticmethod
    def sqrt(kwargs, test_tensor_out_list):
        output = torch.sqrt(kwargs["input"])
        return [output]

    @staticmethod
    def rsqrt(kwargs, test_tensor_out_list):
        output = torch.rsqrt(kwargs["input"])
        return [output]

    @staticmethod
    def erf(kwargs, test_tensor_out_list):
        output = torch.erf(kwargs["input"])
        return [output]

    @staticmethod
    def cmp_lt(kwargs, test_tensor_out_list):
        output = torch.lt(kwargs["input"], kwargs["comparison"])
        return [output]

    @staticmethod
    def cmp_gt(kwargs, test_tensor_out_list):
        output = torch.gt(kwargs["input"], kwargs["comparison"])
        return [output]

    @staticmethod
    def cmp_ge(kwargs, test_tensor_out_list):
        output = torch.ge(kwargs["input"], kwargs["comparison"])
        return [output]

    @staticmethod
    def cmp_le(kwargs, test_tensor_out_list):
        output = torch.le(kwargs["input"], kwargs["comparison"])
        return [output]

    @staticmethod
    def cmp_eq(kwargs, test_tensor_out_list):
        output = torch.eq(kwargs["input"], kwargs["comparison"])
        return [output]

    @staticmethod
    def cmp_ne(kwargs, test_tensor_out_list):
        output = torch.ne(kwargs["input"], kwargs["comparison"])
        return [output]

    @staticmethod
    def logical_not(kwargs, test_tensor_out_list):
        output = torch.logical_not(kwargs["input"])
        return [output]

    @staticmethod
    def logical_and(kwargs, test_tensor_out_list):
        output = torch.logical_and(kwargs["a"], kwargs["b"])
        return [output]

    @staticmethod
    def logical_or(kwargs, test_tensor_out_list):
        output = torch.logical_or(kwargs["a"], kwargs["b"])
        return [output]

    @staticmethod
    def binary_select(kwargs, test_tensor_out_list):
        output = torch.where(kwargs["selector"], kwargs["a"], kwargs["b"])
        return [output]

    @staticmethod
    def reduction(kwargs, test_tensor_out_list):
        # todo(@mbreughe): set default data types for output tensors based on pygraph settings
        dtype = eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))

        out_dims = test_tensor_out_list[0].dim

        axis = []
        for dim_idx, dim_val in enumerate(out_dims):
            if dim_val == 1:
                axis.append(dim_idx)

        if "reduction_mode.MAX" in kwargs["mode"]:
            output = kwargs["input"].amax(dim=tuple(axis), keepdim=True).to(dtype=dtype)
        elif "reduction_mode.MIN" in kwargs["mode"]:
            output = kwargs["input"].amin(dim=tuple(axis), keepdim=True).to(dtype=dtype)
        elif "reduction_mode.AMAX" in kwargs["mode"]:
            output = kwargs["input"].amax(dim=tuple(axis)).to(dtype=dtype)
        elif "reduction_mode.ADD" in kwargs["mode"]:
            output = kwargs["input"].sum(dim=tuple(axis)).to(dtype=dtype)
        else:
            raise ValueError(f"Unhanlded reduction mode")
        # output = kwargs["input"].sum(dim=axis)
        # output = output.type(dtype)

        output = output.reshape(out_dims)
        return [output]

    @staticmethod
    def relu_backward(kwargs, test_tensor_out_list):
        dX = torch.where(kwargs["input"] > 0, kwargs["loss"], 0)
        dX = dX.to(
            eval(convert_to_torch_type_wrapper(test_tensor_out_list[0].data_type))
        )
        return [dX]


# Base class for Tensor and operation nodes
class test_node:
    __test__ = False

    def __init__(self, name):
        self.name = name
        self.visited = False
        self.is_explicit_output = False
        self.consumer_nodes = []
        self.producer_nodes = []

    def set_output_node(self, is_output):
        self.is_explicit_output = is_output

    def is_output_node(self):
        return self.is_explicit_output or len(self.consumer_nodes) == 0

    def clear_meta_data(self):
        self.visited = False

    def add_producer_node(self, node):
        self.producer_nodes.append(node)
        node.consumer_nodes.append(self)

    def set_visited(self):
        self.visited = True

    def is_visited(self):
        return self.visited

    def is_prereq_satisfied(self):
        prereq_satisfied = True
        # Since an input tensor doesnt have any producers, this will result in true
        for node in self.producer_nodes:
            prereq_satisfied = prereq_satisfied and node.is_visited()
        return prereq_satisfied

    # Function that needs to be overriden by its child classes
    def run_cudnn_code(self, cudnn_graph):
        print("NOT IMPLEMENTED")

    def build_cudnntree_recursive(self, cudnn_graph):
        if not self.is_visited() and self.is_prereq_satisfied():
            # print ("Checking {}".format(self.name))
            self.run_cudnn_code(cudnn_graph)
            self.set_visited()
            for node in self.consumer_nodes:
                node.build_cudnntree_recursive(cudnn_graph)

    def run_reftree_recursive(self):
        if not self.is_visited() and self.is_prereq_satisfied():
            self.run_ref()
            self.set_visited()
            for node in self.consumer_nodes:
                node.run_reftree_recursive()


class operation(test_node):

    # @param pyCudnnOp: cudnn.pygraph operation (e.g., cudnn.pygraph.conv)
    # @param ref_func: reference function for the associated cudnn_op
    # @param name: name for this operation (could be passed by kwargs as well)
    def __init__(self, cudnn_op, ref_func, name, num_outputs=1, opName=None):
        super().__init__(name)

        self.cudnn_op = cudnn_op
        self.ref_func = ref_func
        if opName is not None:
            self.op_name = opName

        self.output = []

        if num_outputs > 1:
            for i in range(num_outputs):
                self.output.append(test_tensor("{}_out_{}".format(name, i), self))
        else:
            self.output.append(test_tensor("{}_out".format(name), self))

    # @param kwargs: parameters for the associated cudnn_op
    # @details: All this function needs to do is:
    #   * add the correct producers
    #   * store the kwargs (named parameters from the associated cudnn_op)
    def set_kwargs(self, kwargs):
        self.kwargs = kwargs
        # TODO(@mbreughe): Does .values() make an unnecessary copy?
        for v in self.kwargs.values():
            if isinstance(v, test_tensor):
                self.add_producer_node(v.parent_op)

    # @brief: Run the cudnn node
    # TODO(@mbreughe): extended to multiple output tensors
    def run_cudnn_code(self, cudnn_graph):

        # Besides input tensors, all kwargs can just be passed through to the cudnn method.
        # For the input tensor we need to extract the test_tensor's cudnn tensor.
        # Therefore: copy all kwargs, except for test_tensors
        # TODO(@mbreughe) Avoid copying these kwargs twice (ref and cudnn). Let's do this in the init function once
        if self.cudnn_op is None:
            return
        import cudnn

        new_kwargs = {
            x: self.kwargs[x]
            for x in self.kwargs
            if not isinstance(self.kwargs[x], test_tensor)
        }
        for x in self.kwargs:
            if isinstance(self.kwargs[x], test_tensor):
                new_kwargs[x] = self.kwargs[x].cudnn_tensor

        if "compute_data_type" in new_kwargs and isinstance(
            new_kwargs["compute_data_type"], DataType
        ):
            new_kwargs["compute_data_type"] = eval(
                convert_datatype(new_kwargs["compute_data_type"], "cudnn")
            )
        if new_kwargs.get("compute_data_type", None) == cudnn.data_type.INT8:
            new_kwargs["compute_data_type"] = cudnn.data_type.INT32

        if (
            self.cudnn_op.__name__ == "reduction"
            and "mode" in new_kwargs
            and isinstance(new_kwargs["mode"], str)
        ):
            new_kwargs["mode"] = eval(new_kwargs["mode"])
        cudnn_res = self.cudnn_op(cudnn_graph, **new_kwargs)
        if self.cudnn_op.__name__ == "reduction" and "mode" in new_kwargs:
            self.kwargs["mode"] = str(new_kwargs["mode"])

        # in case we have multiple outputs
        if isinstance(cudnn_res, list):
            assert len(cudnn_res) == len(self.output)
            for output, cudnn_out in zip(self.output, cudnn_res):
                output.cudnn_tensor = cudnn_out
        else:
            self.output[0].cudnn_tensor = cudnn_res

    def run_ref(self):
        new_kwargs = {
            x: self.kwargs[x]
            for x in self.kwargs
            if not isinstance(self.kwargs[x], test_tensor)
        }
        for x in self.kwargs:
            if isinstance(self.kwargs[x], test_tensor):
                tensor_data = (
                    self.kwargs[x].compute_data
                    if self.kwargs[x].compute_data is not None
                    else self.kwargs[x].ref_data
                )
                new_kwargs[x] = tensor_data
        # Note: we could choose to have the ref func set the output
        ref_output = self.ref_func(new_kwargs, self.output)

        for output, ref_out in zip(self.output, ref_output):
            output.ref_data = ref_out


# TODO(@mbreughe): Support multiple distributions (see json graph's fill type)
class random_tensor_generator(test_node):
    __test__ = False

    def __init__(self, kwargs, name, io_data_type):
        super().__init__(name)
        self.kwargs = kwargs

        self.output = [test_tensor(name + "_out", self)]
        if "isByValue" in self.kwargs:
            self.is_by_value = self.kwargs["isByValue"]
            self.output[0].set_is_by_value(self.is_by_value)

        data_type = (
            io_data_type if not "data_type" in self.kwargs else self.kwargs["data_type"]
        )
        self.output[0].set_data_type(data_type)
        self.dist_mean = None
        self.dist_sd = None
        self.init_int = False

    def set_dist_mean_sd(self, mean, sd, init_int=False):
        self.dist_mean = mean
        self.dist_sd = sd
        self.init_int = init_int

    def initialize_torch_tensor(self, torch_dtype, device):
        if torch_dtype == torch.bool:
            self.output[0].ref_data = torch.randint(
                0 if self.dist_mean is None else self.dist_mean,
                2 if self.dist_sd is None else self.dist_sd,
                self.kwargs["dim"],
                requires_grad=False,
                device=device,
                dtype=torch_dtype,
            )
        elif torch_dtype == torch.int8:
            self.output[0].ref_data = torch.randint(
                -2 if self.dist_mean is None else self.dist_mean,
                3 if self.dist_sd is None else self.dist_sd,
                self.kwargs["dim"],
                requires_grad=False,
                device=device,
                dtype=torch_dtype,
            )
        elif torch_dtype == torch.int32:
            self.output[0].ref_data = torch.randint(
                -2 if self.dist_mean is None else self.dist_mean,
                3 if self.dist_sd is None else self.dist_sd,
                self.kwargs["dim"],
                requires_grad=False,
                device=device,
                dtype=torch_dtype,
            )
        elif torch_dtype == torch.float8_e4m3fn:
            self.output[0].ref_data = torch.randint(
                -2 if self.dist_mean is None else self.dist_mean,
                3 if self.dist_sd is None else self.dist_sd,
                self.kwargs["dim"],
                requires_grad=False,
                device=device,
            ).to(dtype=torch.float8_e4m3fn)
        else:
            if self.init_int:
                min_value = -2 if self.dist_mean is None else int(self.dist_mean)
                max_value = 3 if self.dist_sd is None else int(self.dist_sd)

                # Ensure min_value is less than max_value
                if min_value >= max_value:
                    min_value, max_value = (
                        min(min_value, max_value),
                        max(min_value, max_value) + 1,
                    )

                self.output[0].ref_data = torch.randint(
                    min_value,
                    max_value,
                    self.kwargs["dim"],
                    requires_grad=False,
                    device=device,
                    dtype=torch_dtype,
                )
            else:
                self.output[0].ref_data = torch.normal(
                    0.5 if self.dist_mean is None else self.dist_mean,
                    0.5 if self.dist_sd is None else self.dist_sd,
                    self.kwargs["dim"],
                    requires_grad=False,
                    device=device,
                    dtype=torch_dtype,
                )
        if (
            "stride" in self.kwargs
            and "is_tensor_ir" in self.kwargs
            and self.kwargs["is_tensor_ir"]
        ):
            self.output[0].ref_data = torch.as_strided(
                self.output[0].ref_data,
                self.output[0].ref_data.size(),
                stride=self.kwargs["stride"],
            )

    def initialize_random_tensor(self):
        if self.output[0].ref_data is None:
            # The default random generator results in numerical issues
            torch_dtype = eval(convert_to_torch_type_wrapper(self.output[0].data_type))
            try:
                self.initialize_torch_tensor(torch_dtype, "cuda")
            except Exception as error:
                self.initialize_torch_tensor(torch_dtype, "cpu")

            if self.get_layout() == "NHWC":
                size = self.output[0].ref_data.size()
                if len(size) == 4:
                    self.output[0].ref_data = self.output[0].ref_data.to(
                        memory_format=torch.channels_last
                    )
                elif len(size) == 5:
                    # Technically we could reuse this for len(size) == 4, but some tests are showing small numerical errors
                    stride = utils.create_nhwc_strides(size)
                    self.output[0].ref_data = torch.as_strided(
                        self.output[0].ref_data, size, stride
                    )
                else:
                    assert len(size) < 6 and len(size) > 3

    def get_value(self):
        self.initialize_random_tensor()
        return self.output[0].ref_data

    def get_layout(self):
        # TODO(mbreughe): Assume NCHW layout by default for now
        return "NCHW" if not "layout" in self.kwargs else self.kwargs["layout"]

    def run_cudnn_code(self, cudnn_graph):
        if "cudnn" not in sys.modules:
            print("Skip run_cudnn_code()")
            return
        import cudnn

        self.initialize_random_tensor()
        self.output[0].cudnn_tensor = cudnn_graph.tensor(
            name=self.name,
            dim=self.output[0].ref_data.size(),
            stride=self.output[0].ref_data.stride(),
            data_type=(
                eval(convert_datatype(self.output[0].data_type, "cudnn"))
                if isinstance(self.output[0].data_type, DataType)
                else self.output[0].data_type
            ),
        )

    def run_ref(self):
        return self.get_value()


# TODO(@mbreughe): maybe subclass this from random_tensor_generator
# TODO(@mbreughe): consider putting the layout-kwargs as a separate helper function instead of setting it in the kwargs
class ConstantTensor(test_node):
    def __init__(self, kwargs, name, io_data_type, value):
        super().__init__(name)
        self.kwargs = kwargs

        self.output = [test_tensor(name + "_out", self)]

        data_type = (
            io_data_type if not "data_type" in self.kwargs else self.kwargs["data_type"]
        )
        self.output[0].set_data_type(data_type)

        self.value = value

    def instantiate(self):
        if self.output[0].ref_data is None:
            self.output[0].ref_data = torch.full(
                self.kwargs["dim"],
                self.value,
                requires_grad=False,
                device="cpu",
                dtype=eval(convert_to_torch_type_wrapper(self.output[0].data_type)),
            )

            if self.get_layout == "NHWC":
                self.output[0].ref_data = self.output.ref_data.to(
                    memory_format=torch.channels_last
                )

    def get_value(self):
        self.instantiate()
        return self.output[0].ref_data

    def get_layout(self):
        # TODO(mbreughe): Assume NCHW layout by default for now
        return "NCHW" if not "layout" in self.kwargs else self.kwargs["layout"]

    def run_cudnn_code(self, cudnn_graph):
        if "cudnn" not in sys.modules:
            print("Skip run_cudnn_code()")
            return
        import cudnn

        self.instantiate()
        self.output[0].cudnn_tensor = cudnn_graph.tensor(
            name=self.name,
            dim=self.output[0].ref_data.size(),
            stride=self.output[0].ref_data.stride(),
            data_type=(
                eval(convert_datatype(self.output[0].data_type, "cudnn"))
                if isinstance(self.output[0].data_type, DataType)
                else self.output[0].data_type
            ),
            is_pass_by_value=True,
        )

    def run_ref(self):
        return self.get_value()


class test_tensor:
    __test__ = False

    def __init__(self, name, parent_op):
        self.name = name
        # The cudnn.pygraph.tensor instance associated with test_tensor
        self.cudnn_tensor = None
        # The reference data for this tensor
        self.ref_data = None
        # The compute data for this tensor
        self.compute_data = None
        # Initialize no data type is specified
        self._data_type = None

        self.parent_op = parent_op

        self.is_by_value = False

        self.is_virtual = True

    @property
    def data_type(self):
        if self._data_type is not None:
            return self._data_type
        elif self.cudnn_tensor is not None:
            return self.cudnn_tensor.get_data_type()
        else:
            return None

    @data_type.setter
    def data_type(self, dtype):
        self._data_type = dtype

        # Apply it immediately if a cudnn tensor was already created
        if self.cudnn_tensor is not None:
            self.cudnn_tensor.set_data_type(dtype)

    def set_is_virtual(self, is_virtual):
        self.is_virtual = is_virtual

    # Convenience wrapper-function to mimic cudnn.tensor.set_data_type
    def set_data_type(self, dtype):
        # Invoke data_type.setter
        self.data_type = dtype

    def set_dim(self, dim):
        self.dim = dim

        if self.cudnn_tensor is not None:
            self.cudnn_tensor.set_dim(dim)

    def set_stride(self, stride):
        self.stride = stride

        if self.cudnn_tensor is not None:
            self.cudnn_tensor.set_stride(stride)

    def set_is_by_value(self, is_by_value):
        self.is_by_value = is_by_value

    # TODO(@mbreughe): refactor this to avoid looking up strings
    def apply_modifiers(self):
        # If we ever specified a data type, apply it
        if self.data_type is not None and self.cudnn_tensor is not None:
            import cudnn

            self.cudnn_tensor.set_data_type(
                eval(convert_datatype(self.data_type, "cudnn"))
                if isinstance(self.data_type, DataType)
                else self.data_type
            )

        if "dim" in dir(self) and self.cudnn_tensor is not None:
            self.cudnn_tensor.set_dim(self.dim)

        if "stride" in dir(self) and self.cudnn_tensor is not None:
            self.cudnn_tensor.set_stride(self.stride)

    def __repr__(self):
        rep = "[Tensor] {}".format(self.name)
        if "dim" in dir(self):
            rep += " - dim: {}".format(self.dim)

        if "stride" in dir(self):
            rep += " - stride: {}".format(self.stride)

        if "data_type" in dir(self):
            rep += " - dtype: {}".format(self.data_type)

        return rep


class test_graph:
    __test__ = False

    # Add data types, custom test name ,etc.
    def __init__(
        self,
        io_data_type=DataType.HALF,
        intermediate_data_type=DataType.FLOAT,
        compute_data_type=DataType.FLOAT,
        is_tensor_ir=False,
    ):
        # TODO(@barretw): ensure output is deterministic and reproducible for L4 tests
        torch.manual_seed(0)
        self.uid_counter = 0
        self.nodes = []
        self.entrance_nodes = []
        self.graph_name = "test_graph"
        self.output_tensors = []
        self.io_data_type = io_data_type
        self.intermediate_data_type = intermediate_data_type
        self.compute_data_type = compute_data_type
        self.is_tensor_ir = is_tensor_ir

    def getOutputs(self):
        return self.output_tensors

    # @brief: Add a convolution node to the graph

    # @brief: Add an input tensor to the graph
    def test_tensor(self, **kwargs):
        # Create a name if none provided
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.create_unique_name("Tensor")

        if "data_type" in kwargs:
            data_type = kwargs["data_type"]
        else:
            data_type = self.io_data_type
        if self.is_tensor_ir:
            kwargs["is_tensor_ir"] = self.is_tensor_ir
        test_tensor = random_tensor_generator(kwargs, name, data_type)
        self.nodes.append(test_tensor)
        # we are assuming only input tensors are explicitly created
        self.entrance_nodes.append(test_tensor)
        return test_tensor

    def tensor(self, **kwargs):
        return self.test_tensor(**kwargs).output[0]

    # @brief: utility function to create unique names for the graph
    def create_unique_name(self, prefix):
        name = prefix + "_{}".format(self.uid_counter)
        self.uid_counter += 1
        return name

    def conv_fprop(self, **kwargs):
        return self.create_and_add_operation(kwargs, "conv_fprop")

    def conv_dgrad(self, **kwargs):
        return self.create_and_add_operation(kwargs, "conv_dgrad")

    def conv_wgrad(self, **kwargs):
        return self.create_and_add_operation(kwargs, "conv_wgrad")

    # @brief: Add a relu to the graph
    def relu(self, **kwargs):
        return self.create_and_add_operation(kwargs, "relu")

    def batchnorm(self, **kwargs):
        return self.create_and_add_operation(kwargs, "batchnorm")

    def matmul(self, **kwargs):
        return self.create_and_add_operation(kwargs, "matmul")

    def add(self, **kwargs):
        return self.create_and_add_operation(kwargs, "add")

    def bias(self, **kwargs):
        return self.create_and_add_operation(kwargs, "bias")

    def reduction(self, **kwargs):
        return self.create_and_add_operation(kwargs, "reduction")

    # @brief: Create an operation, pass through the kwargs and set up dependencies
    # @param kwargs: the named arguments passed to a cudnn function
    # @param opName: the operation name, this is used to build the pytorch reference graph,
    #                e.g. "conv_fprop", this should be the same as the PyTorch reference function name
    # @return: test_tensor (the output from the added operation)
    def create_and_add_operation(self, kwargs, opName=None, op=None):
        opName = opName if opName is not None else op.__name__
        if "name" in kwargs:
            name = kwargs["name"]
        else:
            name = self.create_unique_name(opName)

        node = test_graph.create_operation(opName, name, op)
        node.set_kwargs(kwargs)
        self.nodes.append(node)

        # cudnn returns either a single tensor, or a list of tensors
        # internally, we always store a list to allow for generalization
        if len(node.output) == 1:
            return node.output[0]
        else:
            return node.output

    # @brief: In esssence, this function is a factory function:
    # @param cudnn_op: the cudnn operation (e.g., cudnn.pygraph.conv)
    # @return: operation
    # it builds an operation by:
    #   * discovering the reference function based on the name of the cudnn op
    @staticmethod
    def create_operation(opName, name, op=None):
        # Fetch the reference function from the reference framework
        # Note that we use PytorchReference here, but we can make this arbitrary
        # TODO(@mbreughe): Add error handling code in case the function is not found
        ref_func = getattr(PytorchReference, opName)

        # TODO(@mbreughe): automate this
        num_outputs = 1
        if opName == "batchnorm":
            num_outputs = 5

        node = operation(op, ref_func, name, num_outputs, opName)
        return node

    # @brief: Create a pycudnn node from the legacy_op
    def create_test_graph_node(self, legacy_op):
        name = legacy_op.get_name()
        opName = legacy_op.get_pycudnn_operation_name()
        return test_graph.create_operation(opName, name)

    # @brief: Utility function
    def clear_node_meta_data(self):
        for node in self.nodes:
            node.clear_meta_data()

    # @brief: Utility function to discover output nodes

    def apply_modifiers_to_node_output_tensors(self):
        for node in self.nodes:
            for tensor in node.output:
                tensor.apply_modifiers()

    def set_io_data_type(self, data_type):
        self.io_data_type = data_type

    def set_intermediate_data_type(self, data_type):
        self.intermediate_data_type = data_type

    def set_compute_data_type(self, data_type):
        self.compute_data_type = data_type

    # @brief: Run the reference for the associated graph
    # @note: this temporarily modifies the is_visited status of the nodes
    # TODO(@mbreughe): to preserve resources, we could consider clearing intermediate results when they are no longer needed
    # @pre: build_cudnn_graph may need to be invoked first because of its call to apply_modifiers_to_node_output_tensors, which may modify tensor layouts,
    # but also because it sets output nodes
    def calc_reference(self):
        # Clear the "is_visited" status of the nodes
        self.clear_node_meta_data()
        for node in self.entrance_nodes:
            node.run_reftree_recursive()

        output = []
        for node in self.nodes:
            if node.is_output_node():
                for out in node.output:
                    output.append(out.ref_data)

        # Clear the "is_visited" status of the nodes
        self.clear_node_meta_data()
        utils.reportCurrentTime("calc_reference")
        return output
