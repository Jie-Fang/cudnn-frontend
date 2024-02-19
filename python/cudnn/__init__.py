from ._compiled_module import (        
    backend_version
    , destroy_handle
    , norm_forward_phase
    , reduction_mode
    , behavior_note
    , create_handle
    , get_stream
    , numerical_note
    , set_stream
    , build_plan_policy
    , data_type
    , heur_mode
    , pygraph
    , tensor
)

from .datatypes import (_library_type, _is_torch_tensor)

__version__ = '1.2.0'

def _tensor(
    self,
    dim,
    stride,
    data_type = data_type.NOT_SET,
    is_virtual = False,
    is_pass_by_value = False,
    ragged_offset = None,
    name = ""
):
    return self._make_tensor(
        dim = dim,
        stride = stride,
        data_type = _library_type(data_type),
        is_virtual = is_virtual,
        is_pass_by_value = is_pass_by_value,
        ragged_offset = ragged_offset,
        name = name
    )

def _set_data_type(
    self,
    data_type = data_type.NOT_SET,
):
    return self._set_data_type(_library_type(data_type))

_compiled_module.tensor.set_data_type = _set_data_type
pygraph.tensor = _tensor

def _library_device_pointer(input_tensor):
    # either pass in pointers directly
    if type(input_tensor) is int:
        return input_tensor
    # directly extract data pointer for torch tensors
    elif _is_torch_tensor(input_tensor):
        return input_tensor.data_ptr()
    # fall back to dlpack support by library
    else:
        return _compiled_module._get_data_ptr(input_tensor)

def _execute(
    self,
    tensor_to_device_buffer,
    workspace
):
    uid_to_tensor_pointer = {
        x if type(x) is int else x.get_uid() : _library_device_pointer(pointer)
        for x, pointer in tensor_to_device_buffer.items() if x is not None
    }

    workspace_pointer = _library_device_pointer(workspace)
    self._execute(uid_to_tensor_pointer, workspace_pointer)
    
def _execute_plan_at_index(
    self,
    tensor_to_device_buffer,
    workspace,
    index
):
    uid_to_tensor_pointer = {
        x if type(x) is int else x.get_uid() : _library_device_pointer(pointer)
        for x, pointer in tensor_to_device_buffer.items() if x is not None
    }

    workspace_pointer = _library_device_pointer(workspace)
    self._execute_plan_at_index(uid_to_tensor_pointer, workspace_pointer, index)

pygraph.execute = _execute
pygraph.execute_plan_at_index = _execute_plan_at_index