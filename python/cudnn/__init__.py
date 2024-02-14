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

from .datatypes import _torch_to_cudnn_data_type 

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
    
    # Convert data type to cudnn
    if type(data_type) != _compiled_module.data_type:
        data_type = _torch_to_cudnn_data_type(data_type)

    return self._make_tensor(
        dim = dim,
        stride = stride,
        data_type = data_type,
        is_virtual = is_virtual,
        is_pass_by_value = is_pass_by_value,
        ragged_offset = ragged_offset,
        name = name
    )

def _set_data_type(
    self,
    data_type = data_type.NOT_SET,
):
    # Convert data type to cudnn
    if type(data_type) != _compiled_module.data_type:
        data_type = _torch_to_cudnn_data_type(data_type)
    
    return self._set_data_type(data_type)

_compiled_module.tensor.set_data_type = _set_data_type
pygraph.tensor = _tensor

def _execute(
    self,
    cudnn_to_library_tensor,
    library_workspace
):
    uid_to_tensor_pointer = {}
    for [cudnn_tensor, library_tensor] in cudnn_to_library_tensor.items():
        # cudnn_tensor can be None
        if cudnn_tensor is None:
            continue
        # cudnn_tensor can also be just a uid
        cudnn_tensor_uid = cudnn_tensor if(type(cudnn_tensor) is int) else cudnn_tensor.get_uid()
        uid_to_tensor_pointer[cudnn_tensor_uid] = library_tensor.data_ptr()
    workspace_pointer = library_workspace.data_ptr()
    self._execute(uid_to_tensor_pointer, workspace_pointer)
    
def _execute_plan_at_index(
    self,
    cudnn_to_library_tensor,
    library_workspace,
    index
):
    uid_to_tensor_pointer = {}
    for [cudnn_tensor, library_tensor] in cudnn_to_library_tensor.items():
        # cudnn_tensor can be None
        if cudnn_tensor is None:
            continue
        # cudnn_tensor can also be just a uid
        cudnn_tensor_uid = cudnn_tensor if(type(cudnn_tensor) is int) else cudnn_tensor.get_uid()
        uid_to_tensor_pointer[cudnn_tensor_uid] = library_tensor.data_ptr()
    workspace_pointer = library_workspace.data_ptr()
    self._execute_plan_at_index(uid_to_tensor_pointer, workspace_pointer, index)

pygraph.execute = _execute
pygraph.execute_plan_at_index = _execute_plan_at_index