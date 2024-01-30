from .compiled_module import data_type as cudnn_data_type

torch_available = None
_torch_to_cudnn_data_type_dict = None

def is_torch_available():
    global torch_available, _torch_to_cudnn_data_type_dict
    # this condition ensures that datatype mapping is only created once
    if torch_available is None:
        try:
            import torch

            torch_available = True
            _torch_to_cudnn_data_type_dict = {
                torch.half: cudnn_data_type.HALF,
                torch.float16: cudnn_data_type.HALF,
                torch.bfloat16: cudnn_data_type.BFLOAT16,
                torch.float: cudnn_data_type.FLOAT,
                torch.float32: cudnn_data_type.FLOAT,
                torch.double: cudnn_data_type.DOUBLE,
                torch.float64: cudnn_data_type.DOUBLE,
                torch.int8: cudnn_data_type.INT8,
                torch.int32: cudnn_data_type.INT32,
                torch.int64: cudnn_data_type.INT64,
                torch.uint8: cudnn_data_type.UINT8,
                torch.bool: cudnn_data_type.BOOLEAN,
            }

            def possibly_add_type(torch_type_name, cudnn_type):
                # Only try adding the type if the version of torch being used supports it
                if hasattr(torch, torch_type_name):
                    torch_type = getattr(torch, torch_type_name)
                    _torch_to_cudnn_data_type_dict[torch_type] = cudnn_type

            possibly_add_type("float8_e4m3fn", cudnn_data_type.FP8_E4M3)
            possibly_add_type("float8_e5m2", cudnn_data_type.FP8_E5M2)

        except ImportError:
            torch_available = False
            _torch_to_cudnn_data_type_dict = {}
    return torch_available

# Returns None in case mapping is not available
def torch_to_cudnn_data_type(torch_data_type) -> cudnn_data_type:
    if is_torch_available():
        return _torch_to_cudnn_data_type_dict.get(torch_data_type, None)
    else:
        return None