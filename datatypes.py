from ._cudnn import data_type

torch_available = None
_torch_to_library_dict = None

def is_torch_available():
    global torch_available, _torch_to_library_dict
    if torch_available is None:
        try:
            import torch

            torch_available = True
            _torch_to_library_dict = {
                torch.half: cudnn.data_type.f16,
                torch.float16: cudnn.data_type.f16,
                torch.bfloat16: cudnn.data_type.bf16,
                torch.float: cudnn.data_type.f32,
                torch.float32: cudnn.data_type.f32,
                torch.double: cudnn.data_type.f64,
                torch.float64: cudnn.data_type.f64,
                torch.int8: cudnn.data_type.s8,
                torch.int32: cudnn.data_type.s32,
                torch.uint8: cudnn.data_type.u8,
            }

            def possibly_add_type(torch_type_name, cutlass_type):
                # Only try adding the type if the version of torch being used supports it
                if hasattr(torch, torch_type_name):
                    torch_type = getattr(torch, torch_type_name)
                    _torch_to_library_dict[torch_type] = cutlass_type
                    _library_to_torch_dict[cutlass_type] = torch_type

            possibly_add_type("float8_e4m3fn", cudnn.data_type.e4m3)
            possibly_add_type("float8_e5m2", cudnn.data_type.e5m2)

        except ImportError:
            torch_available = False
            _torch_to_library_dict = {}
    return torch_available

def torch_library_type(inp) -> cudnn.data_type:
    return _torch_to_library_dict.get(inp, None)

def library_type(inp):
    for cvt_fn in [
        torch_library_type,
    ]:
        out = cvt_fn(inp)
        if out is not None:
            return out

    raise Exception(f"No available conversion from type {inp} to a library type.")