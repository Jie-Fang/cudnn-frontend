from ..datatypes import _torch_to_cutlass_data_type
import cutlass


def convert_to_cutlass_data_type(torch_data_type):
    if isinstance(torch_data_type, type) and issubclass(
        torch_data_type, cutlass.Numeric
    ):
        return torch_data_type
    elif torch_data_type is not None:
        cutlass_data_type = _torch_to_cutlass_data_type(torch_data_type)
        if cutlass_data_type is None:
            raise ValueError("Unsupported tensor data type")
        return cutlass_data_type
    else:
        raise ValueError("None is not a valid tensor data type")
