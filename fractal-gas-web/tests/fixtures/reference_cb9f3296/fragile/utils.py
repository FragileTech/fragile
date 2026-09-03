"""Stub of cb9f3296 src/fragile/utils.py: only the helper core.py imports,
copied verbatim (the original module also pulls panel, PIL and the fragile
typing module, which the reference algorithm does not use)."""
import numpy as np
import numpy
import torch


def numpy_dtype_to_torch_dtype(dtype):
    """Convert a numpy data type to a corresponding torch data type.
    If a torch data type is provided, return it as is.

    Args:
        dtype (numpy.dtype or torch.dtype): The data type to convert.

    Returns:
        torch.dtype: The corresponding torch data type.

    """
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.float16: torch.float16,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
    }
    numpy_to_torch_dtype = {
        np.dtypes.BoolDType: torch.bool,
        np.dtypes.ByteDType: torch.uint8,
        np.dtypes.BytesDType: None,  # No direct equivalent in PyTorch
        np.dtypes.CLongDoubleDType: torch.complex128,  # Closest equivalent
        np.dtypes.Complex128DType: torch.complex128,
        np.dtypes.Complex64DType: torch.complex64,
        np.dtypes.DateTime64DType: None,  # No equivalent in PyTorch
        np.dtypes.Float16DType: torch.float16,
        np.dtypes.Float32DType: torch.float32,
        np.dtypes.Float64DType: torch.float64,
        np.dtypes.Int16DType: torch.int16,
        np.dtypes.Int32DType: torch.int32,
        np.dtypes.Int64DType: torch.int64,
        np.dtypes.Int8DType: torch.int8,
        np.dtypes.IntDType: torch.int32,  # NumPy's int is usually int32, but platform-dependent
        np.dtypes.LongDType: torch.int64,  # NumPy long maps to int64
        np.dtypes.LongDoubleDType: torch.float64,  # Closest match
        np.dtypes.LongLongDType: torch.int64,  # Closest match
        np.dtypes.ObjectDType: None,  # No equivalent in PyTorch
        np.dtypes.ShortDType: torch.int16,
        np.dtypes.StrDType: None,  # No equivalent in PyTorch
        np.dtypes.StringDType: None,  # No equivalent in PyTorch
        np.dtypes.TimeDelta64DType: None,  # No equivalent in PyTorch
        np.dtypes.UByteDType: torch.uint8,
        np.dtypes.UInt16DType: None,  # PyTorch does not support unsigned integers
        np.dtypes.UInt32DType: None,  # PyTorch does not support unsigned integers
        np.dtypes.UInt64DType: None,  # PyTorch does not support unsigned integers
        np.dtypes.UInt8DType: torch.uint8,  # Closest match
        np.dtypes.UIntDType: None,  # No equivalent in PyTorch
        np.dtypes.ULongDType: None,  # No equivalent in PyTorch
        np.dtypes.ULongLongDType: None,  # No equivalent in PyTorch
        np.dtypes.UShortDType: None,  # No equivalent in PyTorch
        np.dtypes.VoidDType: None,  # No equivalent in PyTorch
        **dtype_mapping,
    }
    torch_value = numpy_to_torch_dtype.get(dtype, None)
    if torch_value is None:
        torch_value = numpy_to_torch_dtype.get(type(dtype), None)
    return torch_value

