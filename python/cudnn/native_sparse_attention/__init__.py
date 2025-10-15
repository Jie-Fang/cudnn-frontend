from .selection import SelectionAttention, selection_attention_wrapper
from .compression import CompressionAttention, compression_attention_wrapper
from .sliding_window_attention import (
    SlidingWindowAttention,
    sliding_window_attention_wrapper,
)


class NSANamespace:
    SelectionAttention = staticmethod(SelectionAttention)
    selection_attention_wrapper = staticmethod(selection_attention_wrapper)

    SlidingWindowAttention = staticmethod(SlidingWindowAttention)
    sliding_window_attention_wrapper = staticmethod(sliding_window_attention_wrapper)

    CompressionAttention = staticmethod(CompressionAttention)
    compression_attention_wrapper = staticmethod(compression_attention_wrapper)


NSA = NSANamespace()

__all__ = [
    "NSA",
]
