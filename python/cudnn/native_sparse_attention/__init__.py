from .selection import SelectionAttention, selection_attention_wrapper


class NSANamespace:
    SelectionAttention = staticmethod(SelectionAttention)
    selection_attention_wrapper = staticmethod(selection_attention_wrapper)


NSA = NSANamespace()

__all__ = [
    "NSA",
]
