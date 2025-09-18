from .selection import SelectionAttention, SelectionAttentionWrapper


class NSANamespace:
    SelectionAttention = staticmethod(SelectionAttention)
    SelectionAttentionWrapper = staticmethod(SelectionAttentionWrapper)


NSA = NSANamespace()

__all__ = [
    "NSA",
]
