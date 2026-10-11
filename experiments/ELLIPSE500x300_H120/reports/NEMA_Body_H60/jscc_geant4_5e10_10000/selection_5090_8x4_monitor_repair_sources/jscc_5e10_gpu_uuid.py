"""Strict CUDA/NVIDIA UUID string identity, independent of GPU ordinal mapping."""
from uuid import UUID


def canonical_gpu_uuid(value):
    text = str(value).strip()
    if text.startswith('GPU-'):
        text = text[4:]
    # Accept only the documented full 8-4-4-4-12 CUDA UUID, never a short prefix.
    identifier = UUID(text)
    if text.lower() != str(identifier) or identifier.int == 0:
        raise ValueError('Full nonzero CUDA GPU UUID required')
    return 'GPU-' + str(identifier)
