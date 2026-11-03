"""Thread-local execution settings for the existing SAM image transform."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path


def validate_cpu_prepare_threads(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 8:
        raise ValueError("SAM CPU preparation threads must be an integer from 1 to 8")
    return value


@lru_cache(maxsize=1)
def _openmp_functions():
    # Resolve this Torch's runtime; never call process-wide set_num_threads.
    import ctypes
    import torch

    if not torch.backends.openmp.is_available():
        raise RuntimeError("SAM CPU preparation thread scope requires Torch OpenMP")
    path = Path(torch.__file__).parent / "lib" / "libtorch_cpu.so"
    try:
        library = ctypes.CDLL(str(path))
        get_threads = library.omp_get_max_threads
        set_threads = library.omp_set_num_threads
    except (OSError, AttributeError) as error:
        raise RuntimeError("Torch OpenMP thread-local controls are unavailable") from error
    get_threads.argtypes = []
    get_threads.restype = ctypes.c_int
    set_threads.argtypes = [ctypes.c_int]
    set_threads.restype = None
    return library, get_threads, set_threads


@contextmanager
def cpu_preparation_threads(threads: int):
    threads = validate_cpu_prepare_threads(threads)
    if threads == 1:
        # Default execution preserves the caller's existing configuration.
        yield
        return
    _library, get_threads, set_threads = _openmp_functions()
    previous = get_threads()
    try:
        set_threads(threads)
        if get_threads() != threads:
            raise RuntimeError("Torch did not apply the CPU preparation thread scope")
        yield
    finally:
        set_threads(previous)
