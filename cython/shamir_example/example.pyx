# Declare the C function
cdef extern from "example.h":
    void hello(const char *name)
    double cos_func(double angle)

# Define a Python version
# This uses the Python type annotations
# See here: https://stavshamir.github.io/python/the-other-benefit-of-python-type-annotations/
def py_hello(name: bytes) -> None:
    hello(name)

def py_cos(angle: float) -> float:
    return cos_func(angle)
