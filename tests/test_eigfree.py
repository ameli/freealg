import freealg as fa
import numpy
import sys

# =============
# test eigfree
# =============


def test_eigfree():
    """
    A test for ``eigfree`` function; mostly for speed.
    """
    X = numpy.random.randn(1000, 1000)
    X = (X + X.T) / 2**0.5

    # Compute eigfree 100 times to test speed
    for _ in range(100):
        fa.eigfree(X)


# ===========
# script main
# ===========


if __name__ == "__main__":
    sys.exit(test_eigfree())
