"""What a conformance recorder and its test both compute without storing it: a network's start
state, drawn from each variable's name, and the positions at which a large leaf's update is kept.

Recorders for networks of millions of parameters (ppo2's Nature CNN) start the recorded update from
this state instead of their own initialisation, and keep each large leaf at SAMPLED positions, so
that their committed records stay small. numpy's legacy RandomState stream is fixed across numpy
versions (NEP 19), so a recorder on an old numpy and the test on a new one draw the same values.
Runs on Python 3.6 (the TensorFlow 1.15 recorders); imports nothing from Ajax.
"""

import zlib

import numpy as np

#: A leaf of more entries than this is kept at this many positions.
SAMPLED = 4096


def _stream(name):
    return np.random.RandomState(zlib.crc32(name.encode()))


def start_value(name, shape):
    """A float32 start for the variable ``name`` of ``shape``, normal with standard deviation
    1/sqrt(fan in), the product of all axes but the last, as an orthogonal or LeCun initialisation
    scales a kernel; 0.1 where that product is 1 (a bias, a log-std), so that these start away
    from zero."""
    shape = tuple(int(n) for n in shape)
    fan_in = int(np.prod(shape[:-1]))
    width = 1.0 / np.sqrt(fan_in) if fan_in > 1 else 0.1
    return (_stream(name).standard_normal(shape) * width).astype(np.float32)


def sampled_positions(name, size):
    """The flat positions, sorted, at which a leaf of ``size`` entries is kept: all of them up to
    SAMPLED, else SAMPLED of them drawn without replacement."""
    if size <= SAMPLED:
        return np.arange(size)
    return np.sort(_stream(name + "/sampled").choice(size, SAMPLED, replace=False))
