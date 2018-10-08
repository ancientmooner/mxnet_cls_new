from __future__ import print_function
import sys
import os
import time
import multiprocessing as mp
import unittest
import mxnet as mx
import numpy as np
import unittest
from nose.tools import assert_raises
from mxnet.test_utils import check_consistency, set_default_context, assert_almost_equal
from mxnet.base import MXNetError
from mxnet import autograd
from numpy.testing import assert_allclose

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

data_shape = (1, 16, 16, 512)
grid_shape = (1, 2, 4, 4)
xpus = [mx.gpu(0), mx.cpu(0)]
dtypes = [np.float64, np.float32]
sampler = mx.sym.BilinearSamplerV3

@with_seed(1234)
def test_bilinear_sampler_with_type():
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym = sampler(data=data, grid=grid)
    ctx_list = [{'ctx': xpu, 'data': data_shape, 'grid': grid_shape,
                 'type_dict': {'data': dtype}} for xpu in xpus for dtype in dtypes]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")

if __name__ == '__main__':
	import nose
	nose.runmodule()