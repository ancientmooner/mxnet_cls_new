from __future__ import print_function
import sys
import os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
sys.path.insert(0, os.path.join(curr_path, '../../../python/'))

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

#from common import setup_module, with_seed, teardown

data_shape = (1, 512, 32, 32)
kernel_shape = (3, 3)
#grid_shape = (1, 2, 4, 4)
xpus = [mx.gpu(0), mx.gpu(1)]
dtypes = [np.float32]
#dtypes = [np.float32]
sampler = mx.sym.contrib.DownsampleV2

#@with_seed(1234)
def test_downsample_with_type():
    data = mx.sym.Variable('data')
    kernel = mx.sym.Variable('kernel')
    sym = sampler(data=data, kernel=kernel, backward_kernel=True, rescale=2, dilate=1)
    ctx_list = [{'ctx': xpu, 'data': data_shape, 'kernel': kernel_shape,
                 'type_dict': {'data': dtype}} for xpu in xpus for dtype in dtypes]
    check_consistency(sym, ctx_list)
    check_consistency(sym, ctx_list, grad_req="add")

if __name__ == '__main__':
    test_downsample_with_type()
	#import nose
	#nose.runmodule()
