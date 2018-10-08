import os
import sys
import mxnet as mx
import numpy as np
import unittest
from time import time
import argparse
from collections import OrderedDict

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

from mxnet import profiler

params = OrderedDict()

params['c2'] = {
    'repeat_num': 100,
    'batch': 32,
    'channel': 256,
    'input_shape': (56, 56),
    'output_shape': (56, 56),
}

params['c3'] = {
    'repeat_num': 100,
    'batch': 32,
    'channel': 512,
    'input_shape': (28, 28),
    'output_shape': (28, 28),
}

params['c4'] = {
    'repeat_num': 100,
    'batch': 32,
    'channel': 1024,
    'input_shape': (14, 14),
    'output_shape': (14, 14),
}

params['c5'] = {
    'repeat_num': 100,
    'batch': 32,
    'channel': 2048,
    'input_shape': (7, 7),
    'output_shape': (7, 7),
}

params['d2'] = {
    'repeat_num': 100,
    'batch': 2,
    'channel': 256,
    'input_shape': (256, 256),
    'output_shape': (256, 256),
}

params['d3'] = {
    'repeat_num': 100,
    'batch': 2,
    'channel': 512,
    'input_shape': (128, 128),
    'output_shape': (128, 128),
}

params['d4'] = {
    'repeat_num': 100,
    'batch': 2,
    'channel': 1024,
    'input_shape': (64, 64),
    'output_shape': (64, 64),
}

params['d5'] = {
    'repeat_num': 100,
    'batch': 2,
    'channel': 2048,
    'input_shape': (32, 32),
    'output_shape': (32, 32),
}

@with_seed(1234)
def test_bilinear_sampler_speed(sampler, data_shape, grid_shape, repeat_num, profile):
    
    data = mx.sym.Variable('data')
    grid = mx.sym.Variable('grid')
    sym = sampler(data, grid)
    ctx = mx.gpu(0)
    exe = sym.bind(ctx=ctx, args={
        'data': mx.nd.normal(loc=0, scale=1, shape=data_shape, ctx=ctx),
        'grid': mx.nd.normal(loc=0, scale=1, shape=grid_shape, ctx=ctx)
        },
        args_grad = {
        'data': mx.nd.ones(data_shape, ctx=ctx),
        'grid': mx.nd.ones(grid_shape, ctx=ctx)
        })

    if profile:
        profiler.set_config(profile_all=True,
                            filename='tracing.json',
                            aggregate_stats=True)
        profiler.set_state('run')
    
    t_total = 0.
    for i in range(repeat_num):
        t1 = time()
        out = exe.forward(is_train=True)
        t_total += time() - t1
    print('Forward time:', t_total * 1000 / repeat_num)

    out = out[0].copy()

    t_total = 0.
    for i in range(repeat_num):
        t1 = time()
        exe.backward(out)
        t_total += time() - t1
    print('Backward time:', t_total * 1000 / repeat_num)

    if profile:
        profiler.set_state('stop')
        print(profiler.dumps())
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=1)
    parser.add_argument('-p', default=False, action='store_true')
    args = parser.parse_args()

    sampler_dict = {1: mx.sym.BilinearSampler, 2: mx.sym.BilinearSamplerV2}
    sampler = sampler_dict[args.v]

    for stage, param in params.items():
        if args.v == 1:
            data_shape = (param['batch'], param['channel'], param['output_shape'][0], param['output_shape'][1])
        else:
            data_shape = (param['batch'], param['output_shape'][0], param['output_shape'][1], param['channel'])
        grid_shape = (param['batch'], 2, param['input_shape'][0], param['input_shape'][1])
        repeat_num = param['repeat_num']
        print('Stage:', stage)
        test_bilinear_sampler_speed(sampler, data_shape, grid_shape, repeat_num, args.p)
