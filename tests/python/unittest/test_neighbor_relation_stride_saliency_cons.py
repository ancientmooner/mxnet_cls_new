import os
import sys
#sys.path.insert(0, '../external/mxnet/mxnet_baefc7')
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
sys.path.insert(0, os.path.join(curr_path, '../../../python'))


#from core import module
import mxnet as mx
from mxnet import module
from collections import namedtuple
from math import ceil, floor
import numpy as np
import time
from mxnet.test_utils import *
os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
#os.environ['MXNET_ENGINE_TYPE'] = 'ThreadedEnginePerDevice'
#os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '2'

def get_offset_grids(batch_size, height, width, kernel_size=7, dilate=1):
    k_arange = (mx.sym.arange(0, kernel_size) - int(kernel_size / 2)) * dilate
    offsets_x = mx.sym.tile(mx.sym.reshape(k_arange, (1, -1, 1, 1, 1)), reps=(kernel_size, 1, 1, height, width))
    offsets_y = mx.sym.tile(mx.sym.reshape(k_arange, (-1, 1, 1, 1, 1)), reps=(1, kernel_size, 1, height, width))

    # [49, 2, height, width]
    offsets = mx.sym.reshape(mx.sym.concat(offsets_x, offsets_y, dim=2), shape=(kernel_size * kernel_size, 2, height, width))
    # [49, 2, height, width]
    offset_grids = mx.sym.GridGenerator(offsets, transform_type='warp')
    #[batch_size, 2, height * 49, width]
    offset_grids_repeat = mx.sym.tile(mx.sym.reshape(mx.sym.transpose(offset_grids, axes=(1, 0, 2, 3)), shape=(0, -3, -2)), reps=(batch_size, 1, 1, 1))
    offset_grids_repeat = mx.sym.BlockGrad(offset_grids_repeat)
    return offset_grids_repeat 

def neighbor_relation_v1(value_data, key_data, query_data, pos_weight, scale, batch_size, cur_height, cur_width, num_group=32, kernel_size=7, dilate=1, norm_method=0, sim_method=0, no_define_value=-100.0, stride=1, key_saliency_group=1, mask=None):
    
    offset_grids_repeat = get_offset_grids(batch_size, cur_height, cur_width, kernel_size=kernel_size, dilate=dilate)
    # [batch_size, key_channels, height * 49, width]
    warp_key_data = mx.sym.BilinearSampler(data=key_data, grid=offset_grids_repeat)
    # [batch_size, key_channels, 49, height, width]
    warp_key_data_reshape = mx.sym.reshape(mx.sym.reshape(warp_key_data, shape=(0, 0, -4, kernel_size ** 2, -1, -2)), shape=(0, -3, -2))
    warp_key_data_reshape = mx.sym.Pooling(data=warp_key_data_reshape, kernel=(1, 1), stride=(stride, stride), pool_type='max', pooling_convention='valid')
    warp_key_data_reshape = mx.sym.reshape(data=warp_key_data_reshape, shape=(0, -4, -1, kernel_size ** 2, -2))

    if key_saliency_group == 0:
        end_idx=None
    else:
        end_idx=-key_saliency_group
    warp_key_data_reshape_t = mx.sym.slice_axis(data=warp_key_data_reshape, axis=1, begin=0, end=end_idx)
    warp_key_saliency = mx.sym.slice_axis(data=warp_key_data_reshape, axis=1, begin=end_idx, end=None)
    # [batch_size, value_channels, height * 49, width]
    warp_value_data = mx.sym.BilinearSampler(data=value_data, grid=offset_grids_repeat)
    # [batch_size, num_group, value_channels/num_group, 49, height, width]
    warp_value_data_reshape = mx.sym.reshape(mx.sym.reshape(warp_value_data, shape=(0, 0, -4, kernel_size ** 2, -1, -2)), shape=(0, -3, -2))
    warp_value_data_reshape = mx.sym.Pooling(data=warp_value_data_reshape, kernel=(1, 1), stride=(stride, stride), pool_type='max', pooling_convention='valid')
    warp_value_data_reshape = mx.sym.reshape(mx.sym.reshape(data=warp_value_data_reshape, shape=(0, -4, num_group, -1, -2)), shape=(0, 0, -4, -1, kernel_size ** 2, -2))

    # [batch_size, key_channels, 1, height, width]
    query_data_reshape = mx.sym.expand_dims(query_data, axis=2)

    # [batch_size, key_channels, 49, height, width]
    if sim_method == 0:
        app_dot = mx.sym.broadcast_mul(query_data_reshape, warp_key_data_reshape_t)
    else:
        app_dot = mx.sym.broadcast_add(query_data_reshape, warp_key_data_reshape_t)
    # [batch_size, num_group, key_channels/num_group, 49, height, width]
    app_dot_reshape = mx.sym.reshape(app_dot, shape=(0, -4, num_group, -1, -2))

    # [batch_size, num_group, 49, height, width]
    app_sim = mx.sym.sum(app_dot_reshape, axis=2) * scale
    
    app_sim = mx.sym.broadcast_add(app_sim, warp_key_saliency)

    # [batch_size, num_group, 49, height, width]
    app_geo_sim = mx.sym.broadcast_add(app_sim, mx.sym.reshape(pos_weight, (1, num_group, -1, 1, 1)))

    # [batch_size, num_group, 49, height, width]
    if norm_method == 0:
        app_geo_sim = mx.sym.softmax(app_geo_sim, axis=2)
    else:
        app_geo_sim = mx.sym.Activation(app_geo_sim, act_type='relu')
        sum_app_geo_sim = mx.sym.sum(app_geo_sim, axis=2, keepdims=True) + 1e-6
        app_geo_sim = mx.sym.broadcast_div(app_geo_sim, sum_app_geo_sim)

    if mask is not None:
        mask = mx.sym.reshape(mask, shape=(batch_size, num_group, kernel_size ** 2, cur_height, cur_width))
        app_geo_sim = app_geo_sim * mask

    # [batch_size, num_group, 1, 49, height, width]
    app_geo_sim = mx.sym.expand_dims(app_geo_sim, axis=2)

    output_value = mx.sym.reshape(mx.sym.sum(mx.sym.broadcast_mul(app_geo_sim, warp_value_data_reshape), axis=3), shape=(0, -3, -2))
    return mx.sym.MakeLoss(output_value)

def neighbor_relation_v2(value_data, key_data, query_data, pos_weight, scale, batch_size, cur_height, cur_width, num_group=32, kernel_size=7, dilate=1, norm_method=0, sim_method=0, no_define_value=-100.0, stride=1, key_saliency_group=1):
    
    output_value = mx.sym.contrib.NeighborRelation(key=key_data, value=value_data, query=query_data, 
                    pos_weight=pos_weight, scale=scale, num_group=num_group, kernel_size=kernel_size, dilate=dilate, batch_step=8, norm_method=norm_method, sim_method=sim_method, no_define_value=no_define_value, dropout_ratio=0.0, stride=stride, key_saliency_group=key_saliency_group)
    #output_value = mx.sym.contrib.NeighborRelation(key=key_data, value=value_data, query=query_data, 
    #                pos_weight=pos_weight, scale=scale, num_group=num_group, kernel_size=kernel_size, dilate=dilate, batch_step=32, norm_method=norm_method, sim_method=sim_method, no_define_value=no_define_value, dropout_ratio=0.1)
    #return mx.sym.Group([mx.sym.MakeLoss(output_value), mx.sym.BlockGrad(mask)])
    return mx.sym.MakeLoss(output_value)

def neighbor_relation_v3(value_data, key_data, query_data, pos_weight, scale, batch_size, cur_height, cur_width, num_group=32, kernel_size=7, dilate=1, norm_method=0, sim_method=0, no_define_value=-100.0):
    
    output_value = mx.sym.contrib.NeighborRelationEfficient(key=key_data, value=value_data, query=query_data, 
                    pos_weight=pos_weight, scale=scale, num_group=num_group, kernel_size=kernel_size, dilate=dilate, batch_step=32, norm_method=norm_method, sim_method=sim_method, no_define_value=no_define_value)
    return mx.sym.MakeLoss(output_value)

def consistent_check(key_shape, query_shape, value_shape, kernel_size=7, num_group=32, test_bp=False, stride=1, key_saliency_group=1):
    # param
    key_map = np.random.rand(*key_shape)
    query_map = np.random.rand(*query_shape)
    value_map = np.random.rand(*value_shape)

    print 'key shape:', key_shape
    print 'query shape:', query_shape
    print 'value shape:', value_shape
    pos_weight_shape = (num_group, kernel_size, kernel_size)
    pos_weight_map = np.random.rand(*pos_weight_shape)

    key_data_blob = mx.nd.array(key_map, mx.gpu(0))
    query_data_blob = mx.nd.array(query_map, mx.gpu(0))
    value_data_blob = mx.nd.array(value_map, mx.gpu(0))
    pos_weight_data_blob = mx.nd.array(pos_weight_map, mx.gpu(0))
    key_data = mx.sym.Variable(name="key_data")
    query_data = mx.sym.Variable(name="query_data")
    value_data = mx.sym.Variable(name="value_data")
    pos_weight_data = mx.sym.Variable(name="pos_weight_data")
    
    t_key_data = mx.sym.Variable(name="t_key_data")
    t_query_data = mx.sym.Variable(name="t_query_data")
    t_value_data = mx.sym.Variable(name="t_value_data")
    t_pos_weight_data = mx.sym.Variable(name="t_pos_weight_data")
    
    t_mask_data = mx.sym.Variable(name="t_mask_data")
    #t_mask_data = None

    batch_size = key_shape[0]
    key_channels = key_shape[1]
    height = key_shape[2]
    width = key_shape[3]

    value_channels = value_shape[1]

    scale = 1
    neighbor_relation_out_v2 = neighbor_relation_v2(value_data, key_data, query_data, pos_weight_data, scale, batch_size, height, width, num_group=num_group, kernel_size=kernel_size, dilate=1, norm_method=0, sim_method=0, no_define_value=0.0, stride=stride, key_saliency_group=key_saliency_group)
    
    neighbor_relation_out_v1 = neighbor_relation_v1(t_value_data, t_key_data, t_query_data, t_pos_weight_data, scale, batch_size, height, width, num_group=num_group, kernel_size=kernel_size, dilate=1, norm_method=0, sim_method=0, no_define_value=0.0, stride=stride, key_saliency_group=key_saliency_group)

    neighbor_relation_methods = [neighbor_relation_out_v2, neighbor_relation_out_v1]
    neighbor_relation_method = neighbor_relation_out_v2

    arg_names = neighbor_relation_method.list_arguments()
    data_shape_dict = {'value_data': value_shape, 'key_data': key_shape, 'query_data': query_shape, 'pos_weight_data': pos_weight_shape}

    arg_shapes, out_shapes, _ = neighbor_relation_method.infer_shape(**data_shape_dict)
    arg_shape_dict = dict(zip(arg_names, arg_shapes))
    print arg_shape_dict
    print arg_names
    print arg_shapes
    print out_shapes
    args = {}
    args['value_data'] = value_data_blob
    args['key_data'] = key_data_blob
    args['query_data'] = query_data_blob
    args['pos_weight_data'] = pos_weight_data_blob
    
    t_args = {}
    t_args['t_value_data'] = value_data_blob.copy()
    t_args['t_key_data'] = key_data_blob.copy()
    t_args['t_query_data'] = query_data_blob.copy()
    t_args['t_pos_weight_data'] = pos_weight_data_blob.copy()

    args_grad = {}
    t_args_grad = {}
    for i_arg, arg in enumerate(arg_shapes):
        args_grad[arg_names[i_arg]] = mx.nd.empty(arg, ctx=mx.gpu(0))
        t_args_grad['t_' + arg_names[i_arg]] = mx.nd.empty(arg, ctx=mx.gpu(0))

    #args_grad = [mx.nd.empty(s, ctx=mx.gpu(0)) for s in arg_shapes]
    #t_args_grad = [mx.nd.empty(s, ctx=mx.gpu(0)) for s in arg_shapes]

    programs = []

    args_all = [args, t_args]
    args_grad_all = [args_grad, t_args_grad]
    #dcn_methods = [dcn_methods[1]]
    for idx, neighbor_relation_method in enumerate(neighbor_relation_methods):
        #print args_grad_all[idx]
        prog = neighbor_relation_method.bind(mx.gpu(0), args=args_all[idx], args_grad=args_grad_all[idx])
        prog.forward(is_train=test_bp)

        if idx == 0:
            #print prog.outputs[1].asnumpy()
            #mask_blob = prog.outputs[1]
            #args_all[1]["t_mask_data"] = mask_blob.copy()
            #args_grad_all[1]["t_mask_data"] = mx.nd.empty(mask_blob.shape, ctx=mx.gpu(0))
            pass
        #attrs = vars(prog)
        #print ', '.join("%s: %s" % item for item in attrs.items())
        if test_bp:
            prog.backward()

        programs.append(prog)
   
    print programs[1]
    #for i_arg in range(len(programs[0].outputs)):
    for i_arg in range(1):
        err = abs(programs[1].outputs[i_arg].asnumpy() - programs[0].outputs[i_arg].asnumpy())
        err_ratio = len(np.where(err > 0.1 * abs(programs[0].outputs[i_arg].asnumpy()))[0]) /  float(programs[0].outputs[i_arg].asnumpy().size)
        sum_err = np.sum(abs(programs[1].outputs[i_arg].asnumpy())) - np.sum(abs(programs[0].outputs[i_arg].asnumpy()))
        
        err = np.sum(err)
        print "neighbor_relation out 1", programs[0].outputs[i_arg].asnumpy()
        print "neighbor_relation out 2", programs[1].outputs[i_arg].asnumpy()
        print 'output id ', i_arg, ', shape=', programs[0].outputs[i_arg].asnumpy().shape, ", error=", err, ", error ratio=", err_ratio

    print "tt", programs[0].grad_arrays
    print "tt2", programs[1].grad_arrays
    if test_bp:
        #print type(programs[0].grad_arrays)
        #print vars(programs[0])
        map_ids = [3, 1, 0, 2]
        #map_ids = [0, 1, 2, 3]
        for i_arg in range(len(programs[0].grad_arrays)):
            print "neighbor_relation_grad", programs[0].grad_arrays[i_arg].asnumpy()
            print "neighbor_relation_grad", programs[1].grad_arrays[map_ids[i_arg]].asnumpy()
            err = abs(programs[1].grad_arrays[map_ids[i_arg]].asnumpy() - programs[0].grad_arrays[i_arg].asnumpy())
            err_ratio = len(np.where(err > 0.1 * abs(programs[0].grad_arrays[i_arg].asnumpy()))[0]) /  float(programs[0].grad_arrays[i_arg].asnumpy().size)
            err = np.sum(err)
            print 'grad id ', i_arg, ', shape=', programs[0].grad_arrays[i_arg].asnumpy().shape, ", grad error=", err, ", grad error ratio=", err_ratio
    
    #np.set_printoptions(threshold='nan')

    

def speed_check(input_shape, num_filter, stride, dilate, layout="NHWC", test_bp=False, method=0):
    # param
    feat_map = np.random.rand(*input_shape)
    data_blob = mx.nd.array(feat_map, mx.gpu(0))
    data = mx.sym.Variable(name="data")

    if layout == "NHWC":
        input_channel = input_shape[3]
    else:
        input_channel = input_shape[1]

    prev_data1 = data
    prev_data2 = data
    prev_data3 = data
    for i in range(13):
        prefix = str(i) 

        weight_var = mx.sym.Variable(name=prefix + "dcn_weight", shape=(num_filter, 3, 3, input_channel))
        offset_weight_var = mx.sym.Variable(name=prefix + "dcn_offset_weight", shape=(18, 3, 3, input_channel))
        offset_bias_var = mx.sym.Variable(name=prefix + "dcn_offset_bias", shape=(18, ))
        dcn_v1_out = dconv_v1("dcn_v1_"+prefix, prev_data1, num_filter, stride, dilate, 
                    weight_var, offset_weight_var, offset_bias_var, layout=layout)
        prev_data1 = dcn_v1_out[0]
        dcn_v2_out = dconv_v2("dcn_v2_"+prefix, prev_data2, num_filter, stride, dilate, 
                    weight_var, offset_weight_var, offset_bias_var, layout=layout)
        prev_data2 = dcn_v2_out[0]
        dcn_v3_out = dconv_v3("dcn_v3_"+prefix, prev_data3, num_filter, stride, dilate, 
                    weight_var, offset_weight_var, offset_bias_var, layout=layout)
        prev_data3 = dcn_v2_out[0]

    dcn_v1_out = mx.sym.MakeLoss(dcn_v1_out[0])
    dcn_v2_out = mx.sym.MakeLoss(dcn_v2_out[0])
    dcn_v3_out = mx.sym.MakeLoss(dcn_v3_out[0])

    if method == 0:
        dcn_method = dcn_v1_out
    elif method == 1:
        dcn_method = dcn_v2_out
    else:
        dcn_method = dcn_v3_out

    arg_names = dcn_method.list_arguments()
    arg_shapes, out_shapes, _ = dcn_method.infer_shape(data=input_shape)
    arg_shape_dict = dict(zip(arg_names, arg_shapes))
    #print arg_shape_dict
    #print arg_names
    #print arg_shapes
    #print out_shapes
    args = {}
    args['data'] = data_blob

    for key in arg_shape_dict.keys():
        if not key == 'data':
            args[key] = mx.random.normal(0, 1, arg_shape_dict[key], ctx=mx.cpu()).copyto(mx.gpu(0))

    args_grad = [mx.nd.empty(s, ctx=mx.gpu(0)) for s in arg_shapes]

    prog = dcn_method.bind(mx.gpu(0), args=args, args_grad=args_grad)

    t_fw_total = 0
    t_bw_total = 0
    repeat_num = 20
    for i in range(repeat_num):
        t1 = time.time()
        prog.forward(is_train=True)
        #out = mx.ndarray.reshape(prog.outputs[1], shape=(0, 0, 0, 0))
        out = mx.ndarray.sum(prog.outputs[0]).asnumpy()
        t_fw_total = t_fw_total + time.time() - t1
        if test_bp:
            t1 = time.time()
            prog.backward()
            out = mx.ndarray.sum(prog.grad_arrays[0]).asnumpy()
            t_bw_total += time.time() - t1
            #print prog.grad_arrays[0].shape
    
    print('Forward time:', t_fw_total * 1000 / repeat_num)
    print('Backward time:', t_bw_total * 1000 / repeat_num)

if __name__ == '__main__':
        
    num_times = 1
    stride = 1
    dilate = 1
    method_id = 1
    layout="NHWC"
    test_bp = True
  
    key_dim=1
    #num_group = 64
    values = 4
    batch_size = 32
    #key_shape=(batch_size, num_group * key_dim, 56, 56)
    #value_shape=(batch_size, num_group * values, 56, 56)
    kernel_size = 7
    #consistent_check(key_shape, value_shape, kernel_size=kernel_size, num_group=num_group, test_bp=test_bp)
   
    #num_group = 128
    #key_shape=(batch_size, num_group * key_dim, 28, 28)
    #value_shape=(batch_size, num_group * values, 28, 28)
    #consistent_check(key_shape, value_shape, kernel_size=kernel_size, num_group=num_group, test_bp=test_bp)
    
    #num_group = 256
    #key_shape=(batch_size, num_group * key_dim, 14, 14)
    #value_shape=(batch_size, num_group * values, 14, 14)
    #consistent_check(key_shape, value_shape, kernel_size=kernel_size, num_group=num_group, test_bp=test_bp)
   
    stride = 2
    in_size = 56
    num_group = 32
    key_saliency_group = 1
    out_size = int((in_size - 1) / stride) + 1
    key_shape=(batch_size, num_group * key_dim + key_saliency_group, in_size, in_size)
    query_shape=(batch_size, num_group * key_dim, out_size, out_size)
    value_shape=(batch_size, num_group * values, in_size, in_size)
    consistent_check(key_shape, query_shape, value_shape, kernel_size=kernel_size, num_group=num_group, test_bp=test_bp, stride=stride, key_saliency_group=key_saliency_group)
