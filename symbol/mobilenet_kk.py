import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix='', cudnn_off=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s' % name, cudnn_off=cudnn_off)
    bn = mx.sym.BatchNorm(data=conv, name='%s_bn' % name, fix_gamma=False, use_global_stats=True, eps=0.0001, attr={'lr_mult': '0.1'})
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s_relu' % name )
    return act

def get_symbol(num_classes=1000, **kwargs):
    data = mx.symbol.Variable(name='data')
    label = mx.symbol.Variable(name="label")
    conv1 = Conv(data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1") # 224/112
    
    conv2_1_dw = Conv(conv1, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv2_1_dw", cudnn_off=True) # 112/112
    conv2_1_sep = Conv(conv2_1_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv2_1_sep") # 112/112
    conv2_2_dw = Conv(conv2_1_sep, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv2_2_dw", cudnn_off=True) # 112/56
    conv2_2_sep = Conv(conv2_2_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv2_2_sep") # 56/56
    
    conv3_1_dw = Conv(conv2_2_sep, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv3_1_dw", cudnn_off=True) # 56/56
    conv3_1_sep = Conv(conv3_1_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv3_1_sep") # 56/56
    conv3_2_dw = Conv(conv3_1_sep, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv3_2_dw", cudnn_off=True) # 56/28
    conv3_2_sep = Conv(conv3_2_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv3_2_sep") # 28/28
    
    conv4_1_dw = Conv(conv3_2_sep, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv4_1_dw", cudnn_off=True) # 28/28
    conv4_1_sep = Conv(conv4_1_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv4_1_sep") # 28/28
    conv4_2_dw = Conv(conv4_1_sep, num_group=256, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv4_2_dw", cudnn_off=True) # 28/14
    conv4_2_sep = Conv(conv4_2_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv4_2_sep") # 14/14

    conv5_1_dw = Conv(conv4_2_sep, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv5_1_dw", cudnn_off=True) # 14/14
    conv5_1_sep = Conv(conv5_1_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_1_sep") # 14/14
    conv5_2_dw = Conv(conv5_1_sep, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv5_2_dw", cudnn_off=True) # 14/14
    conv5_2_sep = Conv(conv5_2_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_2_sep") # 14/14
    conv5_3_dw = Conv(conv5_2_sep, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv5_3_dw", cudnn_off=True) # 14/14
    conv5_3_sep = Conv(conv5_3_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_3_sep") # 14/14
    conv5_4_dw = Conv(conv5_3_sep, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv5_4_dw", cudnn_off=True) # 14/14
    conv5_4_sep = Conv(conv5_4_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_4_sep") # 14/14
    conv5_5_dw = Conv(conv5_4_sep, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv5_5_dw", cudnn_off=True) # 14/14
    conv5_5_sep = Conv(conv5_5_dw, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_5_sep") # 14/14
    conv5_6_dw = Conv(conv5_5_sep, num_group=512, num_filter=512, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv5_6_dw", cudnn_off=True) # 14/7
    conv5_6_sep = Conv(conv5_6_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_6_sep") # 7/7
    
    conv6_dw = Conv(conv5_6_sep, num_group=1024, num_filter=1024, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv6_dw", cudnn_off=True) # 7/7
    conv6_sep = Conv(conv6_dw, num_filter=1024, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv6_sep") # 7/7
    
    pool6 = mx.symbol.Pooling(name='pool6', data=conv6_sep , pooling_convention='full', global_pool=True, kernel=(1,1), pool_type='avg')
    
    fc7 = mx.symbol.Convolution(name='fc7', data=pool6 , num_filter=num_classes, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=False)
    flatten = mx.symbol.Flatten(data=fc7, name='flatten')
    softmax = mx.symbol.SoftmaxOutput(data=flatten, name='softmax')
    return softmax