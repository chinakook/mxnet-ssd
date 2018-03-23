
# coding: utf-8

# In[1]:


import sys
import numpy as np
import mxnet as mx
import json
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])
from PIL import Image
from symbol.mobilenet_deploy import get_symbol


def merge_conv_bn(old_checkpoint, old_epoch, new_checkpoint):
    sym, args, auxs = mx.model.load_checkpoint(old_checkpoint, int(old_epoch))
    _, sym_outputs, _ = sym.infer_shape(data=(1,3,224,224))

    sym_deploy = get_symbol(sym_outputs[0][1])

    args_deploy = {}
    auxs_deploy = {}

    graph=json.loads(sym.tojson())

    merge_dict = {}
    for i, n in enumerate(graph['nodes']):
        if n['op'] == 'BatchNorm':
            pre_layer =  graph['nodes'][n['inputs'][0][0]]
            if pre_layer['op'] == 'Convolution':
                merge_dict[pre_layer['name']] = i#n['name']

    for i, n in enumerate(graph['nodes']):
        if n['op'] == 'Convolution':
            if merge_dict.has_key(n['name']):
                bn = graph['nodes'][merge_dict[n['name']]]
                gamma = args[bn['name']+'_gamma']
                beta = args[bn['name']+'_beta']
                moving_mean = auxs[bn['name']+'_moving_mean']
                moving_var = auxs[bn['name']+'_moving_var']
                eps = float(bn['attr']['eps'])

                weight = args[n['name']+'_weight']
                if not n['attr'].has_key('no_bias') or n['attr']['no_bias'] == 'False':
                    bias = args[n['name']+'_bias']
                else:
                    bias = mx.nd.zeros((weight.shape[0],))              
                a = gamma / mx.nd.sqrt(moving_var + eps)
                b = beta - a * moving_mean
                a = mx.nd.reshape(a,(-1,1,1,1))
                weight = weight * a
                bias = bias + b
                args_deploy[n['name'] + '_weight'] = weight
                args_deploy[n['name'] + '_bias'] = bias
            else:
                args_deploy[n['name'] + '_weight'] = args[n['name']+'_weight']
                if not n['attr'].has_key('no_bias') or n['attr']['no_bias'] == 'False':
                    args_deploy[n['name'] + '_bias'] = args[n['name']+'_bias']
        elif n['op'] == 'FullyConnected':
            args_deploy[n['name']+'_weight'] = args[n['name']+'_weight']
            if not n['attr'].has_key('no_bias') or n['attr']['no_bias'] == 'False':
                args_deploy[n['name']+'_bias'] = args[n['name']+'_bias']

    model_deploy = mx.mod.Module(symbol=sym_deploy, data_names=['data'], label_names=None)
    model_deploy.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
    model_deploy.set_params(arg_params=args_deploy, aux_params=auxs_deploy, allow_missing=True)
    model_deploy.save_checkpoint(new_checkpoint, 0)


if __name__ == "__main__":
    merge_conv_bn(sys.argv[1], sys.argv[2], sys.argv[3])


# In[ ]:


# for verification use
#set(args_deploy.keys()) - set(sym_deploy.list_arguments())
# ddd = {}
# for k,v in args_deploy.items():
#     ddd[k] = v.shape
# args_deploy_shapes,_, auxs_deploy_shapes = sym_deploy.infer_shape(data=(1,3,224,224))

# dict(zip(sym_deploy.list_arguments(), args_deploy_shapes))


# In[ ]:





# In[ ]:


# for test use
# model_deploy = mx.mod.Module(symbol=sym_deploy, data_names=['data'], label_names=None)
# model_deploy.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
# model_deploy.set_params(arg_params=args_deploy, aux_params=auxs_deploy, allow_missing=True)
# model_deploy.save_checkpoint('mbn_deploy', 0)

# model = mx.mod.Module(symbol=sym, data_names=['data'], label_names=None)
# model.bind(data_shapes=[('data', (1, 3, 224, 224))], for_training=False)
# model.set_params(arg_params=args, aux_params=auxs, allow_missing=True)

# im_raw = Image.open('cat.jpg')
# im_raw = im_raw.resize((224,224))
# im = np.array(im_raw, dtype=np.float32)
# im = im.transpose((2,0,1))
# im = im[np.newaxis,:,:]

# model.forward(Batch([mx.nd.array(im)]))
# model.get_outputs()[0]

# model_deploy.forward(Batch([mx.nd.array(im)]))
# model_deploy.get_outputs()[0]


# In[ ]:





# In[ ]:




