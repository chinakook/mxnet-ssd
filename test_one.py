import os
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import numpy as np
import math
import cv2
from timeit import default_timer as timer
import mxnet as mx

from symbol.symbol_factory import get_symbol

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

data_shape = 300
#mean_img = np.array([123,117,104], dtype=np.float32)
#mean_img = mean_img[:,np.newaxis,np.newaxis]
cls_mean_val = np.array([[[123.68]],[[116.78]],[[103.94]]])
cls_names = ['red', 'green', 'yellow']
cls_std_scale = 0.017

ctx = mx.cpu()

def get_detection_mod():

    #net = get_symbol('vgg16_reduced', data_shape, num_classes=1, nms_thresh=0.5,force_nms=True, nms_topk=400)
    #sym, args, auxs = mx.model.load_checkpoint('./model/ssd_vgg16_reduced_300', 126)
    net = get_symbol('mobilenet', data_shape, num_classes=1, nms_thresh=0.5,force_nms=True, nms_topk=400)
    sym, args, auxs = mx.model.load_checkpoint('./model/ssd_mobilenet_300', 100)

    mod = mx.mod.Module(net, label_names=None, context=ctx)

    mod.bind(data_shapes=[('data', (1, 3, data_shape, data_shape))])
    mod.set_params(args, auxs, allow_extra=True)

    return mod

def get_classification_mod():
    sym, arg_params, aux_params = mx.model.load_checkpoint('./model/ylb', 199)
    all_layers = sym.get_internals()
    ss = all_layers['softmax_output']
    mod = mx.mod.Module(symbol=ss, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))], label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
    return mod


def det_img(raw_img, mod):
    #raw_img = cv2.imread(testdir + '/' + fn)
    start = timer()
    

    h = raw_img.shape[0]
    w = raw_img.shape[1]
    
    if w > h:
        short_dim = int(h * data_shape / float(w))
        raw_img2 = cv2.resize(raw_img, (data_shape, short_dim))
        pad = data_shape - short_dim
        raw_img2 = cv2.copyMakeBorder(raw_img2, pad//2, (pad+1)//2, 0, 0, cv2.BORDER_CONSTANT)
        scale = data_shape / float(w)
    else:
        short_dim = int(w * data_shape / float(h))
        raw_img2 = cv2.resize(raw_img, (short_dim, data_shape))
        pad = data_shape - short_dim
        raw_img2 = cv2.copyMakeBorder(raw_img2, 0, 0, pad//2, (pad+1)//2, cv2.BORDER_CONSTANT)
        scale = data_shape / float(h)
    
    raw_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2RGB)
    #cv2.imshow("small", raw_img2)
    #cv2.waitKey()
    
    #raw_img2 = cv2.resize(raw_img2, (data_shape,data_shape))
    
    time_elapsed = timer() - start
    print("Det Pre Time:", time_elapsed)

    img = np.transpose(raw_img2, (2,0,1))
    img = img[np.newaxis, :]
    img = cls_std_scale * (img.astype(np.float32) - cls_mean_val)

    start = timer()
    mod.forward(Batch([mx.nd.array(img)]))
    mod.get_outputs()[0].wait_to_read()
    time_elapsed = timer() - start
    print("Det Time:", time_elapsed)

    detections = mod.get_outputs()[0].asnumpy()
    
    res = None
    for i in range(detections.shape[0]):
        det = detections[i, :, :]
        res = det[np.where(det[:, 0] >= 0)[0]]

    final_dets = np.empty(shape=(0, 5))

    for i in range(res.shape[0]):
        cls_id = int(res[i, 0])
        if cls_id >= 0:
            score = res[i, 1]
            if score > 0.6:
                xmin = int(res[i, 2] * data_shape)
                ymin = int(res[i, 3] * data_shape)
                xmax = int(res[i, 4] * data_shape)
                ymax = int(res[i, 5] * data_shape)

                if w > h:
                    pad = w - h
                    xmin2 = xmin / scale
                    xmax2 = xmax / scale
                    ymin2 = ymin / scale - (pad // 2)
                    ymax2 = ymax / scale - (pad // 2)
                else:
                    pad = h - w
                    xmin2 = xmin / scale - (pad // 2)
                    xmax2 = xmax / scale - (pad // 2)
                    ymin2 = ymin / scale
                    ymax2 = ymax / scale


                final_dets = np.vstack((final_dets, [xmin2, ymin2, xmax2, ymax2, score]))

    if final_dets.shape[0] > 0:
        perm_idx = np.argsort(final_dets[:,4], axis=0)[::-1]
        return final_dets[perm_idx[0]]
    else:
        return None



def cls_img(raw_img, mod):
    
    start = timer()
    img = cv2.resize(raw_img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))[np.newaxis,:]
    img = cls_std_scale * (img.astype(np.float32) - cls_mean_val)
    time_elapsed = timer() - start
    print("Cls Pre Time:", time_elapsed)
    #img = img.astype(np.float32) - cls_mean_val

    start = timer()
    mod.forward(Batch([mx.nd.array(img)]), is_train=False)
    mod.get_outputs()[0].wait_to_read()
    time_elapsed = timer() - start
    print("Cls Time:", time_elapsed)

    pred = mod.get_outputs()[0].asnumpy()
    return np.argmax(pred), pred

if __name__ == "__main__":
    testdir = sys.argv[1] #r'C:\Users\ld\Desktop\ylb\zp'

    imgfiles = [i for i in os.listdir(testdir) if i.endswith('.jpg')]

    det_mod = get_detection_mod()

    cls_mod = get_classification_mod()

    for fn in imgfiles:
        start = timer()
        #fn = '222.jpg'
        #raw_img = cv2.imread(fn)
        raw_img = cv2.imread(testdir+'/'+fn)
        time_elapsed = timer() - start
        print("IO Time:", time_elapsed)
        dets = det_img(raw_img, det_mod)
        #print(dets)
        if dets is not None:
            xmin = int(dets[0])
            ymin = int(dets[1])
            xmax = int(dets[2])
            ymax = int(dets[3])
            roi_w = xmax - xmin + 1
            roi_h = ymax - ymin + 1

            if roi_w > roi_h:
                pad = roi_w - roi_h
                ymin = ymin - pad // 2
                ymax = ymax + (pad+1)//2
            else:
                pad = roi_h - roi_w
                xmin = xmin - pad // 2
                xmax = xmax + (pad+1)//2
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(raw_img.shape[1]-1, xmax)
            ymax = min(raw_img.shape[0]-1, ymax)
            if xmin >= xmax or ymin >= ymax:
                print("NONE")
                continue
            
            roi_img = raw_img[ymin:ymax+1, xmin:xmax+1,:]

            ylb_type, pred = cls_img(roi_img, cls_mod)

            cv2.rectangle(raw_img,(xmin, ymin), (xmax, ymax), (0,255,0), 3)

            #print(pred)
            disp_str = cls_names[ylb_type] + ' ' + str(pred[0,ylb_type])
            print(disp_str)
            cv2.putText(raw_img, disp_str, (10,44), 0, 2.4, (255,0,255))
        else:
            print("NONE")
        if raw_img.shape[0] > 1000 or raw_img.shape[1] > 1000:
            raw_img=cv2.resize(raw_img, (0,0), fx=0.3, fy = 0.3)
        cv2.imshow("w", raw_img)
        cv2.waitKey()


