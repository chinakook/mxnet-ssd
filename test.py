import os
import numpy as np
import cv2
from timeit import default_timer as timer
import mxnet as mx

from symbol.symbol_factory import get_symbol

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

data_shape = 300

net = get_symbol('vgg16_reduced', data_shape, num_classes=20, nms_thresh=0.5,
            force_nms=True, nms_topk=400)
#detector = Detector(net, '/home/dingkou/dev/ylb_det/model/ssd_vgg16_reduced_300', 108, 300, (123,117,104), ctx=mx.gpu())
#exit()
sym, args, auxs = mx.model.load_checkpoint('./model/ssd_vgg16_reduced_300', 108)

mod = mx.mod.Module(net, label_names=None, context=mx.gpu())

mod.bind(data_shapes=[('data', (1, 3, data_shape, data_shape))])
mod.set_params(args, auxs, allow_extra=True)


testdir = '/mnt/6B133E147DED759E/tmp/ylb/zp'

imgfiles = [i for i in os.listdir(testdir) if i.endswith('.jpg')]

mean_img = np.array([123,117,104], dtype=np.float32)
mean_img = mean_img[:,np.newaxis,np.newaxis]

for fn in imgfiles:
    raw_img = cv2.imread(testdir + '/' + fn)

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    h = raw_img.shape[0]
    w = raw_img.shape[1]

    if w > h:
        pad = w - h
        raw_img = cv2.copyMakeBorder(raw_img, pad//2, (pad+1)//2, 0, 0, cv2.BORDER_CONSTANT)
    else:
        pad = h - w
        raw_img = cv2.copyMakeBorder(raw_img, 0, 0, pad//2, (pad+1)//2, cv2.BORDER_CONSTANT)
    
    raw_img = cv2.resize(raw_img, (data_shape,data_shape))

    img = np.transpose(raw_img, (2,0,1))
    img = img[np.newaxis, :]
    img = img.astype(np.float32) - mean_img

    
    #Batch[mx.nd.array(img)]
    start = timer()
    mod.forward(Batch([mx.nd.array(img)]))
    mod.get_outputs()[0].wait_to_read()
    time_elapsed = timer() - start

    print(time_elapsed)

    detections = mod.get_outputs()[0].asnumpy()
    # result = []
    res = None
    for i in range(detections.shape[0]):
        det = detections[i, :, :]
        res = det[np.where(det[:, 0] >= 0)[0]]
    #     result.append(res)

    #print(res.shape)
    for i in range(res.shape[0]):
        cls_id = int(res[i, 0])
        if cls_id >= 0:
            score = res[i, 1]
            if score > 0.6:
                xmin = int(res[i, 2] * data_shape)
                ymin = int(res[i, 3] * data_shape)
                xmax = int(res[i, 4] * data_shape)
                ymax = int(res[i, 5] * data_shape)

                cv2.rectangle(raw_img,(xmin, ymin), (xmax, ymax), (0,255,0), 3)

    cv2.imshow("w", raw_img[:,:,::-1])
    cv2.waitKey()

    