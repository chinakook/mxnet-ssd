import os
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
#os.environ['partial_shaping']='True'
import numpy as np
#import math
import cv2
from timeit import default_timer as timer
import mxnet as mx
#from xml.etree import ElementTree as ET
#from xml.dom import minidom
#import codecs
#import shutil
from symbol.symbol_factory import get_symbol

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

data_shape = 300
img_shape = 128
#mean_img = np.array([123,117,104], dtype=np.float32)
#mean_img = mean_img[:,np.newaxis,np.newaxis]
cls_mean_val = np.array([[[123.68]],[[116.78]],[[103.94]]])
#cls_names = ['red', 'green', 'yellow']
cls_name = ['obj']
cls_std_scale = 0.017

ctx = mx.cpu()
#def writeVOCXML(fn, fn_path, w , h, bboxes, outf):
#    folder = ET.Element('folder')
#    folder.text = '.'
#    filename = ET.Element('filename')
#    filename.text=fn[:-4] # without ext
#    path = ET.Element('path')
#    path.text = fn_path
#    source = ET.Element('source')
#    database = ET.Element('database')
#    source.append(database)
#    database.text = 'Unknown'
#    size_tag = ET.Element('size')
#    width = ET.Element('width')
#    width.text = str(w)
#    height = ET.Element('height')
#    height.text = str(h)
#    depth = ET.Element('depth')
#    depth.text = '3'
#    size_tag.append(width)
#    size_tag.append(height)
#    size_tag.append(depth)
#    segmented = ET.Element('segmented')
#    segmented.text = '0'
#
#    root = ET.Element('annotation')
#    tree = ET.ElementTree(root)
#    root.append(folder)
#    root.append(filename)
#    root.append(path)
#    root.append(source)
#    root.append(size_tag)
#    root.append(segmented)
#
#    for b in bboxes:
#        name = ET.Element('name')
#        name.text = 'obj'
#        pose = ET.Element('pose')
#        pose.text = 'Unspecified'
#        truncated = ET.Element('truncated')
#        truncated.text = '0'
#        difficult = ET.Element('difficult')
#        difficult.text = '0'
#        bndbox = ET.Element('bndbox')
#        object_tag = ET.Element('object')
#        root.append(object_tag)
#        object_tag.append(name)
#        object_tag.append(pose)
#        object_tag.append(truncated)
#        object_tag.append(difficult)
#        object_tag.append(bndbox)
#        xmin=ET.Element('xmin')
#        ymin=ET.Element('ymin')
#        xmax=ET.Element('xmax')
#        ymax=ET.Element('ymax')
#        xmin.text=str(int(b[0]))
#        ymin.text=str(int(b[1]))
#        xmax.text=str(int(b[2]))
#        ymax.text=str(int(b[3]))
#        
#        
#        bndbox.append(xmin)
#        bndbox.append(ymin)
#        bndbox.append(xmax)
#        bndbox.append(ymax)
#
#    def prettify(elem):
#        """Return a pretty-printed XML string for the Element.
#        """
#        rough_string = ET.tostring(elem, 'utf-8')
#        reparsed = minidom.parseString(rough_string)
#        return reparsed.toprettyxml(indent="\t")
#
#    xml_text = prettify(root)
#
#    f=codecs.open(outf,'w','utf-8')
#    f.write(xml_text)
#    f.close()

def get_classification_mod():
    sym, arg_params, aux_params = mx.model.load_checkpoint('./model/dg', 100)    
    all_layers = sym.get_internals()
    ss = all_layers['fc7_output']
    ss = mx.symbol.SoftmaxActivation(ss, mode='channel')
    mod = mx.mod.Module(symbol=ss, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1, 3, img_shape, img_shape))], label_shapes=None,force_rebind=False)
    mod.set_params(arg_params=arg_params, aux_params=aux_params,allow_missing=True, allow_extra=True)
    return mod

def get_detection_mod():

    #net = get_symbol('vgg16_reduced', data_shape, num_classes=1, nms_thresh=0.5,force_nms=True, nms_topk=400)
    #sym, args, auxs = mx.model.load_checkpoint('./model/ssd_vgg16_reduced_300', 126)
    net = get_symbol('mobilenet_little', data_shape, num_classes=1, nms_thresh=0.5,force_nms=True, nms_topk=400)
    #sym, args, auxs = mx.model.load_checkpoint('./model/ssd_mobilenet_300', 150)
    sym, args, auxs = mx.model.load_checkpoint('./model/ssd_dg_300', 60)
    
    mod = mx.mod.Module(net, label_names=None, context=ctx)

    mod.bind(for_training=False, data_shapes=[('data', (1, 3, data_shape, data_shape))])
    mod.set_params(args, auxs, allow_extra=True)

    return mod
def cls_img(batch_img, mod):
    
    start = timer()
    #img_shape = 224
    #time_elapsed = timer() - start
    #print("Cls Pre Time:", time_elapsed)
    #img = img.astype(np.float32) - cls_mean_val
    #print batch_img.shape
    n = batch_img.shape[0]
    #x = batch_img[0]
    #x = x[np.newaxis,:,:,:]
    mod.reshape(data_shapes=[('data', (n, 3, img_shape, img_shape))])
    start = timer()
    mod.forward(Batch([mx.nd.array(batch_img)]), is_train=False)
    mod.get_outputs()[0].wait_to_read()
    time_elapsed = timer() - start
    print("Cls Time:", time_elapsed)
    #print(mod.get_outputs()[0])
    pred = mod.get_outputs()[0].asnumpy()
    pred = pred[:,:,0,0]
    #print pred.shape
    return np.argmax(pred,axis=1), pred

def det_img(raw_img, mod,fn):
    #raw_img = cv2.imread(testdir + '/' + fn)
    start = timer()
    image_path = fn

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

    #final_dets = np.empty(shape=(0, 5))
    final_dets = np.empty(shape=(0, 6))
    #print res.shape[0]
    for i in range(res.shape[0]):
        cls_id = int(res[i, 0])
        if cls_id >= 0:
            score = res[i, 1]
            if score > 0.61:
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


                final_dets = np.vstack((final_dets, [xmin2, ymin2, xmax2, ymax2, score,cls_id]))

    if final_dets.shape[0] > 0:
        #perm_idx = np.argsort(final_dets[:,4], axis=0)[::-1]
        return final_dets
    else:
        print image_path
        return None




if __name__ == "__main__":
    #testdir = r'E:\BaiduYunDownload\Archive2017.8.16\1005'
    testdir = sys.argv[1]
    savedir = sys.argv[2]

    imgfiles = [i for i in os.listdir(testdir) if i.endswith('.jpg')]

    det_mod = get_detection_mod()
    '''
     get classify model
    '''
    cls_mod = get_classification_mod()

    for fn in imgfiles:
        #print(fn)
        start = timer()
        #fn = '222.jpg'
        #raw_img = cv2.imread(fn)
        raw_img = cv2.imread(testdir+'/'+fn)
        h,w,channel = raw_img.shape
        time_elapsed = timer() - start
        print("IO Time:", time_elapsed)
        dets = det_img(raw_img, det_mod,fn)
        #print(dets.shape)
        if dets is not None:
            batch_img = []
            #writeVOCXML(fn,testdir+'/'+fn,w,h,dets,testdir+'/'+fn[:-4]+'.xml')            
            num = dets.shape[0]
            bnd_box = []
            for j in range(num):
                det = dets[j]
                xmin = int(det[0])
                ymin = int(det[1])
                xmax = int(det[2])
                ymax = int(det[3])
                roi_w = xmax - xmin + 1
                roi_h = ymax - ymin + 1

                # if roi_w > roi_h:
                #     pad = roi_w - roi_h
                #     ymin = ymin - pad // 2
                #     ymax = ymax + (pad+1)//2
                # else:
                #     pad = roi_h - roi_w
                #     xmin = xmin - pad // 2
                #     xmax = xmax + (pad+1)//2
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(raw_img.shape[1]-1, xmax)
                ymax = min(raw_img.shape[0]-1, ymax)
                if xmin >= xmax or ymin >= ymax:
                    print("NONE")
                    continue
                bound_box = np.array([xmin,ymin,xmax,ymax])
                bnd_box.append(bound_box)
                #roi_img = raw_img[ymin:ymax+1, xmin:xmax+1,:]
                x1 = max(xmin,0)
                y1 = max(ymin, 0)
                x2 = min(xmax, w)
                y2 = min(ymax, h)
                roi_img = raw_img[y1:y2, x1:x2,:]
                roi_rs_img = cv2.resize(roi_img,(img_shape ,img_shape))
                #cv2.imshow("w", roi_rs_img)
                #cv2.waitKey()
                roi_rs_img = cv2.cvtColor(roi_rs_img, cv2.COLOR_BGR2RGB)
                roi_rs_img = np.transpose(roi_rs_img,(2,0,1))
                #roi_rs_img = np.array(roi_rs_img,dtype=np.float16)
                roi_rs_img = cls_std_scale * (roi_rs_img.astype(np.float32) - cls_mean_val)
                
                batch_img.append(roi_rs_img)
            #ylb_type, pred = cls_img(roi_img, cls_mod)
            
            '''
                pred the class
            '''
            _batch = np.array(batch_img)
            _bnd_box = np.array(bnd_box)
            c_type,pred = cls_img(_batch,cls_mod)
            #print c_type
            for i in range(_bnd_box.shape[0]):
                bd = _bnd_box[i,:]
                if c_type[i] == 0:
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                #cv2.rectangle(raw_img,(bd[0]-50,bd[1]-50),(bd[2]+50,bd[3]+50),color,8)
                cv2.rectangle(raw_img,(bd[0],bd[1]),(bd[2],bd[3]),color,8)
            
            #cv2.imwrite( '/home/caizhendong/git/test/ylb_det/test/'+fn, raw_img)        
        #if raw_img.shape[0] > 1000 or raw_img.shape[1] > 1000:
        #    raw_img=cv2.resize(raw_img, (0,0), fx=0.3, fy = 0.3)
        #cv2.imshow("w", raw_img)
        #cv2.waitKey()

        cv2.imwrite(savedir + '/' + fn[:-4]+'_res.jpg', raw_img)


