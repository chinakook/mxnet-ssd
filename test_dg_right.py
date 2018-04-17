import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'

import sys
import numpy as np
import math
import cv2
from timeit import default_timer as timer
import mxnet as mx
from xml.etree import ElementTree as ET
from xml.dom import minidom
import codecs
import shutil

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

det_data_shape = 608
cls_data_shape = 224

cls_mean_val = np.array([[[123.68]],[[116.78]],[[103.94]]])
#cls_names = ['red', 'green', 'yellow']
cls_name = ['obj']
cls_std_scale = 0.017

ctx = mx.cpu()

def writeVOCXML(fn, fn_path, w , h, bboxes, outf):
    folder = ET.Element('folder')
    folder.text = '.'
    filename = ET.Element('filename')
    filename.text=fn[:-4] # without ext
    path = ET.Element('path')
    path.text = fn_path
    source = ET.Element('source')
    database = ET.Element('database')
    source.append(database)
    database.text = 'Unknown'
    size_tag = ET.Element('size')
    width = ET.Element('width')
    width.text = str(w)
    height = ET.Element('height')
    height.text = str(h)
    depth = ET.Element('depth')
    depth.text = '3'
    size_tag.append(width)
    size_tag.append(height)
    size_tag.append(depth)
    segmented = ET.Element('segmented')
    segmented.text = '0'

    root = ET.Element('annotation')
    tree = ET.ElementTree(root)
    root.append(folder)
    root.append(filename)
    root.append(path)
    root.append(source)
    root.append(size_tag)
    root.append(segmented)

    for b in bboxes:
        name = ET.Element('name')
        name.text = 'ng' if b[4] else 'obj'
        pose = ET.Element('pose')
        pose.text = 'Unspecified'
        truncated = ET.Element('truncated')
        truncated.text = '0'
        difficult = ET.Element('difficult')
        difficult.text = '0'
        bndbox = ET.Element('bndbox')
        object_tag = ET.Element('object')
        root.append(object_tag)
        object_tag.append(name)
        object_tag.append(pose)
        object_tag.append(truncated)
        object_tag.append(difficult)
        object_tag.append(bndbox)
        xmin=ET.Element('xmin')
        ymin=ET.Element('ymin')
        xmax=ET.Element('xmax')
        ymax=ET.Element('ymax')
        xmin.text=str(int(b[0]))
        ymin.text=str(int(b[1]))
        xmax.text=str(int(b[2]))
        ymax.text=str(int(b[3]))
        
        
        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)

    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")

    xml_text = prettify(root)

    f=codecs.open(outf,'w','utf-8')
    f.write(xml_text)
    f.close()

def get_detection_mod():

    sym, arg_params, aux_params = mx.model.load_checkpoint('E:/DDL/dg_detection_py/model/deploy_ssd_mobilenet_little_608', 60)

    mod = mx.mod.Module(symbol=sym, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,det_data_shape,det_data_shape))], label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    return mod

def get_classification_mod():

    sym, arg_params, aux_params = mx.model.load_checkpoint('E:/DDL/dg_detection_py/model/dggluon', 0)
    all_layers = sym[0].get_internals()
    mod = mx.mod.Module(symbol=all_layers['mobilenetv20_output_pred_fwd_output'], context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,cls_data_shape,cls_data_shape))], label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    return mod

def det_img(raw_img, mod, fn):
    #raw_img = cv2.imread(testdir + '/' + fn)
    start = timer()

    h = raw_img.shape[0]
    w = raw_img.shape[1]
    ascpect_x =  w / float(det_data_shape)
    ascpect_y =  h / float(det_data_shape)

    raw_img2 = cv2.resize(raw_img, (det_data_shape, det_data_shape))
    raw_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2RGB)
    
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
            if score > 0.6:
                xmin = res[i, 2] * det_data_shape * ascpect_x
                ymin = res[i, 3] * det_data_shape * ascpect_y
                xmax = res[i, 4] * det_data_shape * ascpect_x
                ymax = res[i, 5] * det_data_shape * ascpect_y

                final_dets = np.vstack((final_dets, [xmin, ymin, xmax, ymax, score, cls_id]))
    rects = np.empty(shape=(0,4), dtype=np.int64)
    for det in final_dets:
        x0 = int(max(0, int(det[0])))
        y0 = int(max(0, int(det[1])))
        x1 = int(min(w - 1, int(det[2])))
        y1 = int(min(h - 1, int(det[3])))
        if x0 >= x1 or y0 >= y1:
            continue
        rects = np.vstack((rects, [x0, y0, x1, y1]))


    return rects

def cls_roi(roi, mod):
    raw_img2 = cv2.resize(roi, (cls_data_shape, cls_data_shape))
    raw_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2RGB)
    img = np.transpose(raw_img2, (2,0,1))
    img = img[np.newaxis, :]
    img = cls_std_scale * (img.astype(np.float32) - cls_mean_val)

    mod.forward(Batch([mx.nd.array(img)]))
    preds = mod.get_outputs()[0].asnumpy()
    preds = np.squeeze(preds)
    return preds[0] < preds[1]

if __name__ == "__main__":
    xmldir = r'E:\DDL\dg_detection_py\xml'
    testdir = r'E:\DDL\dg_data\arrow\re2012'
    #testdir = sys.argv[1]

    imgfiles = [i for i in os.listdir(testdir) if i.endswith('.jpg')]

    det_mod = get_detection_mod()

    cls_mod = get_classification_mod()

    for fn in imgfiles:
        fn_path = testdir+'/'+fn
        raw_img = cv2.imread(fn_path)

        dets = det_img(raw_img, det_mod,fn)
        classes = np.zeros(shape=(dets.shape[0],1), dtype=np.int64)
        for i,(xmin, ymin, xmax, ymax) in enumerate(dets):
            c = cls_roi(raw_img[ymin:ymax+1, xmin:xmax+1,:], cls_mod)
            classes[i] = c

        dets_with_cls = np.hstack((dets, classes))

        writeVOCXML(fn, fn_path, raw_img.shape[1], raw_img.shape[0], dets_with_cls, xmldir + '/' + fn[:-4] + '.xml')

        #可视化
        # for i,(xmin, ymin, xmax, ymax) in enumerate(dets_with_cls):
            # color = (0,255,0) if c == 0 else (0,0,255)
            # cv2.rectangle(raw_img,(xmin, ymin), (xmax, ymax), color, 5)
        
        # if raw_img.shape[0] > 1000 or raw_img.shape[1] > 1000:
        #     raw_img=cv2.resize(raw_img, (0,0), fx=0.3, fy = 0.3)
        # cv2.imshow("w", raw_img)
        # cv2.waitKey()


