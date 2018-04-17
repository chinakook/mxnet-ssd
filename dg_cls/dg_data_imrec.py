
# coding: utf-8

# In[1]:


import os
import cv2
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import json

import random
import numpy as np
import mxnet as mx


# In[2]:


#img_path = '\\DDL\\dg_cls\\total_cls'
# 这个是数据路径，每次训练不同的数据之前要给不同的路径，然后用dg_data_imrec.py生成数据
img_path = 'E:\\DDL\\dg_data\\arrow\\re2012'
anno_dict = {}
for root, dirs, files in os.walk(img_path):
    for name in files:
        if name.endswith(".xml"):
            anno_dict[os.path.join(root, name)] = os.path.join(root, name[:-4]+'.jpg')
            

xpad = 100
ypad = 50
scale = 0.4
new_annos = []

DEBUG=False
# In[4]:


train_record = mx.recordio.MXIndexedRecordIO('./dg_train.idx',
    './dg_train.rec', 'w')
val_record = mx.recordio.MXIndexedRecordIO('./dg_val.idx',
    './dg_val.rec', 'w')


it = 0
iv = 0
# 这个是验证数据的比例，一般你训练数据如果太小，就不搞验证数据了，如果数量很大，可以设置一定比例的验证数据
val_p = 0.0
for annoiter in anno_dict.items():
    img = cv2.imread(annoiter[1])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    anno = ET.parse(annoiter[0])
    obj_node=anno.getiterator("object")
    rects = []
    for obj in obj_node:
        
        
        bndbox = obj.find('bndbox')
        name = obj.find('name')
        # 这是你定义的类型，看清楚了，0代表没问题的，1代表NG的，你要加的话，可以在后面再加or
        if name.text =='obj' or name.text == 'x' or name.text == 'ok':
            c = 0
        else:
            c = 1
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        xmin_inner = max(0,int(xmin - xpad))
        ymin_inner = max(0,int(ymin - ypad))
        xmax_inner = min(img.shape[1]-1, int(xmax + xpad))
        ymax_inner = min(img.shape[0]-1, int(ymax + ypad))
        
        roi = img[ymin_inner:ymax_inner+1, xmin_inner:xmax_inner+1,:]
        roi = cv2.resize(roi, (0,0), fx=scale, fy=scale)
        
        xmin = (xmin -xmin_inner)*scale
        xmax = (xmax -xmin_inner)*scale
        ymin = (ymin- ymin_inner)*scale
        ymax = (ymax -ymin_inner)*scale
        
        xmin = max(0,int(xmin))
        ymin = max(0,int(ymin))
        xmax = min(roi.shape[1]-1, int(xmax))
        ymax = min(roi.shape[0]-1, int(ymax))
        
        label = np.array([c, xmin, ymin, xmax, ymax], dtype=np.int32)

        if DEBUG:
            cv2.rectangle(roi, (xmin, ymin), (xmax, ymax),color=(0,255,0))
            cv2.imshow("w", roi)
            cv2.waitKey()
            continue

        r = random.random()
        
        if r < val_p:
            p = mx.recordio.pack_img((0,label,iv,0), roi)
            val_record.write_idx(iv,p)
            iv = iv + 1
        else:
            p = mx.recordio.pack_img((0,label,it,0), roi)
            train_record.write_idx(it,p)
            it = it + 1
        
        #cv2.imwrite(dst_path+'/'+str(k) + '.jpg', roi)
        
        #new_annos.append({'filename': str(k) + '.jpg', 'type': c, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax})
        
#         rects.append(((xmin, ymin), (xmax, ymax), c))
#     for r in rects:
#         if r[2]:
#             cv2.rectangle(img, r[0], r[1], (255,0,0),16)
#         else:
#             cv2.rectangle(img, r[0], r[1], (0,255,0),16)
#     plt.imshow(img)
#     plt.show()

train_record.close()
val_record.close()
