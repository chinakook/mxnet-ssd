
# coding: utf-8

# In[1]:


import os

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
import mxnet.autograd as ag

from mxnet.gluon.data import Dataset, DataLoader
from mxnet import image

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

import time


# In[2]:


train_record = mx.recordio.MXIndexedRecordIO('./dg_train.idx', './dg_train.rec', 'r')
val_record = mx.recordio.MXIndexedRecordIO('./dg_train.idx', './dg_train.rec', 'r')
#val_record = mx.recordio.MXIndexedRecordIO('./dg_val.idx', './dg_val.rec', 'r')


# In[3]:


mean_val = np.array([[[123.68]],[[116.78]],[[103.94]]])
std_scale = 0.017

class DGDataSet(Dataset):
    def __init__(self, record, image_shape, aug):
        self.record = record
        self.image_shape = image_shape
        self.aug = aug
        # 对训练数据进行增强，可参考https://github.com/aleju/imgaug进行修改
        self.seq = iaa.Sequential([
            iaa.Add((-20,20)),
            iaa.AddToHueAndSaturation((-30,30)),
            iaa.GaussianBlur((0.0,1.5)),
            iaa.ContrastNormalization((0.5, 1.5)),
            iaa.Grayscale((0.0,1.0))
        ])
        
    def __len__(self):
        return len(self.record.keys)
    
    def __getitem__(self, idx):
        p = self.record.read_idx(idx)
        header, img = mx.recordio.unpack_img(p)

        dg_type = header.label[0]
        x0 = header.label[1]
        y0 = header.label[2]
        x1 = header.label[3]
        y1 = header.label[4]
        
        dst_w = self.image_shape
        dst_h = self.image_shape
        obj_w = x1 - x0
        obj_h = y1 - y0
        xc = (x0 + x1) / 2.
        yc = (y0 + y1) / 2.
        #scale = dst_w / max(obj_w, obj_h)
        scale_w = dst_w / float(obj_w)
        scale_h = dst_h / float(obj_h)

        if self.aug:
            sj = random.uniform(0.9, 1.3)
            scale_jitter_w = sj #* random.choice([-1,1])
            scale_jitter_h = sj #* random.choice([-1,1])

            scale_w *= scale_jitter_w
            scale_h *= scale_jitter_h

            dst_w_jitter = random.uniform(-25,25)
            dst_h_jitter = random.uniform(-20,20)
            theta = random.uniform(-np.pi/45, np.pi/45)

            T0 = np.array([[1,0,-xc],[0,1,-yc],[0,0,1]])
            S = np.array([[scale_w,0,0],[0, scale_h,0],[0,0,1]])
            R = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0],[0,0,1]])
            T1 = np.array([[1,0,dst_w/2. +dst_w_jitter],[0,1,dst_h/2. +dst_h_jitter],[0,0,1]])
            M = np.dot(S, T0)
            M = np.dot(R, M)
            M = np.dot(T1, M)
            M_warp = M[0:2,:]

            dst_img = cv2.warpAffine(img, M_warp, dsize=(int(dst_w), int(dst_h)))
            dst_img = self.seq.augment_image(dst_img)
        else:
            dst_img = img[int(y0):int(y1)+1, int(x0):int(x1)+1, :]
            dst_img = cv2.resize(dst_img, (int(dst_w), int(dst_h)))
        # 显示增强后的图像
        #cv2.imshow("w", dst_img)
        #cv2.waitKey()
        
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
        dst_img = np.transpose(dst_img,(2,0,1))
        dst_img = std_scale * (dst_img.astype(np.float32) - mean_val)

        return mx.nd.array(dst_img), dg_type

data_shape=224
		
dg_train = DGDataSet(train_record, data_shape, True)
train_loader = DataLoader(dg_train, batch_size=64, shuffle=True, last_batch='rollover')

dg_val = DGDataSet(val_record, data_shape, False)
val_loader = DataLoader(dg_val, batch_size=16, shuffle=False, last_batch='keep')

ctx = [mx.gpu(0)]

net = gluon.model_zoo.vision.MobileNetV2(1.0, 2)
#net = gluon.model_zoo.vision.resnet18_v2()
#net.output=nn.Dense(2)

net.hybridize()

net.collect_params().initialize(ctx=ctx)

metrics = [mx.metric.Accuracy()]

criterion = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {
    'learning_rate': 0.045,
    'wd': 0.0001,
    'momentum': 0.9,
    'clip_gradient': None
})


def accuracy(output, labels):
    return nd.sum(nd.argmax(output, axis=1) == labels).asscalar()

def evaluate(net, data_iter):
    valctx = [mx.gpu(0)]
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        #data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        batch_size = data.shape[0]
        dlist = gluon.utils.split_and_load(data, valctx)
        llist = gluon.utils.split_and_load(label, valctx)
        #output = net(data)
        preds = [net(X) for X in dlist]
        for i in range(len(preds)):
            l = criterion(preds[i], llist[i])
            loss += l.sum().asscalar()
            acc += accuracy(preds[i], llist[i])
        n += batch_size
    return loss, acc/float(n)


# In[ ]:


num_epochs = 150
for epoch in range(num_epochs):
    if epoch > 0 and epoch <=120 and (epoch % 60) == 0:
        trainer.set_learning_rate(trainer.learning_rate*0.1)
    t0 = time.time()
    total_loss = 0
    for m in metrics:
        m.reset()
    for data, label in train_loader:
        batch_size = data.shape[0]
        dlist = gluon.utils.split_and_load(data, ctx)
        llist = gluon.utils.split_and_load(label, ctx)
        with ag.record():
            #losses = [criterion(net(X), y, m) for X, y in zip(dlist, llist, mlist)]
            preds = [net(X) for X in dlist]
            losses = []
            for i in range(len(preds)):
                l = criterion(preds[i], llist[i])
                losses.append(l)
        for l in losses:
            l.backward()
        total_loss += sum([l.sum().asscalar() for l in losses])
        trainer.step(batch_size)
        #print(label.shape, preds.shape)
        for m in metrics:
            m.update(labels=llist, preds=preds)
    
    for m in metrics:
        name, value = m.get()

    
    t1 = time.time()
    print(epoch, t1-t0, total_loss, name, value)#, val_loss, val_acc)
    # 验证数据精度，每20个迭代验证一次
    if epoch > 0 and epoch % 20 == 0:
        val_loss, val_acc = evaluate(net, val_loader)
        net.export('model/dggluon', epoch)
        print("验证精度: ", val_acc, val_loss)
        print("模型已保存，请最后选一个验证精度高的模型，不要训练太久，验证精度达到就停掉算了")
net.export('model/dggluon', num_epochs)

