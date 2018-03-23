import os
import cv2
import json
import numpy as np
from imdb import Imdb

class ylbdb(Imdb):
    def __init__(self, name, root_path, is_train = False):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'zp')
        self.extension = '.jpg'
        self.is_train = is_train
        self.classes = self._load_class_names('names.txt',root_path)
        self.num_classes = len(self.classes)

        if self.is_train:
            self.image_set_index, self.labels = self._load_ylb_dat()
        else:
            self.image_set_index, _ = self._load_ylb_dat()
        self.num_images = len(self.image_set_index)

    def _load_ylb_dat(self):
        #mask_path = self.root_path + '/zp_mask'

        with open(self.data_path + '/bj2.json') as f:
            db = json.load(f)

        imgfiles = [i for i in os.listdir(self.data_path) if i.endswith('.jpg')]
        #maskfiles = [i for i in os.listdir(mask_path) if i.endswith('.png')]

        db_dict = {}

        for i,j in enumerate(db):
            fn = os.path.split(j['filename'])[1]
            db_dict[fn] = i

        labels = []

        for i in imgfiles:
            #mask = cv2.imread(mask_path + '/' + i[:-4] + '.png')
            img = cv2.imread(self.data_path + '/' + i)
            
            h = img.shape[0]
            w = img.shape[1]
            #print(h,w)

            idx = db_dict[i]
            info = db[idx]
            anno = info['annotations']

            label = []

            x0 = info['rect']['x0']
            y0 = info['rect']['y0']
            x1 = info['rect']['x1']
            y1 = info['rect']['y1']

            x0 = x0 / float(w)
            y0 = y0 / float(h)
            x1 = x1 / float(w)
            y1 = y1 / float(h)
            
            print(x0, y0, x1, y1)

            # nonzero = mask.nonzero()
            # y0 = min(nonzero[0])
            # y1 = max(nonzero[0])
            # x0 = min(nonzero[1])
            # x1 = max(nonzero[1])
            #print(x0, y0, x1, y1)
            label.append([0, x0, y0, x1, y1])

            labels.append(np.array(label))

        imgfiles = [self.data_path + '/' + i for i in imgfiles]

        return imgfiles, labels

    def image_path_from_index(self, index):
        return self.image_set_index[index]

    def label_from_index(self, index):
        return self.labels[index]