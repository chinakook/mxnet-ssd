# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import numpy as np
from dataset.imdb import Imdb
import xml.etree.ElementTree as ET


class dgdb(Imdb):
    """
    Base class for loading datasets
    Parameters:
    ----------
    name : str
        name for this dataset
    classes : list or tuple of str
        class names in this dataset
    list_file : str
        filename of the image list file
    root_dir : str
        root directory,(include image and label)
    extension : str
        by default .jpg
    label_extension : str
        by default .xml
    shuffle : bool
        whether to shuffle the initial order when loading this dataset,
        default is True
    """
    def __init__(self, name, classes, root_dir,extension='.jpg', label_extension='.xml', shuffle=True):
        if isinstance(classes, list) or isinstance(classes, tuple):
            num_classes = len(classes)
        elif isinstance(classes, str):
            with open(classes, 'r') as f:
                classes = [l.strip() for l in f.readlines()]
                num_classes = len(classes)
        else:
            raise ValueError("classes should be list/tuple or text file")
        assert num_classes > 0, "number of classes must > 0"
        super(dgdb, self).__init__(name + '_' + str(num_classes))
        self.classes = classes
        self.num_classes = num_classes
        self.extension = extension
        self.label_extension = label_extension
        self.list_file = self._creat_list_file(root_dir)
        self.root_dir = root_dir


        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        self.labels = self._load_image_labels()
        #print self.classes
    def _creat_list_file(self, root_dir):
        assert os.path.exists(root_dir),'root_dir not exists!'
        ext = str(self.label_extension)
        #print ext
        list_file = [i[:-4] for i in os.listdir(root_dir) if i.endswith(ext)]
        print ("total file: " ,len(list_file))
        return list_file
    

    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        
        #assert os.path.exists(self.list_file), 'Path does not exists: {}'.format(self.list_file)
        assert len(self.list_file)>0, 'could not find files'
        #print len(self.list_file)
        image_set_index = self.list_file
        #image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.root_dir, name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.root_dir, index + self.label_extension)
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []

        # load ground-truth from xml annotations
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            tree = ET.parse(label_file)
            root = tree.getroot()
            size = root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            #print width,height,idx
            label = []

            '''
                CHANGED:
                #if cls_name not in self.classes:
                #    continue
                #cls_id = self.classes.index(cls_name)

            '''

            for obj in root.iter('object'):
                #difficult = int(obj.find('difficult').text)
                # if not self.config['use_difficult'] and difficult == 1:
                #     continue
                cls_name = obj.find('name').text
                #if cls_name not in self.classes:
                #    continue
                #cls_id = self.classes.index(cls_name)
                cls_id = 0
                xml_box = obj.find('bndbox')
                xmin = float(xml_box.find('xmin').text) / width
                ymin = float(xml_box.find('ymin').text) / height
                xmax = float(xml_box.find('xmax').text) / width
                ymax = float(xml_box.find('ymax').text) / height
                label.append([cls_id, xmin, ymin, xmax, ymax])
            temp.append(np.array(label))
        return temp