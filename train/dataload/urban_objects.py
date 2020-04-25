# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree
from .common import object_positions


name2id = {
    "no_obj": 0,
    "bicycle": 1,
    "bus": 2,
    "car": 3,
    "motorbike": 4,
    "person": 5,
    "trafficsignal": 6,
    "trafficlight": 7
}
id2name = {v: k for k, v in name2id.items()}


def load(root_dir):
    train_x, train_y, val_x, val_y = [], [], [], []
    image_dir = os.path.join(root_dir, 'JPEGImages')
    annotation_dir = os.path.join(root_dir, 'Annotations')
    
    for t in ('train', 'val'):
        with open(os.path.join(root_dir, 'ImageSets', '{}.txt'.format(t))) as fp:
            for line in fp:
                image_id = line.strip()
                train_x.append(os.path.join(image_dir, image_id + '.jpg'))
                obj_pos = object_positions(os.path.join(annotation_dir, image_id + '.xml'), name2id)
                train_y.append(obj_pos)

    with open(os.path.join(root_dir, 'ImageSets/test.txt')) as fp:
        for line in fp:
            image_id = line.strip()
            val_x.append(os.path.join(image_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_dir, image_id + '.xml'), name2id)
            val_y.append(obj_pos)

    return train_x, train_y, val_x, val_y
