# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree
from .common import object_positions


name2id = {
    "aeroplane": 1,
    "bicycle": 1,
    "bird": 2,
    "boat": 1,
    "bottle": 0,
    "bus": 1,
    "car": 1,
    "cat": 2,
    "chair": 0,
    "cow": 2,
    "diningtable": 0,
    "dog": 2,
    "horse": 2,
    "motorbike": 1,
    "person": 3,
    "pottedplant": 0,
    "sheep": 2,
    "sofa": 0,
    "train": 1,
    "tvmonitor": 0
}
id2name = {
    0: "no_obj",
    1: "vehicle",
    2: "animal",
    3: "person"
}


def load(root_dir):
    """
    Pascal Voc Devkitから訓練・評価データを取得する。
    2007のtrainvalと2012のtrainvalを訓練データ、2007のtestデータを評価データとする。
    """
    train_x, train_y, val_x, val_y = [], [], [], []
    for year in (2007, 2012):
        image_dir = os.path.join(root_dir, 'VOC{}/JPEGImages'.format(year))
        annotation_dir = os.path.join(root_dir, 'VOC{}/Annotations'.format(year))
        with open(os.path.join(root_dir, 'VOC{}/ImageSets/Main/person_trainval.txt'. format(year))) as fp:
            for line in fp:
                image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
                train_x.append(os.path.join(image_dir, image_id + '.jpg'))
                obj_pos = object_positions(os.path.join(annotation_dir, image_id + '.xml'), name2id)
                train_y.append(obj_pos)
    with open(os.path.join(root_dir, 'VOC2007_test/ImageSets/Main/person_test.txt')) as fp:
        image_dir = os.path.join(root_dir, 'VOC2007_test/JPEGImages')
        annotation_dir = os.path.join(root_dir, 'VOC2007_test/Annotations')
        for line in fp:
            image_id, is_obj = re.sub('\s+', ' ', line.strip()).split(' ')
            val_x.append(os.path.join(image_dir, image_id + '.jpg'))
            obj_pos = object_positions(os.path.join(annotation_dir, image_id + '.xml'), name2id)
            val_y.append(obj_pos)

    return train_x, train_y, val_x, val_y
