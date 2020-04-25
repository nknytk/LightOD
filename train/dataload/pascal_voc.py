# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree
from .common import object_positions


name2id = {
    "noobj": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}
id2name = {v: k for k, v in name2id.items()}


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
