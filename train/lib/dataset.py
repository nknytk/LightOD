# coding: utf-8

import os
import sys
from random import random, choice
import numpy
from PIL import Image, ImageOps
from chainer.dataset.dataset_mixin import DatasetMixin

sys.path.append(os.path.dirname(__file__))
from image_util import horizontal_flip, resize
from grid import GridRetriever


class YoloDataset(DatasetMixin):
    def __init__(self, x, y, target_size=224, n_grid=7, augment=False, class_weights=None):
        self._datasets = (x, y)
        self._length = len(x)

        self.rescore_neighbour = True
        self.target_size = target_size
        self.n_grid = n_grid
        self.augment = augment
        self.grid_retriever = GridRetriever(target_size, target_size, n_grid, n_grid)

        max_class_id = 0
        for objs in y:
            for obj in objs:
                max_class_id = max(max_class_id, obj[4])
        self.n_classes = max_class_id + 1
        self.class_weights = class_weights if class_weights else [1.0 for i in range(self.n_classes)]

    def __len__(self):
        return self._length

    def get_example(self, i):
        image_path = self._datasets[0][i]
        objs = self._datasets[1][i]
        np_img, bboxes = self.convert(image_path, objs)
        return np_img, bboxes

    def convert(self, img_path, objs):
        """
        Input:
          img_path: 画像ファイルのパス
          objs: オブジェクトbounding boxの情報を含んだ配列。各オブジェクトは(xmin, ymin, xmax, ymax, class_id)
        Output:
          (3, target_size, target_size)の画像, bboxの配列
          bboxの配列は長さ n_grid**2 で、各要素は
          (中心点のgrid内x, 中心点のgrid内y, 幅, 高さ, class_id)
        """
        orig_img = Image.open(img_path)
        w, h = orig_img.size

        if self.augment:
            if random() > 0.5:
                t_img, t_boxes = orig_img, objs
            else:
                t_img, t_boxes = horizontal_flip(orig_img, objs)
            max_noise = min(w, h) * 0.2
            w_noise = int(max_noise * (random() - 0.5))
            h_noise = int(max_noise * (random() - 0.5))

            t_img, t_boxes = resize(t_img, t_boxes, self.target_size, w_noise, h_noise, min_bbox_pixel=3)
            np_img = self.color_noise(numpy.asarray(t_img.convert('RGB'), dtype=numpy.float32)).transpose(2, 0, 1)
        else:
            t_img, t_boxes = resize(orig_img, objs, self.target_size, min_bbox_pixel=3)
            np_img = numpy.asarray(t_img.convert('RGB'), dtype=numpy.float32).transpose(2, 0, 1)

        np_boxes = numpy.zeros((self.n_grid**2, 5 + self.n_classes), dtype=numpy.float32)
        np_boxes[:,5:] = self.class_weights
        for box in t_boxes:
            try:
                w_min, h_min, w_max, h_max, cls_id = box
                grid_idx, x_grid_index, y_grid_index, grid_center_x, grid_center_y, relative_w, relative_h = self.grid_retriever.grid_position(w_min, h_min, w_max, h_max)
                np_boxes[grid_idx][0] = grid_center_x
                np_boxes[grid_idx][1] = grid_center_y
                np_boxes[grid_idx][2] = relative_w
                np_boxes[grid_idx][3] = relative_h
                np_boxes[grid_idx][4] = cls_id

                if not self.rescore_neighbour:
                    continue

                neighbours = []
                if grid_center_x < 0.5 and x_grid_index > 0:
                    neighbours.append((grid_idx - 1, grid_center_x / 0.5))
                    if grid_center_y < 0.5 and y_grid_index > 0:
                        _distance = (grid_center_x**2 + grid_center_y**2)**0.5
                        neighbours.append((grid_idx - self.n_grid - 1 , min(_distance / 0.5, 1.0)))
                    elif grid_center_y > 0.5 and y_grid_index < self.n_grid - 1:
                        _distance = (grid_center_x**2 + (1.0 - grid_center_y)**2)**0.5
                        neighbours.append((grid_idx + self.n_grid - 1, min(_distance / 0.5, 1.0)))
                elif grid_center_x > 0.5 and x_grid_index < self.n_grid - 1:
                    neighbours.append((grid_idx + 1, (1.0 - grid_center_x) / 0.5))
                    if grid_center_y < 0.5 and y_grid_index > 0:
                        _distance = ((1.0 - grid_center_x)**2 + grid_center_y**2)**0.5
                        neighbours.append((grid_idx - self.n_grid + 1 , min(_distance / 0.5, 1.0)))
                    elif grid_center_y > 0.5 and y_grid_index < self.n_grid - 1:
                        _distance = ((1.0 - grid_center_x)**2 + (1.0 - grid_center_y)**2)**0.5
                        neighbours.append((grid_idx + self.n_grid + 1, min(_distance / 0.5, 1.0)))

                if grid_center_y < 0.5 and y_grid_index > 0:
                    neighbours.append((grid_idx - self.n_grid, grid_center_y / 0.5))
                elif grid_center_y > 0.5 and y_grid_index < self.n_grid - 1:
                    neighbours.append((grid_idx + self.n_grid, (1.0 - grid_center_y) / 0.5))

                for grid_i, w in neighbours:
                    np_boxes[grid_i, 5 + cls_id] = min(np_boxes[grid_i, 5 + cls_id], w)

            except Exception as e:
                print(e)
                print(box)
                print(self.grid_retriever.grid_position(w_min, h_min, w_max, h_max))

        return np_img, np_boxes

    def color_noise(self, np_img):
        noise_range = numpy.random.randint(0, 20)
        if noise_range > 0:
            np_img += numpy.random.randint(-noise_range, noise_range, np_img.shape)
        np_img[:,:,0] *= (1 + (numpy.random.rand() - 0.5) * 0.2)
        np_img[:,:,1] *= (1 + (numpy.random.rand() - 0.5) * 0.2)
        np_img[:,:,2] *= (1 + (numpy.random.rand() - 0.5) * 0.2)
        _min = min(numpy.min(np_img), 0)
        _max = max(numpy.max(np_img), 255)
        if _min != 0 or _max != 255:
            np_img = (-_min + np_img) * 255 / (_max - _min)
        return np_img
