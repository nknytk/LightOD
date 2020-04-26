# coding: utf-8

import numpy
import cv2
import grid


class DrawLabel:
    def __init__(self, detection_target, img_x, img_y, n_grid_x, n_grid_y, min_proba=0.5):
        if detection_target == 'urban_objects':
            self.labels = [
                #('label_name', (r, g, b))
                ('no_obj', (255, 255, 255)),     # no_obj is not drawn
                ('bicycle', (120, 100, 39)),      # darkgreen 
                ('bus', (139, 69, 39)),           # saddlebrown
                ('car', (139, 0, 0)),             # darkred
                ('motorbike', (120, 100, 100)),   # lime
                ('person', (0, 0, 255)),          # blue
                ('trafficsignal', (75, 0, 130)),  # indigo
                ('trafficlight', (148, 0, 211))   # darkviolet 
            ]
        elif detection_target == 'pascal_voc3':
            self.labels = [
                ('no_obj', (255, 255, 255)),  # no_obj is not drawn
                ('vehicle', (255, 0, 0)),     # red
                ('animal', (0, 255, 0)),      # green
                ('person', (0, 0, 255))       # blue
            ]
        self.grid_retriever = grid.GridRetriever(img_x, img_y, n_grid_x, n_grid_y)
        self.min_proba = min_proba


    def draw_rect(self, image, prediction):
        for grid_index, vec in enumerate(prediction):
            obj_probas = softmax(vec[4:])
            obj_id = numpy.argmax(obj_probas)
            if obj_id == 0:
                continue
            proba = obj_probas[obj_id]
            if proba < self.min_proba:
                continue
            x_min, y_min, x_max, y_max = self.grid_retriever.restore_box(grid_index, *vec[:4])

            _, color = self.labels[obj_id]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)


def softmax(x):
    c = numpy.max(x)
    exp_x = numpy.exp(x - c)
    sum_exp_x = numpy.sum(exp_x)
    return exp_x / sum_exp_x

