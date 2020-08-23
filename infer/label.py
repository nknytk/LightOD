# coding: utf-8

import numpy
import cv2
import grid


class DrawLabel:
    def __init__(self, detection_target, img_x, img_y, n_grid_x, n_grid_y, min_proba=0.5, suppress_iou_threshold=0.7):
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
        self.suppress_iou_threshold = suppress_iou_threshold


    def draw_rect(self, image, prediction):
        detected = [[] for i in range(len(self.labels))]

        for grid_index, vec in enumerate(prediction):
            obj_probas = softmax(vec[4:])
            obj_id = numpy.argmax(obj_probas)
            if obj_id == 0:
                continue
            proba = obj_probas[obj_id]
            if proba < self.min_proba:
                continue

            bbox = self.grid_retriever.restore_box(grid_index, *vec[:4])
            detected[obj_id].append([proba, True, bbox])

        for obj_id, detections in enumerate(detected):
            _, color = self.labels[obj_id]
            for i in range(len(detections)):
                for j in range(len(detections)):
                    if i == j or detections[i][1] == False or detections[j][1] == False:
                        continue
                    iou = self.iou(detections[i][2], detections[j][2])
                    if iou >= self.suppress_iou_threshold:
                        if detections[i][0] > detections[j][0]:
                            detections[j][0] = False
                        else:
                            detections[i][0] = False

            for proba, to_draw, bbox in detections:
                if to_draw:
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=2)


    def iou(self, box1, box2):
        # x_min, y_min, x_max, y_max = box
        if box1[2] <= box2[0] or box2[2] <= box1[0] or box1[3] <= box2[1] or box2[3] <= box1[1]:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        area_shared = (min(box1[2], box2[2]) - max(box1[0], box2[0])) * (min(box1[3], box2[3]) - max(box1[1], box2[1]))
        iou = area_shared / (area1 + area2 - area_shared)
        return iou


def softmax(x):
    c = numpy.max(x)
    exp_x = numpy.exp(x - c)
    sum_exp_x = numpy.sum(exp_x)
    return exp_x / sum_exp_x
