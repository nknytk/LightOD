# coding: utf-8

import os
import sys
import chainer
from chainer import functions as F
import numpy

sys.path.append(os.path.dirname(__file__))
from softmax_cross_entropy import softmax_cross_entropy


class LossCalculator:
    def __init__(self, n_classes, class_weights=None, weight_pos=1):
        self.xp = numpy
        if class_weights:
            self.class_weights = self.xp.array(class_weights, dtype=self.xp.float32)
        else:
            self.class_weights = self.xp.ones(n_classes, dtype=self.xp.float32)
        self.weight_pos = weight_pos
        self.n_classes = n_classes

    def to_gpu(self):
        import cupy
        self.xp = cupy
        self.class_weights_cpu = self.class_weights
        self.class_weights = chainer.cuda.to_gpu(self.class_weights_cpu)

    def to_cpu(self):
        self.xp = numpy
        self.class_weights = self.class_weights_cpu

    def loss(self, pred, actual):
        batch_size, n_boxes, _ = actual.shape
        actual_obj_ids = self.xp.array(actual[:,:,4], dtype=self.xp.int32).reshape(batch_size * n_boxes)
        predicted_objs = pred[:,:,4:].reshape(batch_size * n_boxes, self.n_classes)
        class_weights = actual[:,:,5:]
        cl_loss = softmax_cross_entropy(predicted_objs, actual_obj_ids, class_weight=class_weights)
        #cl_loss = focal_loss(predicted_objs, actual_obj_ids, alpha=0.5, gamma=2.0, class_weight=class_weights, xp=self.xp)
        cl_acc = F.accuracy(predicted_objs, actual_obj_ids)

        obj_idx = self.xp.where(actual[:,:,4] > 0)
        if obj_idx[0].size > 0:
            # 教師データ側にオブジェクトが存在するgridのみ、bboxの位置を評価してlossに含める
            pred_centers = pred[obj_idx][:,:2]
            actual_centers = actual[obj_idx][:,:2]
            pred_sizes = pred[obj_idx][:,2:4]
            actual_sizes = actual[obj_idx][:,2:4]
            pos_loss = F.mean_squared_error(pred_centers, actual_centers) + 7 * F.mean_squared_error(pred_sizes, actual_sizes)

        else:
            pos_loss = 0

        loss = cl_loss + self.weight_pos * pos_loss
        result = {
            'loss': loss,
            'pos_loss': pos_loss,  # loss of position
            'cl_loss': cl_loss,  # loss of classification
            'cl_acc': cl_acc  # accuracy of classification
        }
        return result


def focal_loss(predictions, actual_obj_ids, gamma=2.0, alpha=0.25, class_weight=None, xp=numpy):
    pred_probas = F.softmax(predictions)
    actual_probas = xp.eye(predictions.shape[-1])[actual_obj_ids]

    pt_positive = actual_probas * pred_probas
    pt_negative = (1. - actual_probas) * (1. - pred_probas)
    pt = pt_positive + pt_negative

    at_positive = actual_probas * alpha
    at_negative = (1. - actual_probas) * (1. - alpha)
    at = at_positive + at_negative

    fl = -at * (1. - pt)**gamma * F.log(pt)
    if class_weight is not None:
        #print(fl.shape, class_weight.shape)
        weights = xp.array(class_weight.reshape(fl.shape))
        fl = weights * fl
    return F.mean(fl)
