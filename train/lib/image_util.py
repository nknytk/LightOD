# coding: utf-8

from random import random, choice
import numpy
from PIL import Image, ImageOps


def horizontal_flip(img, boxes):
    """ 画像とbounding boxの座標の両方を左右反転する """
    flipped_img = ImageOps.mirror(img)
    X, Y = img.size
    flipped_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax, cls_id = box
        flipped_xmin = X - xmax
        flipped_xmax = X - xmin
        flipped_boxes.append((flipped_xmin, ymin, flipped_xmax, ymax, cls_id))
    return flipped_img, flipped_boxes


def resize(img, boxes, size, w_noise=0, h_noise=0, min_bbox_pixel=3):
    w, h = img.size
    w2 = w - abs(w_noise)
    h2 = h - abs(h_noise)
    w_ratio = size / w2
    h_ratio = size / h2

    xmin = max(0, w_noise)
    ymin = max(0, h_noise)
    xmax = min(w, w + w_noise)
    ymax = min(h, h + w_noise)

    cropped_img = img.crop((xmin, ymin, xmax, ymax)).resize((size, size), Image.LANCZOS)
    offset_boxes = []
    for box in boxes:
        bxmin = max(0, (box[0] - xmin) * w_ratio)
        bymin = max(0, (box[1] - ymin) * h_ratio)
        bxmax = min(size, (box[2] - xmin) * w_ratio)
        bymax = min(size, (box[3] - ymin) * h_ratio)
        if bxmin < size and bymin < size and bxmax > 0 and bymax > 0 and (bxmax - bxmin) >= min_bbox_pixel and (bymax - bymin) >= min_bbox_pixel:
            offset_boxes.append((bxmin, bymin, bxmax, bymax, box[4]))

    return cropped_img, offset_boxes
