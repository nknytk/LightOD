# coding: utf-8

import os
import re
import sys
from xml.etree import ElementTree


def object_positions(xml_file_path, name2id):
    positions = []
    xml_root = ElementTree.parse(xml_file_path).getroot()
    for obj in xml_root.findall('./object'):
        bndbox = obj.find('bndbox')
        if not bndbox:
            continue
        if obj.find('difficult').text == '1':
            continue
        obj_name = obj.find('name').text

        positions.append([
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text),
            name2id[obj_name]
        ])

    return positions
