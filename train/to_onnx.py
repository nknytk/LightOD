# coding: utf-8

import json
import os
import sys
import numpy
import chainer
import onnx
from onnx import optimizer
import onnx_chainer

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(this_dir, '..'))
import detector

n_classes = {
    'urban_objects': 8,
    'pascal_voc3': 4,
    'pascal_voc': 21
}


def main():
    conf_file = sys.argv[1]
    with open(conf_file) as fp:
        conf = json.load(fp)

    results_dir = os.path.join(this_dir, 'results', conf['name'])
    best_files = search_best_files(results_dir)

    model_class = getattr(detector, conf['detector'])
    onnx_dir = os.path.join(this_dir, '../detector/trained')
    for training_menu in best_files:
        chainer_model = model_class(n_classes[training_menu], conf['n_base_units'])
        chainer.serializers.load_npz(best_files[training_menu]['file'], chainer_model)
        onnx_file_path = os.path.join(onnx_dir, '{}_{}.onnx'.format(conf['name'], training_menu))
        chainer_to_onnx(chainer_model, onnx_file_path)
        print('exported {} to {}'.format(best_files[training_menu]['file'], onnx_file_path))


def search_best_files(results_dir):
    best_files = {}
    for each_result in os.listdir(results_dir):
        _, training_menu = each_result.split('_', 1)

        training_logfile = os.path.join(results_dir, each_result, 'log')
        with open(training_logfile) as fp:
            training_log = json.load(fp)
        best_validation_loss = training_log[0]['validation/main/loss']
        for epoch in training_log:
            best_validation_loss = min(best_validation_loss, epoch['validation/main/loss'])

        if training_menu not in best_files or best_validation_loss < best_files[training_menu]['loss']:
            best_files[training_menu] = {'loss': best_validation_loss, 'file': os.path.join(results_dir, each_result, 'best_loss.npz')}

    return best_files


def chainer_to_onnx(chainer_model, onnx_export_path):
    dummy_input = numpy.zeros((1, 3, 224, 224), dtype=numpy.float32)
    onnx_chainer.export(chainer_model, dummy_input, filename=onnx_export_path)
    onnx_model = onnx.load(onnx_export_path)
    onnx_model_optimized = optimizer.optimize(onnx_model, ['fuse_bn_into_conv'])
    onnx.save(onnx_model_optimized, onnx_export_path)


if __name__ == '__main__':
    main()
