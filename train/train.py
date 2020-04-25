# coding: utf-8

import json
import math
import multiprocessing
import os
import sys
import numpy
import chainer
from chainer.training import StandardUpdater, Trainer, extensions, triggers
from chainer.iterators import SerialIterator, MultiprocessIterator
from chainer.optimizers import Adam

this_dir = os.path.dirname(__file__)
sys.path.append(this_dir)
sys.path.append(os.path.join(this_dir, '..'))
from lib.finetune_util import copy_params
from lib.dataset import YoloDataset
from lib.batchsize import AdaptiveBatchsizeIncrement
from lib.loss import LossCalculator
from dataload import pascal_voc, pascal_voc3, urban_objects
from detector.simpleconv import SimpleConvYOLO
from detector.mobile_yolo import MobileYOLO


def main():
    conf_file = sys.argv[1]
    with open(conf_file) as fp:
        conf = json.load(fp)

    data_cache = {}
    pascal_voc_data = pascal_voc3.load(conf['pascal_voc_data_root'])
    urban_objects_data = urban_objects.load(conf['urban_objects_data_root'])

    model_class = SimpleConvYOLO if conf['detector'] == 'SimpleConvYOLO' else MobileYOLO
    trained_model = None

    for i, data_type in enumerate(conf['training_menu']):
        if data_type not in data_cache:
            if data_type == 'pascal_voc':
                data_cache[data_type] = pascal_voc.load(conf['pascal_voc_data_root'])
            elif data_type == 'pascal_voc3':
                data_cache[data_type] = pascal_voc3.load(conf['pascal_voc_data_root'])
            elif data_type == 'urban_objects':
                data_cache[data_type] = urban_objects.load(conf['urban_objects_data_root'])
            else:
                raise RuntimeError('Unknown data type {}'.format(data_type))

        print('training menu {} {}'.format(i, data_type))
        result_dir = os.path.join(this_dir, 'results', conf['name'], '{}_{}'.format(i, data_type))
        trained_model = train(
            model_class,
            conf['n_base_units'],
            trained_model,
            conf[data_type + '_noobj_weight'],
            data_cache[data_type],
            result_dir,
            conf['initial_batch_size'],
            conf['max_batch_size'],
            conf['max_epoch']
        )


def train(model_class, n_base_units, trained_model, no_obj_weight, data, result_dir, initial_batch_size=10, max_batch_size=1000, max_epoch=100):
    train_x, train_y, val_x, val_y = data

    max_class_id = 0
    for objs in val_y:
        for obj in objs:
            max_class_id = max(max_class_id, obj[4])
    n_classes = max_class_id + 1

    class_weights = [1.0 for i in range(n_classes)]
    class_weights[0] = no_obj_weight
    train_dataset = YoloDataset(
        train_x,
        train_y,
        target_size=model_class.img_size,
        n_grid=model_class.n_grid,
        augment=True,
        class_weights=class_weights
    )
    test_dataset = YoloDataset(
        val_x,
        val_y,
        target_size=model_class.img_size,
        n_grid=model_class.n_grid,
        augment=False,
        class_weights=class_weights
    )

    model = model_class(n_classes, n_base_units)
    model.loss_calc = LossCalculator(n_classes, class_weights=class_weights)

    last_result_file = os.path.join(result_dir, 'best_loss.npz')
    if os.path.exists(last_result_file):
        try:
            chainer.serializers.load_npz(last_result_file, model)
            print('this training has done. resuse the result')
            return model
        except:
            pass

    if trained_model:
        print('copy params from trained model')
        copy_params(trained_model, model)

    optimizer = Adam()
    optimizer.setup(model)

    n_physical_cpu = int(math.ceil(multiprocessing.cpu_count() / 2))

    train_iter = MultiprocessIterator(train_dataset, batch_size=initial_batch_size, n_prefetch=n_physical_cpu, n_processes=n_physical_cpu)
    test_iter = MultiprocessIterator(test_dataset, batch_size=initial_batch_size, shuffle=False, repeat=False, n_prefetch=n_physical_cpu, n_processes=n_physical_cpu)
    updater = StandardUpdater(train_iter, optimizer, device=0)
    stopper = triggers.EarlyStoppingTrigger(
        check_trigger=(1, 'epoch'),
        monitor="validation/main/loss",
        patients=10,
        mode="min",
        max_trigger=(max_epoch, "epoch")
    )
    trainer = Trainer(updater, stopper, out=result_dir)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.Evaluator(test_iter, model, device=0))
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss', 'validation/main/loss',
        'main/cl_loss', 'validation/main/cl_loss',
        'main/cl_acc', 'validation/main/cl_acc',
        'main/pos_loss', 'validation/main/pos_loss',
    ]))
    trainer.extend(extensions.snapshot_object(model, 'best_loss.npz'), trigger=triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.snapshot_object(model, 'best_classification.npz'), trigger=triggers.MaxValueTrigger('validation/main/cl_acc'))
    trainer.extend(extensions.snapshot_object(model, 'best_position.npz'), trigger=triggers.MinValueTrigger('validation/main/pos_loss'))
    trainer.extend(extensions.snapshot_object(model, 'model_last.npz'), trigger=(1, 'epoch'))
    trainer.extend(AdaptiveBatchsizeIncrement(maxsize=max_batch_size), trigger=(1, 'epoch'))

    trainer.run()

    chainer.serializers.load_npz(os.path.join(result_dir, 'best_loss.npz'), model)
    return model


if __name__ == '__main__':
    main()
