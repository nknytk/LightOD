# coding: utf-8
# Original MobileNet implementation by peisuke
# https://github.com/peisuke/DeepLearningSpeedComparison/blob/master/chainer/mobilenet/predict.py
import chainer
import chainer.functions as F
import chainer.links as L
from .base import DetectorBase


class ConvBN(chainer.Chain):
    def __init__(self, inp, oup, stride, activation=F.relu):
        super(ConvBN, self).__init__()
        with self.init_scope():
            self.conv=L.Convolution2D(inp, oup, 3, stride=stride, pad=1, nobias=True)
            self.bn=L.BatchNormalization(oup)
        self.activation = activation

    def __call__(self, x):
        h = self.activation(self.bn(self.conv(x)))
        return h


class ConvDW(chainer.Chain):
    def __init__(self, inp, oup, stride, restype=None, activation=F.relu):
        super(ConvDW, self).__init__()
        self.restype = restype
        self.stride = stride
        self.activation = activation
        if self.restype == 'concat':
            self.oup = oup - inp
        else:
            self.oup = oup
        with self.init_scope():
            self.conv_dw=L.DepthwiseConvolution2D(inp, 1, 3, stride=stride, pad=1, nobias=True)
            self.bn_dw=L.BatchNormalization(inp)
            self.conv_sep=L.Convolution2D(inp, self.oup, 1, stride=1, pad=0, nobias=True)
            self.bn_sep=L.BatchNormalization(self.oup)

    def __call__(self, x):
        h = self.activation(self.bn_dw(self.conv_dw(x)))
        h = self.bn_sep(self.conv_sep(h))
        if self.restype == 'add':
            h += x
        elif self.restype == 'concat':
            if self.stride > 1:
                x = F.max_pooling_2d(x, ksize=3, stride=self.stride)
            h = F.concat((h, x), axis=1)
        return self.activation(h)


class MobileYOLO(DetectorBase):

    img_size = 224
    n_grid=7

    def __init__(self, n_classes=1, n_base_units=32):
        activation = F.leaky_relu
        super().__init__()
        self.n_classes = n_classes
        self.loss_calc = None
        with self.init_scope():
            self.conv_bn = ConvBN(3, n_base_units, 2, activation=activation)
            self.conv_ds_2 = ConvDW(n_base_units, n_base_units * 2, 1, restype=None, activation=activation)
            self.conv_ds_3 = ConvDW(n_base_units * 2, n_base_units * 4, 2, restype=None, activation=activation)
            self.conv_ds_4 = ConvDW(n_base_units * 4, n_base_units * 4, 1, restype=None, activation=activation)
            self.conv_ds_5 = ConvDW(n_base_units * 4, n_base_units * 8, 2, restype=None, activation=activation)
            self.conv_ds_6 = ConvDW(n_base_units * 8, n_base_units * 8, 1, restype=None, activation=activation)
            self.conv_ds_7 = ConvDW(n_base_units * 8, n_base_units *16, 2, restype=None, activation=activation)

            self.conv_ds_8 = ConvDW(n_base_units *16, n_base_units *16, 1, restype=None, activation=activation)
            self.conv_ds_9 = ConvDW(n_base_units *16, n_base_units *16, 1, restype=None, activation=activation)
            self.conv_ds_10 = ConvDW(n_base_units *16, n_base_units *16, 1, restype=None, activation=activation)
            self.conv_ds_11 = ConvDW(n_base_units *16, n_base_units *16, 1, restype=None, activation=activation)
            self.conv_ds_12 = ConvDW(n_base_units *16, n_base_units *16, 1, restype=None, activation=activation)

            self.conv_ds_13 = ConvDW(n_base_units *16, n_base_units *32, 2, restype=None, activation=activation)
            self.conv_ds_14 = ConvDW(n_base_units *32, 4 + n_classes, 1, restype=None, activation=activation)

    def predict(self, x):
        h = self.conv_bn(x)
        h = self.conv_ds_2(h)
        h = self.conv_ds_3(h)
        h = self.conv_ds_4(h)
        h = self.conv_ds_5(h)
        h = self.conv_ds_6(h)
        h = self.conv_ds_7(h)
        h = self.conv_ds_8(h)
        h = self.conv_ds_9(h)
        h = self.conv_ds_10(h)
        h = self.conv_ds_11(h)
        h = self.conv_ds_12(h)
        h = self.conv_ds_13(h)
        h = self.conv_ds_14(h)

        # (batch_size, 4 + n_classes, 7, 7) -> (bach_size, 7, 7, 4 + n_classes)
        h = F.transpose(h, (0, 2, 3, 1))
        # (batch_size, 7, 7, 4 + n_classes) -> (batch_size, 49, 4 + n_classes)
        batch_size = int(h.size / (self.n_grid**2 * (4 + self.n_classes)))
        r = F.reshape(h, (batch_size, self.n_grid**2, 4 + self.n_classes))
        return r
