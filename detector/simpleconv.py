import chainer
from chainer import links as L
from chainer import functions as F
from .base import DetectorBase


class ConvLayer1(chainer.Chain):
    def __init__(self, in_channels, out_channels, stride=1, pad=0, nobias=True):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels, out_channels, ksize=3, stride=stride, pad=pad, nobias=nobias)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        return F.leaky_relu(self.bn(self.conv(x)))


class ConvLayer2(chainer.Chain):
    def __init__(self, in_channels, out_channels, stride=1, pad=0, nobias=True):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1, nobias=nobias)
            self.bn1 = L.BatchNormalization(out_channels)
            self.conv2 = L.Convolution2D(out_channels, out_channels, ksize=3, stride=stride, pad=pad, nobias=nobias)
            self.bn2 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        return F.leaky_relu(self.bn2(self.conv2(x)))


class SimpleConvYOLO(DetectorBase):

    img_size = 224
    n_grid = 7

    def __init__(self, n_classes, n_base_units=8):
        super().__init__()
        self.n_classes = n_classes
        self.loss_calc = None
        with self.init_scope():
            # 224 => 112
            self.cl1 = ConvLayer1(3, n_base_units, stride=2, pad=1)
            # 112 => 56
            self.cl2 = ConvLayer2(n_base_units, n_base_units*2, stride=2, pad=1)
            # 56 => 28
            self.cl3 = ConvLayer2(n_base_units*2, n_base_units*4, stride=2, pad=1)
            # 28 => 28
            self.cl4_1 = ConvLayer2(n_base_units*4, n_base_units*8, stride=1, pad=1)
            self.cl4_2 = ConvLayer2(n_base_units*8, n_base_units*8, stride=1, pad=1)
            # 28 => 14
            self.cl4_3 = ConvLayer2(n_base_units*8, n_base_units*8, stride=2, pad=1)
            # 14 => 7
            self.cl5_1 = ConvLayer2(n_base_units*8, n_base_units*16, stride=2, pad=1)
            self.cl5_2 = ConvLayer1(n_base_units*16, n_base_units*16, stride=1, pad=1)
            # 7
            self.cl6 = L.Convolution2D(n_base_units * 16, 4 + n_classes, ksize=1, stride=1, pad=0)

    def predict(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        x = self.cl3(x)
        x = self.cl4_1(x)
        x = self.cl4_2(x)
        x = self.cl4_3(x)
        x = self.cl5_1(x)
        x = self.cl5_2(x)
        x = self.cl6(x)

        # (batch_size, 4 + n_classes, 7, 7) -> (bach_size, 7, 7, 4 + n_classes)
        x = F.transpose(x, (0, 2, 3, 1))
        # (batch_size, 7, 7, 4 + n_classes) -> (batch_size, 49, 4 + n_classes)
        batch_size = int(x.size / (self.n_grid**2 * (4 + self.n_classes)))
        r = F.reshape(x, (batch_size, self.n_grid**2, 4 + self.n_classes))
        return r
