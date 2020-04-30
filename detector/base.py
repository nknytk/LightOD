# coding: utf-8

import chainer

class DetectorBase(chainer.Chain):

    img_size = 224
    n_grid = 7
    loss_calc = None

    def __call__(self, x, t=None):
        pred = self.predict(x)
        if t is None:
            return pred
        evaluated = self.loss_calc.loss(pred, t)
        chainer.report(evaluated, self)
        return evaluated['loss']

    def predict(self, x):
        """ You must implement this method """
        pass

    def to_gpu(self, *args, **kwargs):
        if self.loss_calc:
            self.loss_calc.to_gpu()
        return super().to_gpu(*args, **kwargs)

    def to_cpu(self, *args, **kwargs):
        if self.loss_calc:
            self.loss_calc.to_cpu()
        return super().to_gpu(*args, **kwargs)

    def to_intel64(self):
        return super().to_intel64()
