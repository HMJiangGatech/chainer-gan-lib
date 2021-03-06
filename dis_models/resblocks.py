import math
import chainer
from chainer import functions as F
import chainer.links as L

from common.sn.sn_linear import SNLinear
from common.sn.sn_convolution_2d import SNConvolution2D
from common.orth.orth_linear import ORTHLinear
from common.orth.orth_convolution_2d import ORTHConvolution2D
from common.uv.uv_linear import UVLinear
from common.uv.uv_convolution_2d import UVConvolution2D


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)


class SNBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(SNBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = SNConvolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = SNConvolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedSNBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedSNBlock, self).__init__()
        initializer = chainer.initializers.GlorotUniform(math.sqrt(2))
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = SNConvolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = SNConvolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = SNConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)



class ORTHBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(ORTHBlock, self).__init__()
        initializer = chainer.initializers.Orthogonal(1)
        initializer_sc = chainer.initializers.Orthogonal(1)
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = ORTHConvolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = ORTHConvolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            if self.learnable_sc:
                self.c_sc = ORTHConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def loss_orth(self):
        loss =  self.c1.loss_orth() + self.c2.loss_orth()
        if self.learnable_sc:
            loss += self.c_sc.loss_orth()
        return loss

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedORTHBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu):
        super(OptimizedORTHBlock, self).__init__()
        initializer = chainer.initializers.Orthogonal(1)
        initializer_sc = chainer.initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            # 128 = 27*5+3
            self.c10 = ORTHConvolution2D(in_channels, out_channels//5+1, ksize=ksize, pad=pad, initialW=initializer)
            self.c11 = ORTHConvolution2D(in_channels, out_channels//5+1, ksize=ksize, pad=pad, initialW=initializer)
            self.c12 = ORTHConvolution2D(in_channels, out_channels//5+1, ksize=ksize, pad=pad, initialW=initializer)
            self.c13 = ORTHConvolution2D(in_channels, out_channels//5, ksize=ksize, pad=pad, initialW=initializer)
            self.c14 = ORTHConvolution2D(in_channels, out_channels//5, ksize=ksize, pad=pad, initialW=initializer)
            self.c2 = ORTHConvolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer)
            self.c_sc = L.Convolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc)

    def loss_orth(self):
        loss =  self.c10.loss_orth() + self.c11.loss_orth() + self.c12.loss_orth() + self.c13.loss_orth() + self.c14.loss_orth() + self.c2.loss_orth()
        return loss

    def residual(self, x):
        h = chainer.functions.hstack([self.c10(x),self.c11(x),self.c12(x),self.c13(x),self.c14(x)])
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class UVBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, mode=None):
        super(UVBlock, self).__init__()
        initializer = chainer.initializers.Orthogonal(1)
        initializer_sc = chainer.initializers.Orthogonal(1)
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.c1 = UVConvolution2D(in_channels, hidden_channels, ksize=ksize, pad=pad, initialW=initializer, mode=mode)
            self.c2 = UVConvolution2D(hidden_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer, mode=mode)
            if self.learnable_sc:
                self.c_sc = UVConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc, mode=mode)

    def loss_orth(self):
        loss =  self.c1.loss_orth() + self.c2.loss_orth()
        if self.learnable_sc:
            loss += self.c_sc.loss_orth()
        return loss

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedUVBlock(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu, mode=None):
        super(OptimizedUVBlock, self).__init__()
        initializer = chainer.initializers.Orthogonal(1)
        initializer_sc = w = chainer.initializers.Orthogonal(1)
        self.activation = activation
        with self.init_scope():
            # 128 = 27*5+3
            self.c1 = UVConvolution2D(in_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer, mode=mode)
            self.c2 = UVConvolution2D(out_channels, out_channels, ksize=ksize, pad=pad, initialW=initializer, mode=mode)
            self.c_sc = UVConvolution2D(in_channels, out_channels, ksize=1, pad=0, initialW=initializer_sc, mode=mode)

    def loss_orth(self):
        loss =  self.c1.loss_orth() + self.c2.loss_orth() + self.c_sc.loss_orth()
        return loss

    def residual(self, x):
        h = self.c1(x)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)
