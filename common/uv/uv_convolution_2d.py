import math
import numpy as np
import cupy
import chainer
from chainer import cuda
from chainer.functions.connection import convolution_2d
from chainer.utils import argument
from chainer import variable
from chainer import initializers
from chainer import link
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.functions.array.broadcast import broadcast_to
import chainer.functions as F

'''
W = UdV
Mode 1: ! divided by max SN
Mode 2: ! truncate by 1 SC
Mode 3: ! penalize sum of all log max spectral
Mode 4: ! penalize E(-log(q(x))) q(x)~|N(0,0.2)| & divided by max
Mode 5: ! penalize E(-log(q(x))) q(x)~|N(0,0.2)| & truncate by 1 (worked)
Mode 6: ! penalize dlogd & divided by max
Mode 7: penalize expd & divided by max
Mode 8: penalize logd & divided by max
'''

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class UVConvolution2D(link.Link):

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, mode=None):

        if mode == None:
            raise NotImplementedError()

        super(UVConvolution2D, self).__init__()

        self.mode = mode

        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.out_channels = out_channels

        with self.init_scope():
            U_initializer = initializers._get_initializer(initialW)
            V_initializer = initializers._get_initializer(initialW)
            D_initializer = initializers._get_initializer(chainer.initializers.One())
            self.U = variable.Parameter(U_initializer)
            self.V = variable.Parameter(V_initializer)
            self.D = variable.Parameter(D_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_channels)

    def update_sigma(self):
        if self.mode in (1,4,6,7,8):
            self.D.data = self.D.data/F.absolute(self.D).data.max()
        elif self.mode in (2,5):
            self.D.data = F.clip(self.D,-1,1).data

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        self.update_sigma()
        _D = F.broadcast_to(self.D, (self.out_channels, self.D.size))
        _W = F.matmul(self.U.T * _D, self.V)
        return _W.reshape(self.W_shape)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        self.W_shape = (self.out_channels, in_channels, kh, kw)
        self.total_in_dim = in_channels*kh*kw

        if self.out_channels  <= self.total_in_dim:
            self.U.initialize((self.out_channels, self.out_channels))
            self.D.initialize((self.out_channels))
            self.V.initialize((self.out_channels, self.total_in_dim))
        else:
            self.U.initialize((self.total_in_dim, self.out_channels))
            self.D.initialize((self.total_in_dim))
            self.V.initialize((self.total_in_dim, self.total_in_dim))

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.U.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(
            x, self.W_bar, self.b, self.stride, self.pad)

    def loss_orth(self):
        penalty = 0

        W = self.U
        Wt = W.T
        WWt = F.matmul(W, Wt)
        I = cupy.identity(WWt.shape[0])
        penalty = penalty+ F.sum((WWt-I)**2)

        W = self.V
        Wt = W.T
        WWt = F.matmul(W, Wt)
        I = cupy.identity(WWt.shape[0])
        penalty = penalty+ F.sum((WWt-I)**2)

        spectral_penalty = 0
        if self.mode in (3,):
            spectral_penalty += F.log(F.max(F.absolute(self.D)))
        elif self.mode in (4,5):
            if(self.D.size > 1):
                sd2 = 0.1**2
                _d = self.D[cupy.argsort(self.D.data)]
                spectral_penalty += F.mean( (1 - _d[:-1])**2/sd2-F.log((_d[1:] - _d[:-1])+1e-7) ) * 0.05
        elif self.mode == 6:
            spectral_penalty += F.mean(self.D*F.log(self.D))
        elif self.mode == 7:
            spectral_penalty += F.mean(torch.exp(self.D))
        elif self.mode == 8:
            spectral_penalty += -F.mean(torch.log(self.D))

        return penalty + spectral_penalty*0.1

    def showOrthInfo(self):
        _D = F.broadcast_to(self.D, (self.out_channels, self.D.size))
        _W = F.matmul(self.U.T * _D, self.V)
        _, s, _ = cupy.linalg.svd(_W.data)
        return s
