import math
import numpy as np
import cupy
import chainer
from chainer import cuda
from chainer.functions.connection import linear
from chainer import variable
from chainer import initializers
from chainer import link
from chainer.links.connection.linear import Linear
from chainer.functions.array.broadcast import broadcast_to
import chainer.functions as F


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x

class UVLinear(link.Link):
    def __init__(self, in_size, out_size, use_gamma=False, nobias=False,
                 initialW=None, initial_bias=None, mode=None):
        if mode == None:
            raise NotImplementedError()
        super(UVLinear, self).__init__()
        self.mode = mode
        self.in_size = in_size
        self.out_size = out_size

        with self.init_scope():
            U_initializer = initializers._get_initializer(initialW)
            V_initializer = initializers._get_initializer(initialW)
            D_initializer = initializers._get_initializer(chainer.initializers.One())
            self.U = variable.Parameter(U_initializer)
            self.V = variable.Parameter(V_initializer)
            self.D = variable.Parameter(D_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def update_sigma(self):
        if self.mode in (1,4,6,7,8):
            self.D.data = self.D.data/F.absolute(self.D).data.max()
        elif self.mode in (2,5):
            self.D.data = F.clip(self.D,-1.,1.).data

    @property
    def W_bar(self):
        """
        Spectral Normalized Weight
        """
        self.update_sigma()
        _D = F.broadcast_to(self.D, (self.out_size, self.D.size))
        _W = F.matmul(self.U * _D, self.V)
        return _W

    def _initialize_params(self, in_size):

        if self.out_size  <= self.in_size:
            self.U.initialize((self.out_size, self.out_size))
            self.D.initialize((self.out_size))
            self.V.initialize((self.out_size, self.in_size))
        else:
            self.U.initialize((self.out_size, self.in_size))
            self.D.initialize((self.in_size))
            self.V.initialize((self.in_size, self.in_size))

    def log_d_max(self):
        return F.log(F.max(F.absolute(self.D)))

    def loss_orth(self):
        penalty = 0

        W = self.U.T
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
        if self.mode in (4,5):
            if(self.D.size > 1):
                sd2 = 0.1**2
                _d = self.D[cupy.argsort(self.D.data)]
                spectral_penalty += F.mean( (1 - _d[:-1])**2/sd2-F.log((_d[1:] - _d[:-1])+1e-7) ) * 0.05
        elif self.mode == 6:
            spectral_penalty += F.mean(self.D*F.log(self.D))
        elif self.mode == 7:
            spectral_penalty += F.mean(F.exp(self.D))
        elif self.mode == 8:
            spectral_penalty += -F.mean(F.log(self.D))

        return penalty + spectral_penalty*0.1

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.U.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)

    def showOrthInfo(self):
        _D = F.broadcast_to(self.D, (self.out_size, self.D.size))
        _W = F.matmul(self.U * _D, self.V)
        _, s, _ = cupy.linalg.svd(_W.data)
        print('Singular Value Summary: ')
        print('max :',s.max())
        print('mean:',s.mean())
        print('min :',s.min())
        print('var :',s.var())
        return s
