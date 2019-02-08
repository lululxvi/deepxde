# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math

import numpy as np
import tensorflow as tf
from SALib.sample import sobol_sequence
from sklearn import preprocessing

from nnlearn import config
from .fractional import Fractional, FractionalTime
from .utils import runifnone


class Data(object):
    """Training data
    """

    def __init__(self, target):
        self.target = target

    @abc.abstractmethod
    def train_next_batch(self, batch_size, *args, **kwargs):
        return np.array([])

    @abc.abstractmethod
    def test(self, n, *args, **kwargs):
        return np.array([])


class DataSet(Data):
    def __init__(self, fname_train, fname_test, col_x=None, col_y=None):
        super(DataSet, self).__init__('func')
        if col_x is None:
            col_x = (0,)
        if col_y is None:
            col_y = (-1,)

        train_data = np.loadtxt(fname_train)
        self.train_x, self.train_y = train_data[:, col_x], train_data[:, col_y]
        test_data = np.loadtxt(fname_test)
        self.test_x, self.test_y = test_data[:, col_x], test_data[:, col_y]

        self.scaler_x, self.scaler_y = None, None
        self._standardize()

    def _standardize(self):
        def standardize_one(X1, X2):
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
            X1 = scaler.fit_transform(X1)
            X2 = scaler.transform(X2)
            return scaler, X1, X2

        self.scaler_x, self.train_x, self.test_x = standardize_one(self.train_x, self.test_x)
        self.scaler_y, self.train_y, self.test_y = standardize_one(self.train_y, self.test_y)

    def inverse_transform_y(self, y):
        return self.scaler_y.inverse_transform(y)

    def train_next_batch(self, batch_size, *args, **kwargs):
        return self.train_x, self.train_y

    def test(self, n, *args, **kwargs):
        return self.test_x, self.test_y


class DataSet2(Data):
    def __init__(self, X_train, y_train, X_test, y_test):
        super(DataSet2, self).__init__('func')

        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.scaler_x = None
        self._standardize()

    def _standardize(self):
        def standardize_one(X1, X2):
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
            X1 = scaler.fit_transform(X1)
            X2 = scaler.transform(X2)
            return scaler, X1, X2

        self.scaler_x, self.train_x, self.test_x = standardize_one(self.train_x, self.test_x)

    def train_next_batch(self, batch_size, *args, **kwargs):
        return self.train_x, self.train_y

    def test(self, n, *args, **kwargs):
        return self.test_x, self.test_y


class DataFunc(Data):
    """Training data for function approximation
    """

    def __init__(self, func, geom, online=False):
        super(DataFunc, self).__init__('func')
        self.func = func
        self.geom = geom
        self.online = online

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def train_next_batch(self, batch_size, *args, **kwargs):
        if self.online:
            self.train_x = self.geom.random_points(batch_size, 'pseudo')
            self.train_y = self.func(self.train_x)
        elif self.train_x is None:
            # self.train_x = self.geom.random_points(batch_size, 'sobol')
            self.train_x = self.geom.uniform_points(batch_size, True)
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y


class DataMF(Data):
    """Training data for multi-fidelity function approximation
    """

    def __init__(self, flow, fhi, geom):
        super(DataMF, self).__init__('func')
        self.flow, self.fhi = flow, fhi
        self.geom = geom

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def train_next_batch(self, batch_size, *args, **kwargs):
        keeps = [0, 2, 5, 8, 10]
        x = self.geom.uniform_points(batch_size, True)
        self.train_x = np.empty((0, 1))
        self.train_y = np.empty((0, 2))
        for _ in range(10):
            ylow = self.flow(x)
            yhi = self.fhi(x)
            for i in range(batch_size):
                if i not in keeps:
                    yhi[i, 0] = ylow[i, 0] + 2*np.random.randn()
            self.train_x = np.vstack((self.train_x, x))
            self.train_y = np.vstack((self.train_y, np.hstack((ylow, yhi))))

        x = x[keeps]
        ylow = self.flow(x)
        yhi = self.fhi(x)
        for _ in range(500):
            self.train_x = np.vstack((self.train_x, x))
            self.train_y = np.vstack((self.train_y, np.hstack((ylow, yhi))))

        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        ylow = self.flow(self.test_x)
        yhi = self.fhi(self.test_x)
        self.test_y = np.hstack((ylow, yhi))
        return self.test_x, self.test_y


class DataClassification(Data):
    """Training data for classification
    """

    def __init__(self, func, geom, online=False):
        super(DataClassification, self).__init__('classification')
        self.func = func
        self.geom = geom
        self.online = online

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

    def train_next_batch(self, batch_size, *args, **kwargs):
        if self.online:
            self.train_x = self.geom.random_points(batch_size, 'pseudo')
            self.train_y = self.func(self.train_x)
        elif self.train_x is None:
            # self.train_x = self.geom.random_points(batch_size, 'sobol')
            self.train_x = self.geom.uniform_points(batch_size, True)
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y


class DataPDE(Data):
    """Training data for solving PDE
    """

    def __init__(self, pde, func, geom, anchors):
        super(DataPDE, self).__init__('pde')
        self.pde, self.func, self.geom = pde, func, geom
        self.anchors = anchors

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None
        self.nbc = len(anchors)

    def get_x(self, n):
        x = self.geom.uniform_points(n, True)
        x = np.append(self.anchors, x, axis=0)
        return x, self.func(x)

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x, self.train_y = self.get_x(batch_size - self.nbc)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x = self.geom.uniform_points(n, True)
        self.test_x = np.roll(self.test_x, 1, axis=0)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y


class DataIDE(Data):
    """Training data for solving IDE
    """

    def __init__(self, ide, func, geom, nbc, quad_deg):
        assert nbc == 2
        super(DataIDE, self).__init__('ide')
        self.ide, self.func, self.geom = ide, func, geom
        self.nbc = nbc
        self.quad_deg = quad_deg

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None
        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(quad_deg)

    def gen_data(self, size):
        def get_quad_points(x):
            return (self.quad_x + 1) * x / 2

        x = self.geom.uniform_points(size, True)
        x = np.roll(x, 1)
        quad_x = np.hstack(map(lambda xi: get_quad_points(xi[0]), x))
        x = np.vstack((x, quad_x[:, None]))
        return x, self.func(x)

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x, self.train_y = self.gen_data(batch_size)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x, self.test_y = self.gen_data(n)
        return self.test_x, self.test_y

    def get_int_matrix(self, size, training):
        def get_quad_weights(x):
            return self.quad_w * x / 2

        if training:
            if self.train_x is None:
                self.train_next_batch(size)
            x = self.train_x
        else:
            if self.test_x is None:
                self.test(size)
            x = self.test_x
        int_mat = np.zeros((size, x.size), dtype=config.real(np))
        for i in range(size):
            int_mat[i, size+self.quad_deg*i: size + self.quad_deg*(i+1)] = \
                get_quad_weights(x[i, 0])
        return int_mat


class DataFrac(Data):
    """Training data for solving fractional DE
    """

    def __init__(self, frac, alpha, func, geom, disc):
        if disc.meshtype == 'static':
            assert geom.idstr == 'Interval', 'Only Interval supports static mesh.'

        super(DataFrac, self).__init__('frac')
        self.frac, self.alpha, self.func, self.geom = frac, alpha, func, geom
        self.disc = disc

        self.nbc = disc.nanchor
        self.train_x, self.train_y, self.frac_train = None, None, None
        self.test_x, self.test_y, self.frac_test = None, None, None

    def get_x(self, size):
        if self.disc.meshtype == 'static':
            if size != self.disc.resolution[0] - 2 + self.disc.nanchor:
                raise ValueError('Mesh resolution does not match batch size.')
            discreteop = Fractional(self.alpha, self.geom, self.disc, None)
            x = discreteop.get_x()
            x = np.roll(x, len(x)-1)
        elif self.disc.meshtype == 'dynamic':
            # x = self.geom.random_points(size-self.disc.nanchor, 'sobol')
            x = self.geom.uniform_points(size-self.disc.nanchor, False)
            discreteop = Fractional(self.alpha, self.geom, self.disc, x)
            x = discreteop.get_x()
        if self.disc.nanchor > 0:
            x = np.vstack((self.geom.random_boundary_points(self.disc.nanchor, 'sobol'), x))
        y = self.func(x)
        return x, y, discreteop

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x, self.train_y, self.frac_train = self.get_x(batch_size)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x, self.test_y, self.frac_test = self.get_x(n)
        return self.test_x, self.test_y

    def get_int_matrix(self, size, training):
        if training:
            if self.train_x is None:
                self.train_next_batch(size)
            int_mat = self.frac_train.get_matrix(True)
        else:
            if self.test_x is None:
                self.test(size)
            int_mat = self.frac_test.get_matrix(True)
        if self.disc.meshtype == 'static':
            int_mat = np.roll(int_mat, int_mat.shape[1]-1, axis=1)
            int_mat = int_mat[1:-1]
        return int_mat


class DataFracTime(Data):
    """Training data for solving time-dependent fractional DE
    """

    def __init__(self, frac, alpha, func, geom, t_min, t_max, disc):
        super(DataFracTime, self).__init__('frac time')
        self.frac, self.alpha, self.func, self.geom = frac, alpha, func, geom
        self.t_min, self.t_max = t_min, t_max
        self.disc = disc

        self.train_x, self.train_y, self.frac_train = None, None, None
        self.test_x, self.test_y, self.frac_test = None, None, None
        self.nt, self.nbc = None, None

    def get_x(self, size):
        if self.disc.meshtype == 'static':
            self.nt = int(round(size / self.disc.resolution[0]))
            self.nbc = self.disc.resolution[0] + 2*self.nt - 2
            discreteop = FractionalTime(
                self.alpha, self.geom, self.t_min, self.t_max, self.disc, self.nt, None)
            x = discreteop.get_x()
        elif self.disc.meshtype == 'dynamic':
            self.nbc = 0
            # x = np.random.rand(size, 2)
            x = sobol_sequence.sample(size + 1, 2)[1:]
            x = x * [self.geom.diam, self.t_max-self.t_min] - [self.geom.l, self.t_min]
            discreteop = FractionalTime(
                self.alpha, self.geom, self.t_min, self.t_max, self.disc, None, x)
            x = discreteop.get_x()
        y = self.func(x)
        return x, y, discreteop

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x, self.train_y, self.frac_train = self.get_x(batch_size)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x, self.test_y, self.frac_test = self.get_x(n)
        return self.test_x, self.test_y

    def get_int_matrix(self, size, training):
        if training:
            if self.train_x is None:
                self.train_next_batch(size)
            int_mat = self.frac_train.get_matrix(True)
        else:
            if self.test_x is None:
                self.test(size)
            int_mat = self.frac_test.get_matrix(True)
        return int_mat


class DataFracInv(Data):
    """Training data for solving fractional DE inverse problem
    """

    def __init__(self, frac, func, geom, disc):
        if disc.meshtype == 'static':
            assert geom.idstr == 'Interval', 'Only Interval supports static mesh.'

        super(DataFracInv, self).__init__('frac inv')
        self.frac, self.func, self.geom = frac, func, geom
        self.disc = disc

        self.nbc = disc.nanchor
        self.train_x, self.train_y, self.frac_train = None, None, None
        self.test_x, self.test_y, self.frac_test = None, None, None

        self.alpha = 1.5
        self.alpha_train = tf.Variable(self.alpha, dtype=config.real(tf))

    def get_x(self, size):
        if self.disc.meshtype == 'static':
            if size != self.disc.resolution[0] - 2 + self.disc.nanchor:
                raise ValueError('Mesh resolution does not match batch size.')
            discreteop = Fractional(self.alpha_train, self.geom, self.disc, None)
            x = discreteop.get_x()
            x = np.roll(x, len(x)-1)
        elif self.disc.meshtype == 'dynamic':
            x = self.geom.random_points(size-self.disc.nanchor, 'sobol')
            discreteop = Fractional(self.alpha_train, self.geom, self.disc, x)
            x = discreteop.get_x()
        if self.disc.nanchor > 0:
            x = np.vstack((self.geom.random_points(self.disc.nanchor, 'sobol'), x))
        y = self.func(x)
        return x, y, discreteop

    @runifnone('train_x', 'train_y')
    def train_next_batch(self, batch_size, *args, **kwargs):
        self.train_x, self.train_y, self.frac_train = self.get_x(batch_size)
        return self.train_x, self.train_y

    @runifnone('test_x', 'test_y')
    def test(self, n, *args, **kwargs):
        self.test_x, self.test_y, self.frac_test = self.get_x(n)
        return self.test_x, self.test_y

    def get_int_matrix(self, size, training):
        if training:
            if self.train_x is None:
                self.train_next_batch(size)
            int_mat = self.frac_train.get_matrix(True)
        else:
            if self.test_x is None:
                self.test(size)
            int_mat = self.frac_test.get_matrix(True)
        if self.disc.meshtype == 'static':
            int_mat = tf.manip.roll(int_mat, self.disc.resolution[0]-1, 1)
            int_mat = int_mat[1:-1]
        return int_mat


class DataFracInvHetero(Data):
    """Training data for solving fractional DE
    """

    def __init__(self, frac, func, x_dim, y_dim, x_min, x_max, nbc, lam, maxnpt):
        assert x_dim == 1 and y_dim == 1 and nbc == 2
        super(DataFracInvHetero, self).__init__('frac inv hetero')
        self.frac, self.func = frac, func
        self.x_dim, self.y_dim = x_dim, y_dim
        self.x_min, self.x_max = x_min, x_max
        self.nbc = nbc
        self.lam = lam
        self.alpha1, self.alpha2, self.c = 1.9, 1.9, 0.5

        self.train_x, self.train_y = None, None
        self.test_x, self.test_y = None, None

        self.alpha_train1 = tf.Variable(self.alpha1, dtype=config.real(tf))
        self.alpha_train2 = tf.Variable(self.alpha2, dtype=config.real(tf))
        self.c_train = tf.Variable(self.c, dtype=config.real(tf))
        self.ws1 = self.init_quad_weights(self.alpha_train1, maxnpt)
        self.ws2 = self.init_quad_weights(self.alpha_train2, maxnpt)

    def init_quad_weights(self, alpha, n):
        w = [1]
        for j in range(1, n):
            w.append(w[-1] * (j - 1 - alpha) / j)
        return tf.convert_to_tensor(w)

    def get_npts(self, dx):
        return int(math.ceil(self.lam * dx))

    def get_quad_weight(self, n, pos):
        if pos == 'left':
            return self.ws1[:n+1]
        elif pos == 'right':
            return self.ws2[:n+1]

    def get_int_matrix_loose(self, x, size):
        pass

    def get_int_matrix_compact(self, size):
        int_mat = tf.zeros((2, size), dtype=config.real(tf))
        h = 2 / (size - 1)
        c1 = self.c_train * h**(-self.alpha_train1)
        c2 = (1 - self.c_train) * h**(-self.alpha_train2)
        for i in range(1, size - 1):
            if self.alpha1 > 1:
                row = tf.concat([tf.zeros(1, dtype=config.real(tf)),
                                 c1 * tf.reverse(self.get_quad_weight(i, 'left'), [0]),
                                 tf.zeros(size-i-2, dtype=config.real(tf))], 0)
                row += tf.concat([tf.zeros(i - 1, dtype=config.real(tf)),
                                  c2 * self.get_quad_weight(size - 1 - i, 'right'),
                                  tf.zeros(1, dtype=config.real(tf))], 0)
            else:
                row = tf.concat([tf.reverse(self.get_quad_weight(i), [0]),
                                 tf.zeros(size - i - 1)], 0)
                row += tf.concat([tf.zeros(i),
                                  self.get_quad_weight(size - 1 - i)], 0)
            row = tf.expand_dims(row, 0)
            int_mat = tf.concat([int_mat, row], 0)
        int_mat = tf.manip.roll(int_mat, 1, 1)
        return int_mat

    def get_int_matrix(self, size, training):
        if training:
            if self.train_x is None:
                self.train_next_batch(size)
            x = self.train_x
        else:
            if self.test_x is None:
                self.test(size)
            x = self.test_x
        if self.lam == 0:
            assert size == x.size
            return self.get_int_matrix_compact(size)
        else:
            return self.get_int_matrix_loose(x, size)

    def gen_data(self, size):
        def get_quad_points(x):
            if x in [-1, 1]:
                return np.array([])
            dx = x + 1
            n = self.get_npts(dx)
            h = dx / n
            quad_xl = x - np.arange(n + 1) * h
            dx = 1 - x
            n = self.get_npts(dx)
            h = dx / n
            quad_xr = x + np.arange(n + 1) * h
            return np.hstack((quad_xl, quad_xr))

        x = np.linspace(self.x_min[0], self.x_max[0], size)[:, None]
        x = np.roll(x, 1)
        if self.lam != 0:
            quad_x = np.hstack(map(lambda xi: get_quad_points(xi[0]), x))
            x = np.vstack((x, quad_x[:, None]))
        y = self.func(x)
        return x, y

    def train_next_batch(self, batch_size):
        # only support x_dim = 1, y_dim = 1
        if self.train_x is None:
            self.train_x, self.train_y = self.gen_data(batch_size)
            noisey = 0.01 * np.random.randn(*self.train_y.shape)
            self.train_y += noisey
        return self.train_x, self.train_y

    def test(self, n, dist=None):
        if self.test_x is None:
            self.test_x, self.test_y = self.gen_data(n)
        return self.test_x, self.test_y


class DataFunctional(Data):
    """Training data for functional approximation
    """

    def __init__(self, functional, x_dim, y_dim, x_min, x_max, func2sensors, nsensor):
        super(DataFunctional, self).__init__('functional')
        self.functional = functional
        self.x_dim, self.y_dim = x_dim, y_dim
        self.x_min, self.x_max = x_min, x_max
        self.func2sensors, self.nsensor = func2sensors, nsensor

        # sensors in [0, 1]
        self.sensors = np.linspace(0, 1, num=nsensor)

    def train_next_batch(self, batch_size, *args, **kwargs):
        return self.test(batch_size, 'grid')

    def test(self, n, *args, **kwargs):
        x, y = super(DataFunctional, self).test(n)
        return self.func2sensors(x, self.sensors), y


class DataFunctional2(Data):
    """Training data for functional approximation
    """

    def __init__(self, functional, x_dim, y_dim, x_min, x_max, func2sensors, nsensor):
        super(DataFunctional2, self).__init__('functional')
        self.functional = functional
        self.x_dim, self.y_dim = x_dim, y_dim
        self.x_min, self.x_max = x_min, x_max
        self.func2sensors, self.nsensor = func2sensors, nsensor

        # sensors in [0, 1]
        self.sensors = np.linspace(0, 1, num=nsensor)

    def train_next_batch(self, batch_size, *args, **kwargs):
        return self.test(batch_size, 'grid')

    def test(self, n, *args, **kwargs):
        x, _ = super(DataFunctional2, self).test(n)
        x = self.func2sensors(x, self.sensors)
        x = np.hstack((x, np.random.rand(n, 1)))
        return x, self.func(x)
