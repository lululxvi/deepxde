from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf

from . import config
from .callbacks import CallbackList
from .metrics import get_metrics
from .utils import timing


class TrainingState(object):

    def __init__(self, sess, X_test, y_test):
        self.epoch, self.step = 0, 0
        self.sess = sess
        self.X_test, self.y_test = X_test, y_test

        self.X_train, self.y_train, self.y_train_pred = None, None, None
        self.y_test_pred = None
        self.loss_train, self.loss_test = None, None

        # the best results correspond to the min training loss
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_y, self.best_ystd = None, None

    def update(self, X_train, y_train, y_train_pred, y_test_pred,
               loss_train, loss_test, y_test_predstd=None):
        self.X_train, self.y_train, self.y_train_pred = X_train, y_train, y_train_pred
        self.y_test_pred = y_test_pred
        self.loss_train, self.loss_test = loss_train, loss_test

        if self.best_loss_train > np.sum(loss_train):
            self.best_loss_train = np.sum(loss_train)
            self.best_loss_test = np.sum(loss_test)
            self.best_y, self.best_ystd = y_test_pred, y_test_predstd


class Model(object):
    """model
    """
    scipy_opts = ['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG']

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.optimizer = None
        self.batch_size, self.ntest = None, None

        self.losses, self.totalloss = None, None
        self.train_op = None
        self.metrics = None

        self.sess = None

    @timing
    def compile(self, optimizer, lr, batch_size, ntest, metrics=None, decay=None, loss_weights=None):
        print('Compiling model...')

        self.optimizer = optimizer
        self.batch_size, self.ntest = batch_size, ntest

        # self.losses = self.get_losses(self.net.x, self.net.y, self.net.y_, batch_size, ntest, self.net.training)
        self.losses = tf.convert_to_tensor(self.data.losses(self.net.y_, self.net.y, self))
        if loss_weights is not None:
            self.losses *= loss_weights
        self.totalloss = tf.reduce_sum(self.losses)
        lr, global_step = self.get_learningrate(lr, decay)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.optimizer in Model.scipy_opts:
                self.train_op = tf.contrib.opt.ScipyOptimizerInterface(
                    self.totalloss, method=self.optimizer, options={'disp': True})
            else:
                self.train_op = self.get_optimizer(self.optimizer, lr).minimize(
                    self.totalloss, global_step=global_step)

        self.metrics = get_metrics(metrics)

    @timing
    def train(self, epochs, uncertainty=False, errstop=None, callbacks=None, print_model=False):
        print('Training model...')

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.sess.run(tf.global_variables_initializer())
        if print_model:
            self.print_model()

        if self.optimizer in Model.scipy_opts:
            losshistory, training_state = self.train_scipy()
        else:
            losshistory, training_state = self.train_sgd(epochs, uncertainty, errstop, callbacks)

        if print_model:
            self.print_model()
        self.sess.close()
        return losshistory, training_state

    def train_sgd(self, epochs, uncertainty, errstop, callbacks):
        losshistory = []
        callbacks = CallbackList(callbacks=callbacks)

        test_xs, test_ys = self.data.test(self.ntest)
        training_state = TrainingState(self.sess, test_xs, test_ys)

        callbacks.on_train_begin(training_state)

        for i in range(epochs):
            callbacks.on_epoch_begin(training_state)
            callbacks.on_batch_begin(training_state)

            batch_xs, batch_ys = self.data.train_next_batch(self.batch_size)
            self.sess.run([self.losses, self.train_op], feed_dict={self.net.training: True, self.net.x: batch_xs, self.net.y_: batch_ys})

            training_state.epoch += 1
            training_state.step += i

            if i % 1000 == 0 or i + 1 == epochs:
                self.test(i, batch_xs, batch_ys, test_xs, test_ys, training_state, losshistory, uncertainty)

                # if errstop is not None and err_norm < errstop:
                #     break
            
            callbacks.on_batch_end(training_state)
            callbacks.on_epoch_end(training_state)

        callbacks.on_train_end(training_state)

        return np.array(losshistory), training_state

    def train_scipy(self):
        batch_xs, batch_ys = self.data.train_next_batch(self.batch_size)
        self.train_op.minimize(self.sess, feed_dict={self.net.training: True, self.net.x: batch_xs, self.net.y_: batch_ys})

        test_xs, test_ys = self.data.test(self.ntest)
        y_pred = self.sess.run(self.net.y, feed_dict={self.net.training: False, self.net.x: test_xs})

        training_state = TrainingState(self.sess, test_xs, test_ys)
        training_state.X_train, training_state.y_train = batch_xs, batch_ys
        training_state.best_y = y_pred

        return None, training_state

    def test(self, i, batch_xs, batch_ys, test_xs, test_ys, training_state, losshistory, uncertainty):
        loss, y_pred, y_std = None, None, None
        if uncertainty:
            losses, y_preds = [], []
            for _ in range(1000):
                loss, y_pred = self.sess.run([self.losses, self.net.y], feed_dict={self.net.training: True, self.net.x: test_xs, self.net.y_: test_ys})
                losses.append(loss)
                y_preds.append(y_pred)
            loss = np.mean(losses, axis=0)
            y_pred, y_std = np.mean(y_preds, axis=0), np.std(y_preds, axis=0)
        else:
            loss_train, ytrain_pred = self.sess.run([self.losses, self.net.y], feed_dict={
                self.net.training: False, self.net.x: batch_xs, self.net.y_: batch_ys})
            loss, y_pred = self.sess.run([self.losses, self.net.y], feed_dict={
                self.net.training: False, self.net.x: test_xs, self.net.y_: test_ys})

        training_state.update(batch_xs, batch_ys, ytrain_pred, y_pred, loss_train, loss, y_test_predstd=y_std)

        metrics = [m(test_ys, y_pred) for m in self.metrics]
        losshistory.append([i] + list(loss) + metrics)
        print('Epoch: %d, loss: %s, val_loss: %s, val_metric: %s' % (i, loss_train, loss, metrics))
        sys.stdout.flush()

    def get_optimizer(self, name, lr):
        return {
            'sgd': tf.train.GradientDescentOptimizer(lr),
            'sgdnesterov': tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
            'adagrad': tf.train.AdagradOptimizer(0.01),
            'adadelta': tf.train.AdadeltaOptimizer(),
            'rmsprop': tf.train.RMSPropOptimizer(lr),
            'adam': tf.train.AdamOptimizer(lr)
        }[name]

    def get_learningrate(self, lr, decay):
        if decay is None:
            return lr, None
        global_step = tf.Variable(0, trainable=False)
        return {
            'inverse time': tf.train.inverse_time_decay(lr, global_step, decay[1], decay[2]),
            'cosine': tf.train.cosine_decay(lr, global_step, decay[1], alpha=decay[2])
        }[decay[0]], global_step

    def get_losses(self, x, y, y_, batch_size, ntest, training):
        if self.data.target in ['func', 'functional']:
            l = [tf.losses.mean_squared_error(y_, y)]
            # l = [tf.reduce_mean(tf.abs(y_ - y) / y_)]
        elif self.data.target == 'classification':
            l = [tf.losses.softmax_cross_entropy(y_, y)]
        elif self.data.target == 'pde':
            f = self.data.pde(x, y)[self.data.nbc:]
            l = [tf.losses.mean_squared_error(y_[:self.data.nbc], y[:self.data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif self.data.target == 'ide':
            int_mat_train = self.data.get_int_matrix(batch_size, True)
            int_mat_test = self.data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: self.data.ide(x, y, int_mat_train),
                        lambda: self.data.ide(x, y, int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:self.data.nbc], y[:self.data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif self.data.target == 'frac':
            int_mat_train = self.data.get_int_matrix(batch_size, True)
            int_mat_test = self.data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: self.data.frac(x[self.data.nbc:], y[self.data.nbc:], int_mat_train),
                        lambda: self.data.frac(x[self.data.nbc:], y[self.data.nbc:], int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:self.data.nbc], y[:self.data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif self.data.target == 'frac time':
            int_mat_train = self.data.get_int_matrix(batch_size, True)
            int_mat_test = self.data.get_int_matrix(ntest, False)
            dy_t = tf.gradients(y, x)[0][self.data.nbc:, -1:]
            f = tf.cond(training,
                        lambda: self.data.frac(x[self.data.nbc:], y[self.data.nbc:], dy_t, int_mat_train),
                        lambda: self.data.frac(x[self.data.nbc:], y[self.data.nbc:], dy_t, int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:self.data.nbc], y[:self.data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif self.data.target == 'frac inv':
            int_mat_train = self.data.get_int_matrix(batch_size, True)
            int_mat_test = self.data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: self.data.frac(self.data.alpha_train, x[self.data.nbc:], y[self.data.nbc:], int_mat_train),
                        lambda: self.data.frac(self.data.alpha_train, x[self.data.nbc:], y[self.data.nbc:], int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:self.data.nbc], y[:self.data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif self.data.target == 'frac inv hetero':
            int_mat_train = self.data.get_int_matrix(batch_size, True)
            int_mat_test = self.data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: self.data.frac(x, y, int_mat_train)[self.data.nbc:],
                        lambda: self.data.frac(x, y, int_mat_test)[self.data.nbc:])
            l = [tf.losses.mean_squared_error(y_, y),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        else:
            raise ValueError('target')
        return tf.convert_to_tensor(l)

    def print_model(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)
