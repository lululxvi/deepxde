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


class TrainState(object):

    def __init__(self):
        self.epoch, self.step = 0, 0

        self.sess = None

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.y_pred_train, self.y_pred_test, self.y_std_test = None, None, None
        self.loss_train, self.loss_test = None, None
        self.metrics_test = None

        # the best results correspond to the min train loss
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_y, self.best_ystd = None, None
        self.best_metrics = None

    def update_tfsession(self, sess):
        self.sess = sess

    def update_data_train(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def update_data_test(self, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y, self.best_ystd = self.y_pred_test, self.y_std_test
            self.best_metrics = self.metrics_test


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
        self.train_state = TrainState()

    @timing
    def compile(self, optimizer, lr, batch_size, ntest, metrics=None, decay=None, loss_weights=None):
        print('Compiling model...')

        self.optimizer = optimizer
        self.batch_size, self.ntest = batch_size, ntest

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
    def train(self, epochs, validation_every=1000, uncertainty=False, errstop=None, callbacks=None, print_model=False):
        print('Training model...')

        self.open_tfsession()
        self.sess.run(tf.global_variables_initializer())

        if print_model:
            self.print_model()
        if self.optimizer in Model.scipy_opts:
            losshistory = self.train_scipy()
        else:
            losshistory = self.train_sgd(epochs, validation_every, uncertainty, errstop, callbacks)
        if print_model:
            self.print_model()

        self.close_tfsession()
        return losshistory, self.train_state

    def open_tfsession(self):
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.train_state.update_tfsession(self.sess)

    def close_tfsession(self):
        self.sess.close()

    def train_sgd(self, epochs, validation_every, uncertainty, errstop, callbacks):
        losshistory = []
        callbacks = CallbackList(callbacks=callbacks)

        self.train_state.update_data_test(*self.data.test(self.ntest))

        callbacks.on_train_begin(self.train_state)

        for i in range(epochs):
            callbacks.on_epoch_begin(self.train_state)
            callbacks.on_batch_begin(self.train_state)

            self.train_state.update_data_train(*self.data.train_next_batch(self.batch_size))
            self.sess.run(
                [self.losses, self.train_op],
                feed_dict={
                    self.net.training: True, self.net.dropout: True, self.net.data_id: 0,
                    self.net.x: self.train_state.X_train, self.net.y_: self.train_state.y_train})

            self.train_state.epoch += 1
            self.train_state.step += 1

            if i % validation_every == 0 or i + 1 == epochs:
                self.test(uncertainty)

                losshistory.append([i] + list(self.train_state.loss_test) + self.train_state.metrics_test)
                print('Epoch: %d, loss: %s, val_loss: %s, val_metric: %s' % (
                    i, self.train_state.loss_train, self.train_state.loss_test, self.train_state.metrics_test))
                sys.stdout.flush()

                # if errstop is not None and err_norm < errstop:
                #     break

            callbacks.on_batch_end(self.train_state)
            callbacks.on_epoch_end(self.train_state)

        callbacks.on_train_end(self.train_state)

        return np.array(losshistory)

    def train_scipy(self):
        self.train_state.update_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_op.minimize(
            self.sess,
            feed_dict={
                self.net.training: True, self.net.dropout: True, self.net.data_id: 0,
                self.net.x: self.train_state.X_train, self.net.y_: self.train_state.y_train})

        self.train_state.update_data_test(*self.data.test(self.ntest))
        y_pred = self.sess.run(
            self.net.y,
            feed_dict={
                self.net.training: False, self.net.dropout: False, self.net.data_id: 1,
                self.net.x: self.train_state.X_test})

        self.train_state.best_y = y_pred

        return None

    def test(self, uncertainty):
        self.train_state.loss_train, self.train_state.y_pred_train = self.sess.run(
            [self.losses, self.net.y],
            feed_dict={
                self.net.training: False, self.net.dropout: False, self.net.data_id: 0,
                self.net.x: self.train_state.X_train, self.net.y_: self.train_state.y_train})

        if uncertainty:
            losses, y_preds = [], []
            for _ in range(1000):
                loss_one, y_pred_test_one = self.sess.run(
                    [self.losses, self.net.y],
                    feed_dict={
                        self.net.training: False, self.net.dropout: True, self.net.data_id: 1,
                        self.net.x: self.train_state.X_test, self.net.y_: self.train_state.y_test})
                losses.append(loss_one)
                y_preds.append(y_pred_test_one)
            self.train_state.loss_test = np.mean(losses, axis=0)
            self.train_state.y_pred_test, self.train_state.y_std_test = np.mean(y_preds, axis=0), np.std(y_preds, axis=0)
        else:
            self.train_state.loss_test, self.train_state.y_pred_test = self.sess.run(
                [self.losses, self.net.y],
                feed_dict={
                    self.net.training: False, self.net.dropout: False, self.net.data_id: 1,
                    self.net.x: self.train_state.X_test, self.net.y_: self.train_state.y_test})

        self.train_state.metrics_test = [m(self.train_state.y_test, self.train_state.y_pred_test) for m in self.metrics]
        self.train_state.update_best()

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

    def print_model(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)
