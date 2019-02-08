from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf

from . import config
from .utils import timing


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

    @timing
    def compile(self, optimizer, lr, batch_size, ntest, decay=None, loss_weights=None):
        print('Compiling model...')

        self.optimizer = optimizer
        self.batch_size, self.ntest = batch_size, ntest

        self.losses = self.get_losses(self.net.x, self.net.y, self.net.y_, batch_size, ntest, self.net.training)
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

    @timing
    def train(self, epochs, uncertainty=False, errstop=None, print_model=False, callback=None):
        print('Training model...')

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        sess.run(tf.global_variables_initializer())
        if print_model:
            self.print_model(sess)

        testloss = []
        test_xs, test_ys = self.data.test(self.ntest)
        if self.optimizer in Model.scipy_opts:
            batch_xs, batch_ys = self.data.train_next_batch(self.batch_size)
            self.train_op.minimize(sess, feed_dict={self.net.training: True, self.net.x: batch_xs, self.net.y_: batch_ys})
            y_pred = sess.run(self.net.y, feed_dict={self.net.training: False, self.net.x: test_xs})
            return None, None, None, np.hstack((test_xs, test_ys, y_pred))

        minloss, besty, bestystd = np.inf, None, None
        for i in range(epochs):
            batch_xs, batch_ys = self.data.train_next_batch(self.batch_size)
            sess.run([self.losses, self.train_op], feed_dict={self.net.training: True, self.net.x: batch_xs, self.net.y_: batch_ys})

            if i % 1000 == 0 or i + 1 == epochs:
                if uncertainty:
                    errs, y_preds = [], []
                    for _ in range(1000):
                        err, y_pred = sess.run([self.losses, self.net.y], feed_dict={self.net.training: True, self.net.x: test_xs, self.net.y_: test_ys})
                        errs.append(err)
                        y_preds.append(y_pred)
                    err = np.mean(errs, axis=0)
                    y_pred, y_std = np.mean(y_preds, axis=0), np.std(y_preds, axis=0)
                else:
                    err_train, ytrain_pred = sess.run([self.losses, self.net.y], feed_dict={
                        self.net.training: False, self.net.x: batch_xs, self.net.y_: batch_ys})
                    err, y_pred = sess.run([self.losses, self.net.y], feed_dict={
                        self.net.training: False, self.net.x: test_xs, self.net.y_: test_ys})

                if self.data.target == 'classification':
                    err_norm = np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(test_ys, 1)))
                elif self.data.target in ['frac']:
                    err_norm = np.linalg.norm(test_ys[self.data.nbc:self.ntest] - y_pred[self.data.nbc:self.ntest]) / np.linalg.norm(test_ys[self.data.nbc:self.ntest])
                else:
                    err_norm = np.linalg.norm(test_ys[:self.ntest] - y_pred[:self.ntest]) / np.linalg.norm(test_ys[:self.ntest])
                    # err_norm = np.mean(np.abs(test_ys[:ntest] - y_pred[:ntest]) / test_ys[:ntest])
                testloss.append([i] + list(err) + [err_norm])

                if self.data.target == 'frac inv':
                    alpha = sess.run(self.data.alpha_train)
                    print(i, err, err_norm, alpha)
                elif self.data.target == 'frac inv hetero':
                    alphac = sess.run([self.data.alpha_train1, self.data.alpha_train2, self.data.c_train])
                    print(i, err, err_norm, alphac)
                else:
                    print(i, err_train, err, err_norm)
                if callback is not None:
                    callback(i, batch_xs, batch_ys, ytrain_pred, test_xs, test_ys, y_pred)
                sys.stdout.flush()

                if np.sum(err) < minloss:
                    minloss, besty, besty_train = np.sum(err), y_pred, ytrain_pred
                    if 'y_std' in locals():
                        bestystd = y_std
                    if self.data.target == 'frac inv':
                        self.data.alpha = alpha
                    elif self.data.target == 'frac inv hetero':
                        self.data.alpha1, self.data.alpha2, self.data.c = alphac

                if errstop is not None and err_norm < errstop:
                    break

        if print_model:
            self.print_model(sess)
        model = sess.run(tf.trainable_variables())
        sess.close()
        if bestystd is None:
            # return model, np.array(testloss), np.hstack((batch_xs, batch_ys, besty_train)), np.hstack((test_xs, test_ys, besty))
            return model, np.array(testloss), np.hstack((batch_xs, batch_ys, ytrain_pred)), np.hstack((test_xs, test_ys, y_pred))
        else:
            return model, np.array(testloss), np.hstack((test_xs, test_ys, besty, bestystd))

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

    def print_model(self, sess):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)
