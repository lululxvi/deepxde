# author: Lu Lu
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import numpy as np
import tensorflow as tf

from . import config


class Model(object):
    """model
    """

    def __init__(self, net, optimizer):
        self.net = net
        self.optimizer = optimizer

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

    def loss(self, data, x, y, y_, batch_size, ntest, training):
        if data.target in ['func', 'functional']:
            l = [tf.losses.mean_squared_error(y_, y)]
            # l = [tf.reduce_mean(tf.abs(y_ - y) / y_)]
        elif data.target == 'classification':
            l = [tf.losses.softmax_cross_entropy(y_, y)]
        elif data.target == 'pde':
            f = data.pde(x, y)[data.nbc:]
            l = [tf.losses.mean_squared_error(y_[:data.nbc], y[:data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif data.target == 'ide':
            int_mat_train = data.get_int_matrix(batch_size, True)
            int_mat_test = data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: data.ide(x, y, int_mat_train),
                        lambda: data.ide(x, y, int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:data.nbc], y[:data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif data.target == 'frac':
            int_mat_train = data.get_int_matrix(batch_size, True)
            int_mat_test = data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: data.frac(x[data.nbc:], y[data.nbc:], int_mat_train),
                        lambda: data.frac(x[data.nbc:], y[data.nbc:], int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:data.nbc], y[:data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif data.target == 'frac time':
            int_mat_train = data.get_int_matrix(batch_size, True)
            int_mat_test = data.get_int_matrix(ntest, False)
            dy_t = tf.gradients(y, x)[0][data.nbc:, -1:]
            f = tf.cond(training,
                        lambda: data.frac(x[data.nbc:], y[data.nbc:], dy_t, int_mat_train),
                        lambda: data.frac(x[data.nbc:], y[data.nbc:], dy_t, int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:data.nbc], y[:data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif data.target == 'frac inv':
            int_mat_train = data.get_int_matrix(batch_size, True)
            int_mat_test = data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: data.frac(data.alpha_train, x[data.nbc:], y[data.nbc:], int_mat_train),
                        lambda: data.frac(data.alpha_train, x[data.nbc:], y[data.nbc:], int_mat_test))
            l = [tf.losses.mean_squared_error(y_[:data.nbc], y[:data.nbc]),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        elif data.target == 'frac inv hetero':
            int_mat_train = data.get_int_matrix(batch_size, True)
            int_mat_test = data.get_int_matrix(ntest, False)
            f = tf.cond(training,
                        lambda: data.frac(x, y, int_mat_train)[data.nbc:],
                        lambda: data.frac(x, y, int_mat_test)[data.nbc:])
            l = [tf.losses.mean_squared_error(y_, y),
                 tf.losses.mean_squared_error(tf.zeros(tf.shape(f)), f)]
        else:
            raise ValueError('target')
        return tf.convert_to_tensor(l)

    def train(self, data, batch_size, lr, nepoch, ntest, uncertainty=False,
              regularization=None, decay=None, errstop=None, lossweight=None,
              print_model=False, callback=None):
        print('Building neural network...')
        scipy_opts = ['BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG']

        training, x, y, y_ = self.net.training, self.net.x, self.net.y, self.net.y_

        loss = self.loss(data, x, y, y_, batch_size, ntest, training)
        if lossweight is not None:
            loss *= lossweight
        totalloss = tf.reduce_sum(loss)
        lr, global_step = self.get_learningrate(lr, decay)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.optimizer in scipy_opts:
                opt = tf.contrib.opt.ScipyOptimizerInterface(
                    totalloss, method=self.optimizer, options={'disp': True})
            else:
                train = self.get_optimizer(self.optimizer, lr).minimize(
                    totalloss, global_step=global_step)

        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config=tfconfig)
        sess.run(tf.global_variables_initializer())
        if print_model:
            self.print_model(sess)

        print('Training...')
        testloss = []
        test_xs, test_ys = data.test(ntest)
        if self.optimizer in scipy_opts:
            batch_xs, batch_ys = data.train_next_batch(batch_size)
            opt.minimize(sess, feed_dict={training: True, x: batch_xs, y_: batch_ys})
            y_pred = sess.run(y, feed_dict={training: False, x: test_xs})
            return None, None, None, np.hstack((test_xs, test_ys, y_pred))

        minloss, besty, bestystd = np.inf, None, None
        for i in range(nepoch):
            batch_xs, batch_ys = data.train_next_batch(batch_size)
            sess.run([loss, train], feed_dict={training: True, x: batch_xs, y_: batch_ys})

            if i % 1000 == 0 or i + 1 == nepoch:
                if uncertainty:
                    errs, y_preds = [], []
                    for _ in range(1000):
                        err, y_pred = sess.run([loss, y], feed_dict={training: True, x: test_xs, y_: test_ys})
                        errs.append(err)
                        y_preds.append(y_pred)
                    err = np.mean(errs, axis=0)
                    y_pred, y_std = np.mean(y_preds, axis=0), np.std(y_preds, axis=0)
                else:
                    err_train, ytrain_pred = sess.run([loss, y], feed_dict={
                        training: False, x: batch_xs, y_: batch_ys})
                    err, y_pred = sess.run([loss, y], feed_dict={
                        training: False, x: test_xs, y_: test_ys})

                if data.target == 'classification':
                    err_norm = np.mean(np.equal(np.argmax(y_pred, 1), np.argmax(test_ys, 1)))
                elif data.target in ['frac']:
                    err_norm = np.linalg.norm(test_ys[data.nbc:ntest] - y_pred[data.nbc:ntest]) / np.linalg.norm(test_ys[data.nbc:ntest])
                else:
                    err_norm = np.linalg.norm(test_ys[:ntest] - y_pred[:ntest]) / np.linalg.norm(test_ys[:ntest])
                    # err_norm = np.mean(np.abs(test_ys[:ntest] - y_pred[:ntest]) / test_ys[:ntest])
                testloss.append([i] + list(err) + [err_norm])

                if data.target == 'frac inv':
                    alpha = sess.run(data.alpha_train)
                    print(i, err, err_norm, alpha)
                elif data.target == 'frac inv hetero':
                    alphac = sess.run([data.alpha_train1, data.alpha_train2, data.c_train])
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
                    if data.target == 'frac inv':
                        data.alpha = alpha
                    elif data.target == 'frac inv hetero':
                        data.alpha1, data.alpha2, data.c = alphac

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

    def print_model(self, sess):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)
