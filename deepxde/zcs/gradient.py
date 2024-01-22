"""Gradients for ZCS"""

from typing import Tuple

import numpy as np

from ..backend import backend_name, tf, torch, paddle  # noqa


class LazyGrad:
    """Gradients for ZCS with lazy evaluation."""

    def __init__(self, zcs_parameters, u):
        self.zcs_parameters = zcs_parameters
        self.n_dims = len(zcs_parameters["leaves"])

        # create tensor $a_{ij}$
        if backend_name == "tensorflow":
            self.a = tf.Variable(tf.ones_like(u), trainable=True)
        elif backend_name == "pytorch":
            self.a = torch.ones_like(u).requires_grad_()
        elif backend_name == "paddle":
            self.a = paddle.ones_like(u)  # noqa
            self.a.stop_gradient = False
        else:
            raise NotImplementedError(
                f"ZCS is not implemented for backend {backend_name}"
            )

        # omega
        if backend_name == "tensorflow":
            self.a_tape = tf.GradientTape(
                persistent=True, watch_accessed_variables=False
            )
            with self.a_tape:  # z_tape is already watching
                self.a_tape.watch(self.a)
                omega = tf.math.reduce_sum(u * self.a)
        else:
            omega = (u * self.a).sum()

        # cached lower-order derivatives of omega
        self.cached_omega_grads = {
            # the only initial element is omega itself, with all orders being zero
            (0,)
            * self.n_dims: omega
        }

    def grad_wrt_z(self, y, z):
        if backend_name == "tensorflow":
            with self.a_tape:  # z_tape is already watching
                return self.zcs_parameters["tape"].gradient(y, z)
        if backend_name == "pytorch":
            return torch.autograd.grad(y, z, create_graph=True)[0]
        if backend_name == "paddle":
            return paddle.grad(y, z, create_graph=True)[0]  # noqa
        raise NotImplementedError(
            f"ZCS is not implemented for backend {backend_name}"
        )

    def grad_wrt_a(self, y):
        if backend_name == "tensorflow":
            # no need to watch here because we don't need higher-orders w.r.t. a
            return self.a_tape.gradient(y, self.a)
        if backend_name == "pytorch":
            return torch.autograd.grad(y, self.a, create_graph=True)[0]
        if backend_name == "paddle":
            return paddle.grad(y, self.a, create_graph=True)[0]  # noqa
        raise NotImplementedError(
            f"ZCS is not implemented for backend {backend_name}"
        )

    def compute(self, required_orders: Tuple[int, ...]):
        if required_orders in self.cached_omega_grads.keys():
            # derivative w.r.t. a
            return self.grad_wrt_a(self.cached_omega_grads[required_orders])

        # find the start
        orders = np.array(required_orders)
        exists = np.array(list(self.cached_omega_grads.keys()))
        diffs = orders[None, :] - exists
        # existing orders no greater than target element-wise
        avail_indices = np.where(diffs.min(axis=1) >= 0)[0]
        # start from the closet
        start_index = np.argmin(diffs[avail_indices].sum(axis=1))
        start_orders = exists[avail_indices][start_index]

        # dim loop
        for i, zi in enumerate(self.zcs_parameters["leaves"]):
            # order loop
            while start_orders[i] != required_orders[i]:
                omega_grad = self.grad_wrt_z(
                    self.cached_omega_grads[tuple(start_orders)], zi
                )
                start_orders[i] += 1
                self.cached_omega_grads[tuple(start_orders)] = omega_grad

        # derivative w.r.t. a
        return self.grad_wrt_a(self.cached_omega_grads[required_orders])
