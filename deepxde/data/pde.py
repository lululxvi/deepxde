from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .data import Data
from .. import config
from ..backend import tf
from ..utils import get_num_args, run_if_all_none


class PDE(Data):
    """ODE or time-independent PDE solver.

    Args:
        geometry: Instance of ``Geometry``.
        pde: A global PDE or a list of PDEs. ``None`` if no global PDE.
        bcs: A boundary condition or a list of boundary conditions. ``[]`` if no boundary condition.
        num_domain (int): The number of training residual points sampled inside the domain.
        num_boundary (int): The number of training residual points sampled on the boundary.
        train_distribution (string): The distribution to sample training residual points. One of the following:
            "uniform" (equispaced grid), "pseudo" (pseudorandom), "LHS" (Latin hypercube sampling), "Halton" (Halton
            sequence), "Hammersley" (Hammersley sequence), or "Sobol" (Sobol sequence).
        anchors: A Numpy array of training residual points, in addition to the `num_domain` and `num_boundary` sampled
            points.
        exclusions: A Numpy array of points to be excluded for training.
        solution: The reference solution.
        num_test: The number of residual points for testing the PDE residual.
        auxiliary_var_function: A function that inputs `train_x` or `test_x` and outputs auxiliary variables.

    Attributes:
        train_x_all: A Numpy array of all residual points for training. `train_x_all` is unordered,
            and does not have duplication.
        train_x: A Numpy array of the residual points fed into the network for training.
            `train_x` is a subset of `train_x_all`, ordered from BCs to PDE, and may have duplicate points.
        train_x_bc: A Numpy array of the residual points on boundary.
            `train_x_bc` is a subset of `train_x_all` at the first step of training, by default it won't be updated
            when `train_x_all` changes. To update `train_x_bc`, set it to `None` and call `bc_points`,
            and then update the loss function by `model.compile`.
        num_bcs (list): `num_bcs[i]` is the number of residual points for `bcs[i]`.
        test_x: A Numpy array of the residual points fed into the network for testing the PDE residual.
        train_aux_vars: Auxiliary variables that associate with `train_x`.
        test_aux_vars: Auxiliary variables that associate with `test_x`.
    """

    def __init__(
        self,
        geometry,
        pde,
        bcs,
        num_domain=0,
        num_boundary=0,
        train_distribution="Sobol",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
    ):
        self.geom = geometry
        self.pde = pde
        self.bcs = bcs if isinstance(bcs, (list, tuple)) else [bcs]

        self.num_domain = num_domain
        self.num_boundary = num_boundary
        if train_distribution not in [
            "uniform",
            "pseudo",
            "LHS",
            "Halton",
            "Hammersley",
            "Sobol",
        ]:
            raise ValueError(
                "train_distribution == {} is not available choices.".format(
                    train_distribution
                )
            )
        self.train_distribution = train_distribution
        self.anchors = anchors
        self.exclusions = exclusions

        self.soln = solution
        self.num_test = num_test

        self.auxiliary_var_fn = auxiliary_var_function

        self.train_x_all = None
        self.train_x, self.train_y = None, None
        self.train_x_bc = None
        self.num_bcs = None
        self.test_x, self.test_y = None, None
        self.train_aux_vars, self.test_aux_vars = None, None

        self.train_next_batch()
        self.test()

    def losses(self, targets, outputs, loss, model):
        f = []
        # Always build the gradients in the PDE here, so that we can reuse all the gradients in dde.grad. If we build
        # the gradients in losses_train(), then error occurs when we use these gradients in losses_test() during
        # sess.run(), because one branch in tf.cond() cannot use the Tensors created in the other branch.
        if self.pde is not None:
            if get_num_args(self.pde) == 2:
                f = self.pde(model.net.inputs, outputs)
            elif get_num_args(self.pde) == 3:
                if self.auxiliary_var_fn is None:
                    raise ValueError("Auxiliary variable function not defined.")
                f = self.pde(model.net.inputs, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]

        if not isinstance(loss, (list, tuple)):
            loss = [loss] * (len(f) + len(self.bcs))
        elif len(loss) != len(f) + len(self.bcs):
            raise ValueError(
                "There are {} errors, but only {} losses.".format(
                    len(f) + len(self.bcs), len(loss)
                )
            )

        def losses_train():
            f_train = f
            bcs_start = np.cumsum([0] + self.num_bcs)
            error_f = [fi[bcs_start[-1] :] for fi in f_train]
            losses = [
                loss[i](tf.zeros(tf.shape(error), dtype=config.real(tf)), error)
                for i, error in enumerate(error_f)
            ]
            for i, bc in enumerate(self.bcs):
                beg, end = bcs_start[i], bcs_start[i + 1]
                error = bc.error(self.train_x, model.net.inputs, outputs, beg, end)
                losses.append(
                    loss[len(error_f) + i](
                        tf.zeros(tf.shape(error), dtype=config.real(tf)), error
                    )
                )
            return losses

        def losses_test():
            f_test = f
            return [
                loss[i](tf.zeros(tf.shape(fi), dtype=config.real(tf)), fi)
                for i, fi in enumerate(f_test)
            ] + [tf.constant(0, dtype=config.real(tf)) for _ in self.bcs]

        return tf.cond(tf.equal(model.net.data_id, 0), losses_train, losses_test)

    @run_if_all_none("train_x", "train_y", "train_aux_vars")
    def train_next_batch(self, batch_size=None):
        self.train_x_all = self.train_points()
        self.train_x = self.bc_points()
        if self.pde is not None:
            self.train_x = np.vstack((self.train_x, self.train_x_all))
        self.train_y = self.soln(self.train_x) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.train_aux_vars = self.auxiliary_var_fn(self.train_x)
        return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None:
            self.test_x = self.train_x_all
        else:
            self.test_x = self.test_points()
        self.test_y = self.soln(self.test_x) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.test_aux_vars = self.auxiliary_var_fn(self.test_x)
        return self.test_x, self.test_y, self.test_aux_vars

    def resample_train_points(self):
        """Resample the training residual points.

        Warning: After resampling, need to call ``Model.compile()`` to update the loss.
        """
        self.train_x, self.train_y, self.train_aux_vars = None, None, None
        self.train_next_batch()

    def add_anchors(self, anchors):
        """Add new anchors into the training residual points.

        Warning: After adding anchors, need to call ``Model.compile()`` to update the loss.
        """
        if self.anchors is None:
            self.anchors = anchors
        else:
            self.anchors = np.vstack((anchors, self.anchors))
        self.train_x_all = np.vstack((anchors, self.train_x_all))
        self.train_x = self.bc_points()
        if self.pde is not None:
            self.train_x = np.vstack((self.train_x, self.train_x_all))
        self.train_y = self.soln(self.train_x) if self.soln else None
        if self.auxiliary_var_fn is not None:
            self.train_aux_vars = self.auxiliary_var_fn(self.train_x)

    def train_points(self):
        X = np.empty((0, self.geom.dim))
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geom.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geom.random_points(
                    self.num_domain, random=self.train_distribution
                )
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geom.random_boundary_points(
                    self.num_boundary, random=self.train_distribution
                )
            X = np.vstack((tmp, X))
        if self.anchors is not None:
            X = np.vstack((self.anchors, X))
        if self.exclusions is not None:

            def is_not_excluded(x):
                return not np.any([np.allclose(x, y) for y in self.exclusions])

            X = np.array(list(filter(is_not_excluded, X)))
        return X

    @run_if_all_none("train_x_bc")
    def bc_points(self):
        x_bcs = [bc.collocation_points(self.train_x_all) for bc in self.bcs]
        self.num_bcs = list(map(len, x_bcs))
        self.train_x_bc = (
            np.vstack(x_bcs) if x_bcs else np.empty([0, self.train_x_all.shape[-1]])
        )
        return self.train_x_bc

    def test_points(self):
        return self.geom.uniform_points(self.num_test, True)


class TimePDE(PDE):
    """Time-dependent PDE solver.

    Args:
        num_domain: Number of f training points.
        num_boundary: Number of boundary condition points on the geometry boundary.
        num_initial: Number of initial condition points.
    """

    def __init__(
        self,
        geometryxtime,
        pde,
        ic_bcs,
        num_domain=0,
        num_boundary=0,
        num_initial=0,
        train_distribution="Sobol",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test=None,
        auxiliary_var_function=None,
    ):
        self.num_initial = num_initial
        super(TimePDE, self).__init__(
            geometryxtime,
            pde,
            ic_bcs,
            num_domain,
            num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
            auxiliary_var_function=auxiliary_var_function,
        )

    def train_points(self):
        X = super(TimePDE, self).train_points()
        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geom.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geom.random_initial_points(
                    self.num_initial, random=self.train_distribution
                )
            if self.exclusions is not None:

                def is_not_excluded(x):
                    return not np.any([np.allclose(x, y) for y in self.exclusions])

                tmp = np.array(list(filter(is_not_excluded, tmp)))
            X = np.vstack((tmp, X))
        return X
