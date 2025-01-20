# Rewrite of the original file in DeepXDE: https://github.com/lululxvi/deepxde
# ==============================================================================


from __future__ import annotations

from typing import Callable, Sequence, Union, Optional, Dict, List

import brainstate as bst
import brainunit as u
import jax.tree
import numpy as np

from deepxde.pinnx import utils
from deepxde.pinnx.geometry import GeometryXTime, DictPointGeometry
from .base import Problem
from ..icbc.base import ICBC

__all__ = [
    "PDE",
    "TimePDE"
]


class PDE(Problem):
    """ODE or time-independent PDE solver.

    Args:
        geometry: Instance of ``Geometry``.
        constraints: A boundary condition or a list of boundary conditions. Use ``[]`` if no
            boundary condition.
        approximator: A neural network trainer for approximating the solution.
        num_domain (int): The number of training points sampled inside the domain.
        num_boundary (int): The number of training points sampled on the boundary.
        train_distribution (string): The distribution to sample training points. One of
            the following: "uniform" (equispaced grid), "pseudo" (pseudorandom), "LHS"
            (Latin hypercube sampling), "Halton" (Halton sequence), "Hammersley"
            (Hammersley sequence), or "Sobol" (Sobol sequence).
        anchors: A Numpy array of training points, in addition to the `num_domain` and
            `num_boundary` sampled points.
        exclusions: A Numpy array of points to be excluded for training.
        solution: The reference solution.
        num_test: The number of points sampled inside the domain for testing PDE loss.
            The testing points for BCs/ICs are the same set of points used for training.
            If ``None``, then the training points will be used for testing.

    Warning:
        The testing points include points inside the domain and points on the boundary,
        and they may not have the same density, and thus the entire testing points may
        not be uniformly distributed. As a result, if you have a reference solution
        (`solution`) and would like to compute a metric such as

        .. code-block:: python

            Trainer.compile(metrics=["l2 relative error"])

        then the metric may not be very accurate. To better compute a metric, you can
        sample the points manually, and then use ``Trainer.predict()`` to predict the
        solution on these points and compute the metric:

        .. code-block:: python

            x = geometry.uniform_points(num, boundary=True)
            y_true = ...
            y_pred = trainer.predict(x)
            error= pinnx.metrics.l2_relative_error(y_true, y_pred)

    Attributes:
        train_x_all: A Numpy array of points for PDE training. `train_x_all` is
            unordered, and does not have duplication. If there is PDE, then
            `train_x_all` is used as the training points of PDE.
        train_x_bc: A Numpy array of the training points for BCs. `train_x_bc` is
            constructed from `train_x_all` at the first step of training, by default it
            won't be updated when `train_x_all` changes. To update `train_x_bc`, set it
            to `None` and call `bc_points`, and then update the loss function by
            ``trainer.compile()``.
        num_bcs (list): `num_bcs[i]` is the number of points for `constraints[i]`.
        train_x: A Numpy array of the points fed into the network for training.
            `train_x` is ordered from BC points (`train_x_bc`) to PDE points
            (`train_x_all`), and may have duplicate points.
        test_x: A Numpy array of the points fed into the network for testing, ordered
            from BCs to PDE. The BC points are exactly the same points in `train_x_bc`.
    """

    def __init__(
        self,
        geometry: DictPointGeometry,
        pde: Callable,
        constraints: Union[ICBC, Sequence[ICBC]],
        approximator: Optional[bst.nn.Module] = None,
        solution: Callable[[bst.typing.PyTree], bst.typing.PyTree] = None,
        loss_fn: str | Callable = 'MSE',
        num_domain: int = 0,
        num_boundary: int = 0,
        num_test: int = None,
        train_distribution: str = "Hammersley",
        anchors: Optional[bst.typing.ArrayLike] = None,
        exclusions=None,
        loss_weights: Sequence[float] = None,
    ):
        super().__init__(
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights
        )

        assert isinstance(geometry, DictPointGeometry), f"Expected DictPointGeometry, got {type(geometry)}"
        # geometry is a Geometry object
        self.geometry = geometry

        # PDE function
        self._pde = pde
        if pde is not None:
            assert callable(pde), f"Expected callable, got {type(pde)}"

        # initial and boundary conditions
        self.constraints = constraints if isinstance(constraints, (list, tuple)) else [constraints]
        for bc in self.constraints:
            assert isinstance(bc, ICBC), f"Expected ICBC, got {type(bc)}"
            bc.apply_geometry(self.geometry)
            bc.apply_problem(self)

        # anchors
        self.anchors = (None
                        if anchors is None else
                        jax.tree.map(lambda x: x.astype(bst.environ.dftype()), anchors))

        # solution
        if solution is not None:
            assert callable(solution), f"Expected callable, got {type(solution)}"
        self.solution = solution

        # exclusions
        self.exclusions = exclusions

        # others
        self.num_domain = num_domain
        self.num_boundary = num_boundary
        self.num_test = num_test
        self.train_distribution = train_distribution

        # training data
        self.train_x_all: Dict[str, bst.typing.ArrayLike] = None
        self.train_x_bc: Dict[str, bst.typing.ArrayLike] = None
        self.num_bcs: List[int] = None

        # these include both BC and PDE points
        self.train_x: Dict[str, bst.typing.ArrayLike] = None
        self.train_y: Dict[str, bst.typing.ArrayLike] = None
        self.test_x: Dict[str, bst.typing.ArrayLike] = None
        self.test_y: Dict[str, bst.typing.ArrayLike] = None

        # generate training data and testing data
        self.train_next_batch()
        self.test()

    def pde(self, *args, **kwargs):
        """
        Compute the PDE residual.
        """
        if self._pde is not None:
            return self._pde(*args, **kwargs)
        else:
            raise NotImplementedError("PDE is not defined.")

    def call_pde_errors(self, inputs, outputs, **kwargs):
        bcs_start = np.cumsum([0] + self.num_bcs)

        # PDE inputs and outputs, computing PDE losses
        pde_inputs = jax.tree.map(lambda x: x[bcs_start[-1]:], inputs)
        pde_outputs = jax.tree.map(lambda x: x[bcs_start[-1]:], outputs)
        pde_kwargs = jax.tree.map(lambda x: x[bcs_start[-1]:], kwargs)

        # error
        pde_errors = self.pde(pde_inputs, pde_outputs, **pde_kwargs)
        return pde_errors

    def call_bc_errors(self, loss_fns, loss_weights, inputs, outputs, **kwargs):
        bcs_start = np.cumsum([0] + self.num_bcs)
        losses = []
        for i, bc in enumerate(self.constraints):
            # ICBC inputs and outputs, computing ICBC losses
            beg, end = bcs_start[i], bcs_start[i + 1]
            icbc_inputs = jax.tree.map(lambda x: x[beg:end], inputs)
            icbc_outputs = jax.tree.map(lambda x: x[beg:end], outputs)
            icbc_kwargs = jax.tree.map(lambda x: x[beg:end], kwargs)

            # error
            error: Dict = bc.error(icbc_inputs, icbc_outputs, **icbc_kwargs)

            # loss and weights
            f_loss = loss_fns[i]
            if loss_weights is not None:
                w = loss_weights[i]
                bc_loss = jax.tree.map(lambda err: f_loss(u.math.zeros_like(err), err) * w, error)
            else:
                bc_loss = jax.tree.map(lambda err: f_loss(u.math.zeros_like(err), err), error)

            # append to losses
            losses.append({f'ibc{i}': bc_loss})
        return losses

    @utils.check_not_none('num_bcs')
    def losses(self, inputs, outputs, targets, **kwargs):
        # PDE inputs and outputs, computing PDE losses
        pde_errors = self.call_pde_errors(inputs, outputs, **kwargs)
        if not isinstance(pde_errors, (list, tuple)):
            pde_errors = [pde_errors]

        # loss functions
        if not isinstance(self.loss_fn, (list, tuple)):
            loss_fn = [self.loss_fn] * (len(pde_errors) + len(self.constraints))
        else:
            loss_fn = self.loss_fn
        if len(loss_fn) != len(pde_errors) + len(self.constraints):
            raise ValueError(f"There are {len(pde_errors) + len(self.constraints)} errors, "
                             f"but only {len(loss_fn)} losses.")

        # PDE loss
        losses = [loss_fn[i](u.math.zeros_like(error), error) for i, error in enumerate(pde_errors)]
        if self.loss_weights is not None:
            n_loss = len(losses) + len(self.constraints)
            if len(self.loss_weights) != len(losses) + len(self.constraints):
                raise ValueError(f"Expected {n_loss} weights, got {len(self.loss_weights)}. "
                                 f"There are {len(losses)} PDE losses and {len(self.constraints)} IC+BC losses.")
            del n_loss
            losses = [w * loss for w, loss in zip(self.loss_weights[:len(losses)], losses)]

        # loss of boundary or initial conditions
        bc_errors = self.call_bc_errors(
            loss_fn[len(pde_errors):],
            self.loss_weights[len(pde_errors):] if self.loss_weights is not None else None,
            inputs,
            outputs,
            **kwargs
        )
        losses.extend(bc_errors)
        return losses

    @utils.run_if_all_none("train_x", "train_y")
    def train_next_batch(self, batch_size=None):
        # Generate `self.train_x_all`
        self.train_points()

        # Generate `self.num_bcs` and `self.train_x_bc`
        self.bc_points()

        if self.pde is not None:
            # include data in boundary, initial conditions, and PDE
            if len(self.train_x_bc):
                self.train_x = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=0),
                                            self.train_x_bc,
                                            self.train_x_all)
            else:
                self.train_x = self.train_x_all

        else:
            # only include data in boundary or initial conditions
            self.train_x = self.train_x_bc

        self.train_y = self.solution(self.train_x) if self.solution is not None else None
        return self.train_x, self.train_y

    @utils.run_if_all_none("test_x", "test_y")
    def test(self):
        if self.num_test is None:
            # assign the training points to the testing points
            self.test_x = self.train_x
        else:
            # Generate `self.test_x`, resampling the test points
            self.test_x = self.test_points()

        # solution on the test points
        self.test_y = self.solution(self.test_x) if self.solution is not None else None
        return self.test_x, self.test_y

    def resample_train_points(self, pde_points=True, bc_points=True):
        """Resample the training points for PDE and/or BC."""
        if pde_points:
            self.train_x_all = None
        if bc_points:
            self.train_x_bc = None
        self.train_x, self.train_y = None, None
        self.train_next_batch()

    def add_anchors(self, anchors: bst.typing.PyTree):
        """
        Add new points for training PDE losses.

        The BC points will not be updated.
        """
        anchors = jax.tree.map(lambda x: x.astype(bst.environ.dftype()), anchors)
        if self.anchors is None:
            self.anchors = anchors
        else:
            self.anchors = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                        self.anchors,
                                        anchors)

        # include anchors in the training points
        self.train_x_all = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                        anchors,
                                        self.train_x_all)

        if self.pde is not None:
            # include data in boundary, initial conditions, and PDE
            self.train_x = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                        self.bc_points(),
                                        self.train_x_all)

        else:
            # only include data in boundary or initial conditions
            self.train_x = self.bc_points()

        # solution on the training points
        self.train_y = self.solution(self.train_x) if self.solution is not None else None

    def replace_with_anchors(self, anchors):
        """Replace the current PDE training points with anchors.

        The BC points will not be changed.
        """
        self.anchors = jax.tree.map(lambda x: x.astype(bst.environ.dftype()), anchors)
        self.train_x_all = self.anchors

        if self.pde is not None:
            # include data in boundary, initial conditions, and PDE
            self.train_x = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=-1),
                                        self.bc_points(),
                                        self.train_x_all)
        else:
            # only include data in boundary or initial conditions
            self.train_x = self.bc_points()

        # solution on the training points
        self.train_y = self.solution(self.train_x) if self.solution is not None else None

    @utils.run_if_all_none("train_x_all")
    def train_points(self):
        X = None

        # sampling points in the domain
        if self.num_domain > 0:
            if self.train_distribution == "uniform":
                X = self.geometry.uniform_points(self.num_domain, boundary=False)
            else:
                X = self.geometry.random_points(self.num_domain, random=self.train_distribution)

        # sampling points on the boundary
        if self.num_boundary > 0:
            if self.train_distribution == "uniform":
                tmp = self.geometry.uniform_boundary_points(self.num_boundary)
            else:
                tmp = self.geometry.random_boundary_points(self.num_boundary, random=self.train_distribution)
            X = (tmp
                 if X is None else
                 jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=0), X, tmp))

        # add anchors
        if self.anchors is not None:
            X = (self.anchors
                 if X is None else
                 jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=0), self.anchors, X))

        # exclude points
        if self.exclusions is not None:
            raise NotImplementedError

            # TODO: Check if this is correct
            def is_not_excluded(x):
                return not np.any([np.allclose(x, y) for y in self.exclusions])

            X = np.array(list(filter(is_not_excluded, X)))

        # save the training points
        self.train_x_all = X
        return X

    @utils.run_if_all_none("train_x_bc")
    def bc_points(self):
        """
        Generate boundary condition points.

        Returns:
            np.ndarray: The boundary condition points.
        """
        x_bcs = [bc.collocation_points(self.train_x_all) for bc in self.constraints]
        # self.num_bcs = list([len(x[self.geometry.names[0]]) for x in x_bcs])
        self.num_bcs = list([len(tuple(x.values())[0]) for x in x_bcs])
        if len(self.num_bcs):
            self.train_x_bc = jax.tree.map(lambda *x: u.math.concatenate(x, axis=0), *x_bcs)
        else:
            self.train_x_bc = dict()
        return self.train_x_bc

    def test_points(self):
        # different points from self.train_x_all
        x = self.geometry.uniform_points(self.num_test, boundary=False)

        # # different BC points from self.train_x_bc
        # x_bcs = [bc.collocation_points(x) for bc in self.constraints]
        # x_bcs = jax.tree.map(lambda *x: u.math.vstack(x), *x_bcs)

        # reuse the same BC points
        if len(self.num_bcs):
            x_bcs = self.train_x_bc
            x = jax.tree.map(lambda x_, y_: u.math.concatenate((x_, y_), axis=0), x_bcs, x)
        return x


class TimePDE(PDE):
    """Time-dependent PDE solver.

    Args:
        num_initial (int): The number of training points sampled on the initial
            location.
    """

    def __init__(
        self,
        geometry: DictPointGeometry,
        pde: Callable,
        constraints: Union[ICBC, Sequence[ICBC]],
        approximator: Optional[bst.nn.Module] = None,
        num_domain: int = 0,
        num_boundary: int = 0,
        num_initial: int = 0,
        train_distribution: str = "Hammersley",
        anchors=None,
        exclusions=None,
        solution=None,
        num_test: int = None,
        loss_fn: str | Callable = 'MSE',
        loss_weights: Sequence[float] = None,
    ):
        self.num_initial = num_initial
        super().__init__(
            geometry,
            pde,
            constraints,
            num_domain=num_domain,
            num_boundary=num_boundary,
            train_distribution=train_distribution,
            anchors=anchors,
            exclusions=exclusions,
            solution=solution,
            num_test=num_test,
            approximator=approximator,
            loss_fn=loss_fn,
            loss_weights=loss_weights,
        )

    @utils.run_if_all_none("train_x_all")
    def train_points(self):
        self.geometry: GeometryXTime

        X = super().train_points()

        if self.num_initial > 0:
            if self.train_distribution == "uniform":
                tmp = self.geometry.uniform_initial_points(self.num_initial)
            else:
                tmp = self.geometry.random_initial_points(self.num_initial, random=self.train_distribution)
            if self.exclusions is not None:
                def is_not_excluded(x):
                    return not np.any([np.allclose(x, y) for y in self.exclusions])

                tmp = np.array(list(filter(is_not_excluded, tmp)))
            X = jax.tree.map(lambda x, y: u.math.concatenate((x, y), axis=0), X, tmp)
        self.train_x_all = X
        return X
