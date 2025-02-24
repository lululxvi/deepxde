"""PDE operator extensions for ZCS support"""

import numpy as np

from .. import backend as bkd
from ..data import PDEOperatorCartesianProd as BasePDEOperatorCartesianProd


class PDEOperatorCartesianProd(BasePDEOperatorCartesianProd):
    """Derived `PDEOperatorCartesianProd` class for ZCS support."""

    def _losses(self, outputs, loss_fn, inputs, model, num_func, aux):
        # PDE
        f = []
        if self.pde.pde is not None:
            f = self.pde.pde(model.zcs_parameters, outputs, model.net.auxiliary_vars)
            if not isinstance(f, (list, tuple)):
                f = [f]
        bcs_start = np.cumsum([0] + self.pde.num_bcs)
        error_f = [fi[:, bcs_start[-1] :] for fi in f]
        losses = [loss_fn(bkd.zeros_like(error), error) for error in error_f]  # noqa

        # BC
        for k, bc in enumerate(self.pde.bcs):
            beg, end = bcs_start[k], bcs_start[k + 1]
            error_k = []
            # NOTE: this loop over functions can also be avoided if we implement collective ic/bc
            for i in range(num_func):
                output_i = outputs[i]
                if bkd.ndim(output_i) == 1:  # noqa
                    output_i = output_i[:, None]
                error_ki = bc.error(
                    self.train_x[1],
                    inputs[1],
                    output_i,
                    beg,
                    end,
                    aux_var=model.net.auxiliary_vars[i][:, None],
                )
                error_k.append(error_ki)
            error_k = bkd.stack(error_k, axis=0)  # noqa
            loss_k = loss_fn(bkd.zeros_like(error_k), error_k)  # noqa
            losses.append(loss_k)
        return losses
