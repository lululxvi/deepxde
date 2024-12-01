"""Model extension for ZCS support"""

from .. import gradients as grad
from .. import optimizers
from ..backend import tf, torch, paddle # noqa
from ..model import Model as BaseModel


class Model(BaseModel):
    """Derived `Model` class for ZCS support."""

    def __init__(self, data, net):
        super().__init__(data, net)
        # store ZCS parameters, sent to user for PDE calculation
        self.zcs_parameters = None

    def _compile_tensorflow_compat_v1(self, lr, loss_fn, decay):
        """tensorflow.compat.v1"""
        raise NotImplementedError(
            "ZCS is not implemented for backend tensorflow.compat.v1"
        )

    def _compile_tensorflow(self, lr, loss_fn, decay):
        """tensorflow"""
        super()._compile_tensorflow(lr, loss_fn, decay)

        def process_inputs_zcs(inputs):
            # get inputs
            branch_inputs, trunk_inputs = inputs

            # convert to tensors with grad disabled
            branch_inputs = tf.constant(branch_inputs)
            trunk_inputs = tf.constant(trunk_inputs)

            # create ZCS scalars
            n_dim_crds = trunk_inputs.shape[1]
            zcs_scalars = [tf.Variable(0.0, trainable=True) for _ in range(n_dim_crds)]

            # add ZCS to truck inputs
            # from now until loss must be taped
            with tf.GradientTape(
                persistent=True, watch_accessed_variables=False
            ) as tape:
                for z in zcs_scalars:
                    tape.watch(z)
                zcs_vector = tf.stack(zcs_scalars)
                trunk_inputs = trunk_inputs + zcs_vector[None, :]

            # return inputs and ZCS parameters
            return (branch_inputs, trunk_inputs), {"leaves": zcs_scalars, "tape": tape}

        def outputs_losses_zcs(training, inputs, targets, auxiliary_vars, losses_fn):
            # aux
            self.net.auxiliary_vars = auxiliary_vars

            # inputs
            inputs, self.zcs_parameters = process_inputs_zcs(inputs)

            # forward and loss must be taped
            with self.zcs_parameters["tape"]:
                # forward
                outputs_ = self.net(inputs, training=training)

                # losses
                if targets is not None:
                    targets = tf.constant(targets)
                losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
                if not isinstance(losses, list):
                    losses = [losses]

            # regularization
            if self.net.regularizer is not None:
                losses += [tf.math.reduce_sum(self.net.losses)]
            losses = tf.convert_to_tensor(losses)

            # weighted
            if self.loss_weights is not None:
                losses *= self.loss_weights
            return outputs_, losses

        def outputs_losses_train_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        # base class used a temporary variable
        self.opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)  # noqa

        def train_step_zcs(inputs, targets, auxiliary_vars):
            with tf.GradientTape() as tape:
                losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            self.opt.apply_gradients(zip(grads, trainable_variables))

        def train_step_tfp_zcs(
            inputs, targets, auxiliary_vars, previous_optimizer_results=None
        ):
            def build_loss():
                losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
                return tf.math.reduce_sum(losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            return self.opt(trainable_variables, build_loss, previous_optimizer_results)

        # overwrite callables
        self.outputs_losses_train = outputs_losses_train_zcs
        self.outputs_losses_test = outputs_losses_test_zcs
        self.train_step = (
            train_step_zcs
            if not optimizers.is_external_optimizer(self.opt_name)  # noqa
            else train_step_tfp_zcs
        )

    def _compile_pytorch(self, lr, loss_fn, decay):
        """pytorch"""
        super()._compile_pytorch(lr, loss_fn, decay)

        def process_inputs_zcs(inputs):
            # get inputs
            branch_inputs, trunk_inputs = inputs

            # convert to tensors with grad disabled
            branch_inputs = torch.as_tensor(branch_inputs)
            trunk_inputs = torch.as_tensor(trunk_inputs)

            # create ZCS scalars
            n_dim_crds = trunk_inputs.shape[1]
            zcs_scalars = [
                torch.as_tensor(0.0).requires_grad_() for _ in range(n_dim_crds)
            ]

            # add ZCS to truck inputs
            zcs_vector = torch.stack(zcs_scalars)
            trunk_inputs = trunk_inputs + zcs_vector[None, :]

            # return inputs and ZCS scalars
            return (branch_inputs, trunk_inputs), {"leaves": zcs_scalars}

        def outputs_losses_zcs(training, inputs, targets, auxiliary_vars, losses_fn):
            # aux
            self.net.auxiliary_vars = None
            if auxiliary_vars is not None:
                self.net.auxiliary_vars = torch.as_tensor(auxiliary_vars)

            # inputs
            inputs, self.zcs_parameters = process_inputs_zcs(inputs)

            # forward
            self.net.train(mode=training)
            outputs_ = self.net(inputs)

            # losses
            if targets is not None:
                targets = torch.as_tensor(targets)
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            losses = torch.stack(losses)

            # TODO: regularization

            # weighted
            if self.loss_weights is not None:
                losses *= torch.as_tensor(self.loss_weights)

            # clear cached gradients (actually not used with ZCS)
            grad.clear()
            return outputs_, losses

        def outputs_losses_train_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        def train_step_zcs(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
                total_loss = torch.sum(losses)
                self.opt.zero_grad()
                total_loss.backward()
                return total_loss

            self.opt.step(closure)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # overwrite callables
        self.outputs_losses_train = outputs_losses_train_zcs
        self.outputs_losses_test = outputs_losses_test_zcs
        self.train_step = train_step_zcs

    def _compile_jax(self, lr, loss_fn, decay):
        """jax"""
        raise NotImplementedError("ZCS is not implemented for backend jax")

    def _compile_paddle(self, lr, loss_fn, decay):
        """paddle"""
        super()._compile_paddle(lr, loss_fn, decay)

        def process_inputs_zcs(inputs):
            # get inputs
            branch_inputs, trunk_inputs = inputs

            # convert to tensors with grad disabled
            branch_inputs = paddle.to_tensor(branch_inputs, stop_gradient=True)  # noqa
            trunk_inputs = paddle.to_tensor(trunk_inputs, stop_gradient=True)  # noqa

            # create ZCS scalars
            n_dim_crds = trunk_inputs.shape[1]
            zcs_scalars = [
                paddle.to_tensor(0.0, stop_gradient=False)  # noqa
                for _ in range(n_dim_crds)
            ]

            # add ZCS to truck inputs
            zcs_vector = paddle.concat(
                [paddle.tile(z, 1) for z in zcs_scalars], axis=0
            )  # noqa
            trunk_inputs = trunk_inputs + zcs_vector[None, :]

            # return inputs and ZCS scalars
            return (branch_inputs, trunk_inputs), {"leaves": zcs_scalars}

        def outputs_losses_zcs(training, inputs, targets, auxiliary_vars, losses_fn):
            # aux
            self.net.auxiliary_vars = auxiliary_vars

            # inputs
            inputs, self.zcs_parameters = process_inputs_zcs(inputs)

            # forward
            if training:
                self.net.train()
            else:
                self.net.eval()
            outputs_ = self.net(inputs)

            # losses
            if targets is not None:
                targets = paddle.to_tensor(targets)  # noqa
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            losses = paddle.stack(losses, axis=0)

            # TODO: regularization

            # weighted
            if self.loss_weights is not None:
                losses *= paddle.to_tensor(self.loss_weights)  # noqa

            # clear cached gradients (actually not used with ZCS)
            grad.clear()
            return outputs_, losses

        def outputs_losses_train_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        def outputs_losses_test_zcs(inputs, targets, auxiliary_vars):
            return outputs_losses_zcs(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        def train_step_zcs(inputs, targets, auxiliary_vars):
            losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
            total_loss = paddle.sum(losses)  # noqa
            total_loss.backward()
            self.opt.step()
            self.opt.clear_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        def train_step_lbfgs_zcs(inputs, targets, auxiliary_vars):
            def closure():
                losses = outputs_losses_train_zcs(inputs, targets, auxiliary_vars)[1]
                total_loss = paddle.sum(losses)  # noqa
                self.opt.clear_grad()
                total_loss.backward()
                return total_loss

            self.opt.step(closure)

        # overwrite callables
        self.outputs_losses_train = outputs_losses_train_zcs
        self.outputs_losses_test = outputs_losses_test_zcs
        self.train_step = (
            train_step_zcs
            if not optimizers.is_external_optimizer(self.opt_name)  # noqa
            else train_step_lbfgs_zcs
        )
