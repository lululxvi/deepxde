# Correct `model.py`
## 1. Correct `_train_sgd(self, iterations, display_every):`
> Original
```
    def _train_sgd(self, iterations, display_every):
        for i in range(iterations):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break
```
> modified
```
    def _train_sgd(self, iterations, display_every):
        for i in range(iterations):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
			### Correct this one
            self._train_step(
                self.train_state.X_train,
                self.train_state.y_train,
                self.train_state.train_aux_vars, self.loss_weights
            )
			###

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == iterations:
                self._test()

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break
```
## 2. Correct `_train_step(self, inputs, targets, auxiliary_vars):`
> Original
```
    def _train_step(self, inputs, targets, auxiliary_vars, loss_weights=Nones):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(True, inputs, targets, auxiliary_vars)
            self.sess.run(self.train_step, feed_dict=feed_dict)
        elif backend_name in ["tensorflow", "paddle"]:
            self.train_step(inputs, targets, auxiliary_vars)
        elif backend_name == "pytorch":
            self.train_step(inputs, targets, auxiliary_vars)
        elif backend_name == "jax":
            # TODO: auxiliary_vars
            self.params, self.opt_state = self.train_step(
                self.params, self.opt_state, inputs, targetss
            )
            self.net.params, self.external_trainable_variables = self.params
```
> Modified
```
    def _train_step(self, inputs, targets, auxiliary_vars, loss_weights=None):
        if backend_name == "tensorflow.compat.v1":
            feed_dict = self.net.feed_dict(True, inputs, targets, auxiliary_vars)
            self.sess.run(self.train_step, feed_dict=feed_dict)
        elif backend_name in ["tensorflow", "paddle"]:
			### Correct this one
            self.train_step(inputs, targets, auxiliary_vars, loss_weights)
			###
        elif backend_name == "pytorch":
            self.train_step(inputs, targets, auxiliary_vars)
        elif backend_name == "jax":
            # TODO: auxiliary_vars
            self.params, self.opt_state = self.train_step(
                self.params, self.opt_state, inputs, targets
            )
            self.net.params, self.external_trainable_variables = self.params
```
## 3. Correct `train_step(inputs, targets, auxiliary_vars):`
> Original
```
        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))
```
> Modified
```
        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars, loss_weights=None):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, targets, auxiliary_vars, loss_weights)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            opt.apply_gradients(zip(grads, trainable_variables))
```

## 4. Correct `def outputs_losses_train(inputs, targets, auxiliary_vars, loss_weights=None):`
> Original
```

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            ) 
```
> Modified		
```
        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, targets, auxiliary_vars, loss_weights=None):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train, loss_weights
            ) 	
```
## 5. Correct `def outputs_losses_test(inputs, targets, auxiliary_vars, loss_weights=None):`
> Original
```
        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )
```
> Modified
```
        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, targets, auxiliary_vars, loss_weights=None):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test, loss_weights
            )
```
## 6. Correct `def compile()` 
> Original
```self.loss_weights = loss_weights```
> Modified
```self.loss_weights = tf.convert_to_tensor(loss_weights, dtype=config.default_float())```

## 7. Correct `_outputs_losses(self, training, inputs, targets, auxiliary_vars, loss_weights)`
> original
```
if backend_name == "tensorflow":
            outs = outputs_losses(inputs, targets, auxiliary_vars, loss_weights)
```
> Modified
```
if backend_name == "tensorflow":
            outs = outputs_losses(inputs, targets, auxiliary_vars)
```

##. 8. Correct `def _test(self):`
> Original
```
        # TODO Now only print the training loss in rank 0. The correct way is to print the average training loss of all ranks.
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars
        )
```
> Modified
```
        # TODO Now only print the training loss in rank 0. The correct way is to print the average training loss of all ranks.
        (
            self.train_state.y_pred_train,
            self.train_state.loss_train,
        ) = self._outputs_losses(
            True,
            self.train_state.X_train,
            self.train_state.y_train,
            self.train_state.train_aux_vars,
            self.loss_weights
        )
        self.train_state.y_pred_test, self.train_state.loss_test = self._outputs_losses(
            False,
            self.train_state.X_test,
            self.train_state.y_test,
            self.train_state.test_aux_vars,
            self.loss_weights
        )
```
# Add function to `callbacks.py`
## 1. Add `PrintLossWeight(Callback):`
```
class PrintLossWeight(Callback):
    def __init__(self, period):
        super().__init__()
        '''

        '''
        self.period = period
        self.initial_loss_weights = None
        self.current_loss_weights = None

    def on_epoch_begin(self):
        if self.model.current_epoch == 0:
            self.initial_loss_weights = self.model.loss_weights.numpy().tolist()
        else:
            self.current_loss_weights = self.model.loss_weights.numpy().tolist()
        
        if self.model.current_epoch % self.period  == 0:
            print('Initial loss weights:', self.initial_loss_weights)
            print('Current loss weights:', self.current_loss_weights)
```
## 2. Add `ManualDynamicLossWeight`
```
class ManualDynamicLossWeight(Callback):
    def __init__(self, epoch, value, idx):
        super().__init__()
        '''

        '''
        self.epoch = epoch
        self.value = value
        self.idx = idx

    def on_epoch_begin(self):
        import tensorflow as tf
        if self.model.current_epoch == self.epoch:
            current_loss_weights = self.model.loss_weights.numpy()
            current_loss_weights[self.idx] = self.value
            self.model.loss_weights = tf.convert_to_tensor(current_loss_weights,dtype=config.default_float())
```