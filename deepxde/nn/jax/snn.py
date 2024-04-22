from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers
from ...utils import list_handler

class SPINN(NN):
    layer_sizes: Any
    activation: Any
    kernel_initializer: Any
    mlp: str = 'mlp'
    pos_enc: int = 0

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        self.in_dim = self.layer_sizes[0]  # input dimension
        self.r = self.layer_sizes[-2]  # rank of the approximated tensor
        self.out_dim = self.layer_sizes[-1]  # output dimension
        self.init = initializers.get(self.kernel_initializer)
        self.features = self.layer_sizes[1:-2]


    @nn.compact
    def __call__(self, inputs, training=False):

        if self._input_transform is not None:
            x = self._input_transform(x)

        list_inputs = []
        for i in range(self.in_dim):
            if inputs.ndim == 1:
                list_inputs.append(inputs[i:i+1])
            else:
                list_inputs.append(inputs[:, i:i+1])

        if self.in_dim == 1:
            raise ValueError("Input dimension must be greater than 1")
        elif self.in_dim == 2:
            outputs = self.SPINN2d(list_inputs)
        elif self.in_dim == 3:
            outputs = self.SPINN3d(list_inputs)
        elif self.in_dim == 4:
            outputs = self.SPINN4d(list_inputs)
        else:
            outputs = self.SPINNnd(list_inputs)

        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)

        return outputs

    def SPINN2d(self, inputs):
        # inputs = [x, y]
        flat_inputs = inputs[0].ndim == 1
        if flat_inputs:
            inputs = [inputs_elem.reshape(-1, 1) for inputs_elem in inputs]
        outputs, pred = [], []
        if self.mlp == 'mlp':
            for X in inputs:
                for fs in self.features:
                    X = nn.Dense(fs, kernel_init=self.init)(X)
                    X = nn.activation.tanh(X)
                X = nn.Dense(self.r*self.out_dim, kernel_init=self.init)(X)
                outputs += [X]
        else:
            for X in inputs:
                U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                for fs in self.features:
                    Z = nn.Dense(fs, kernel_init=self.init)(H)
                    Z = nn.activation.tanh(Z)
                    H = (jnp.ones_like(Z)-Z)*U + Z*V
                H = nn.Dense(self.r*self.out_dim, kernel_init=self.init)(H)
                outputs += [H]

        for i in range(self.out_dim):
            pred += [
                jnp.dot(
                    outputs[0][:,self.r * i : self.r * (i + 1)],
                    outputs[-1][:,self.r * i : self.r * (i + 1)].T,
                ).reshape(-1,1)
            ]
        
        if len(pred) == 1:
            # 1-dimensional output
            return pred[0].reshape(-1) if flat_inputs else pred[0]
        else:
            return jnp.stack(pred, axis=1).squeeze() if flat_inputs else jnp.stack(pred, axis=1) 

    def SPINN3d(self, inputs):
        '''
        inputs: input factorized coordinates
        outputs: feature output of each body network
        xy: intermediate tensor for feature merge btw. x and y axis
        pred: final model prediction (e.g. for 2d output, pred=[u, v])
        '''
        [x, y, z] = inputs
        if self.pos_enc != 0:
            # positional encoding only to spatial coordinates
            freq = jnp.expand_dims(jnp.arange(1, self.pos_enc+1, 1), 0)
            y = jnp.concatenate((jnp.ones((y.shape[0], 1)), jnp.sin(y@freq), jnp.cos(y@freq)), 1)
            z = jnp.concatenate((jnp.ones((z.shape[0], 1)), jnp.sin(z@freq), jnp.cos(z@freq)), 1)

            # causal PINN version (also on time axis)
            #  freq_x = jnp.expand_dims(jnp.power(10.0, jnp.arange(0, 3)), 0)
            # x = x@freq_x
            
        inputs, outputs, xy, pred = [x, y, z], [], [], []
        init = nn.initializers.glorot_normal()

        if self.mlp == 'mlp':
            for X in inputs:
                for fs in self.features:
                    X = nn.Dense(fs, kernel_init=self.init)(X)
                    X = nn.activation.tanh(X)
                X = nn.Dense(self.r*self.out_dim, kernel_init=self.init)(X)
                outputs += [jnp.transpose(X, (1, 0))]

        elif self.mlp == 'modified_mlp':
            for X in inputs:
                U = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                V = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                H = nn.activation.tanh(nn.Dense(self.features[0], kernel_init=self.init)(X))
                for fs in self.features:
                    Z = nn.Dense(fs, kernel_init=self.init)(H)
                    Z = nn.activation.tanh(Z)
                    H = (jnp.ones_like(Z)-Z)*U + Z*V
                H = nn.Dense(self.r*self.out_dim, kernel_init=self.init)(H)
                outputs += [jnp.transpose(H, (1, 0))]
        
        for i in range(self.out_dim):
            xy += [jnp.einsum('fx, fy->fxy', outputs[0][self.r*i:self.r*(i+1)], outputs[1][self.r*i:self.r*(i+1)])]
            pred += [jnp.einsum('fxy, fz->xyz', xy[i], outputs[-1][self.r*i:self.r*(i+1)]).ravel()]

        if len(pred) == 1:
            # 1-dimensional output
            return pred[0]
        else:
            # n-dimensional output
            return jnp.stack(pred, axis=1)

    def SPINN4d(self, inputs):
        outputs, tx, txy, pred = [], [], [], []
        # inputs = [t, x, y, z]
        init = nn.initializers.glorot_normal()
        for X in inputs:
            for fs in self.features:
                X = nn.Dense(fs, kernel_init=self.init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.r*self.out_dim, kernel_init=self.init)(X)
            outputs += [jnp.transpose(X, (1, 0))]

        for i in range(self.out_dim):
            tx += [jnp.einsum('ft, fx->ftx', 
            outputs[0][self.r*i:self.r*(i+1)], 
            outputs[1][self.r*i:self.r*(i+1)])]

            txy += [jnp.einsum('ftx, fy->ftxy', 
            tx[i], 
            outputs[2][self.r*i:self.r*(i+1)])]

            pred += [jnp.einsum('ftxy, fz->txyz', 
            txy[i], 
            outputs[3][self.r*i:self.r*(i+1)]).ravel()]


        if len(pred) == 1:
            # 1-dimensional output
            return pred[0]
        else:
            # n-dimensional output
            return jnp.stack(pred, axis=1)

    def SPINNnd(self, inputs):
        # inputs = [t, *x]
        dim = len(inputs)
        # inputs, outputs, tx, txy, pred = [t, x, y, z], [], [], [], []
        # inputs, outputs = [t, x, y, z], []
        outputs = []
        init = nn.initializers.glorot_normal()
        for X in inputs:
            for fs in self.features[:-1]:
                X = nn.Dense(fs, kernel_init=self.init)(X)
                X = nn.activation.tanh(X)
            X = nn.Dense(self.r, kernel_init=self.init)(X)
            outputs += [jnp.transpose(X, (1, 0))]

        # einsum(a,b->c)
        a = 'za'
        b = 'zb'
        c = 'zab'
        pred = jnp.einsum(f'{a}, {b}->{c}', outputs[0], outputs[1])
        for i in range(dim-2):
            a = c
            b = f'z{chr(97+i+2)}'
            c = c+chr(97+i+2)
            if i == dim-3:
                c = c[1:]
            pred = jnp.einsum(f'{a}, {b}->{c}', pred, outputs[i+2]).ravel()
            # pred = jnp.einsum('fab, fc->fabc', pred, outputs[i+2])

        return pred