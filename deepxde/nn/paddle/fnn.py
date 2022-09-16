import paddle
import numpy as np
from .nn import NN
from .. import activations
from .. import initializers


class FNN(NN):
    """Fully-connected neural network."""
    
    def __init__(self, layer_sizes, activation, kernel_initializer, w_array):
        super().__init__()
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")

        self.linears = paddle.nn.LayerList()
        for i in range(1, len(layer_sizes)):
            weight_attr_ = paddle.ParamAttr(initializer = paddle.nn.initializer.Assign(w_array[i-1]))
            self.linears.append(paddle.nn.Linear(layer_sizes[i - 1], layer_sizes[i],weight_attr=weight_attr_))
            
            # initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)
        
        # debug info
        if paddle.in_dynamic_mode():
            import os
            f = open('paddle_param_dygraph.log','ab')
            for linear in self.linears:
                np.savetxt(f,linear.weight.numpy().reshape(1,-1),delimiter=",")
                np.savetxt(f,linear.bias.numpy().reshape(1,-1),delimiter=",")
            f.close()
        # debug info end

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
