from jax import nn
from Code.utilities import MSELoss_method
from Code.descent_methods import SGD_adam
from Code.softmax_regression import softmax

import numpy as np
import jax.numpy as jnp
from jax import grad
import jax.scipy as jsp
import jax.lax as lax

from collections import OrderedDict


def _beta_init_conv(input_shape, window_size_list, hidden_layer_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """

    n_filters = 5

    beta0 = OrderedDict()

    hidden_layer_list.append(10)

    hidden_layer_list.insert(0, (input_shape[0] )*(input_shape[1]))

    # beta0[f"conv0"] = np.random.normal(0, 1, (1, n_filters, window_size_list[0], window_size_list[0]))


    # Add random initialisation
    for i in range(0, len(window_size_list)):
        # Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"conv{i}"] = np.random.normal(0, 1, (1, window_size_list[i], window_size_list[i]))


    for i in range(1, len(hidden_layer_list)):
        # Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"W{i}"] = np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (hidden_layer_list[i - 1] + hidden_layer_list[i])),
            size=(hidden_layer_list[i - 1], hidden_layer_list[i]),
        )

        # Bias vector
        beta0[f"b{i}"] = 0 * np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (hidden_layer_list[i - 1] + hidden_layer_list[i])),
            size=(hidden_layer_list[i]),
        )





    # beta0[f"W_out"] = np.random.normal(
    #         loc=0,
    #         scale=np.sqrt(2 / ((input_shape[0])*(input_shape[1]) + 10)),
    #         size=((input_shape[0] )*(input_shape[1] ), 10),
    #     )
        
    # # Bias vector
    # beta0[f"b_out"] = 0 * np.random.normal(
    #     loc=0,
    #     scale=np.sqrt(2 / ((input_shape[0])*(input_shape[1]) + 10)),
    #     size=(10),
    # )
    return beta0


def convolutional_model_method(n_convs, hidden_activation=nn.tanh):
    return (lambda beta, X: convolutional_model(beta, X, n_convs=n_convs, hidden_activation=hidden_activation))




def convolutional_model(beta, X, n_convs, hidden_activation=nn.tanh):
    out = X.copy()
    
    for key in list(beta.keys())[:n_convs]:
        # out = lax.conv(out, beta[key], window_strides=(1, 1), padding="SAME")
        out = jsp.signal.convolve(out, beta[key], mode="full")[:, :-beta[key].shape[1]+1, :-beta[key].shape[1]+1]/(beta[key].shape[1]**2.0)
    
    out = jnp.resize(out, (out.shape[0], out.shape[1]*out.shape[2]))
    

    for key_w, key_b in zip(list(beta.keys())[n_convs::2], list(beta.keys())[n_convs + 1::2]):
         out = hidden_activation(jnp.dot(out, beta[key_w]) + beta[key_b])
    
    
    
    return softmax(out)

#a = '[:, :-beta["conv{i}"].shape[1]+1, :-beta["conv{i}"].shape[1]+1]/(beta["conv{i}"].shape[1]**2.0)'


def get_convolutional_model(beta):
    n_conv = len(beta.keys()) - 2

    func_string = "def convolutional_model(beta, X):\n\tout = X.copy()\n"

    for i in range(n_conv):
        func_string += f'\tout = jsp.signal.convolve(X, beta["conv{i}"], mode="full")\n'

    func_string += '\tout = jnp.resize(out, (out.shape[0], out.shape[1]*out.shape[2]))\n'
    func_string += '\tout = (jnp.dot(out, beta["W_out"]) + beta["b_out"])\n'
    func_string += '\treturn out\n'
    func_string += 'model = convolutional_model'
    exec(func_string, globals())

    model = convolutional_model

    return model