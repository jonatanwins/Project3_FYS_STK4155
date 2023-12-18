from jax import nn
from Code.utilities import MSELoss_method
from Code.descent_methods import SGD_adam
from Code.softmax_regression import softmax

import numpy as np
import jax.numpy as jnp
from jax import grad
import jax.scipy as jsp
import jax.lax as lax
from flax import linen as nn_flax
from math import ceil


from collections import OrderedDict


def _beta_init_conv(input_shape, window_size_list, hidden_layer_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """


    beta0 = OrderedDict()

    hidden_layer_c = hidden_layer_list.copy()

    hidden_layer_c.append(10)

    hidden_layer_c.insert(0, ceil((input_shape[0])*(input_shape[1])*window_size_list[-1][1]/2**(2*len(window_size_list))))
    hidden_layer_c.insert(0, input_shape[0])

    for i in range(len(window_size_list)):
        hidden_layer_c[0] = ceil(hidden_layer_c[0]/2)

    hidden_layer_c[0] = (hidden_layer_c[0]**2)*window_size_list[-1][1]

    # beta0[f"conv0"] = np.random.normal(0, 1, (1, n_filters, window_size_list[0], window_size_list[0]))


    # Add random initialisation
    for i in range(0, len(window_size_list)):
        # Xavier Initialization #TODO Reference
        if i == 0:
            in_dim = 1
        else:
            in_dim = window_size_list[i-1][1]

        # Weight matrix
        beta0[f"conv{i}"] = np.random.normal(0, 1, (window_size_list[i][1], in_dim, window_size_list[i][0], window_size_list[i][0]))


    for i in range(1, len(hidden_layer_c)):
        # Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"W{i}"] = np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (hidden_layer_c[i - 1] + hidden_layer_c[i])),
            size=(hidden_layer_c[i - 1], hidden_layer_c[i]),
        )

        # Bias vector
        beta0[f"b{i}"] = 0 * np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (hidden_layer_c[i - 1] + hidden_layer_c[i])),
            size=(hidden_layer_c[i]),
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
        # out = nn.relu(jsp.signal.convolve(out, beta[key], mode="full")[:, :-beta[key].shape[1]+1, :-beta[key].shape[1]+1]/(beta[key].shape[1]**2.0))
        # print(out.shape)
        out = nn.relu(lax.conv(out, beta[key],window_strides=(1, 1), padding="SAME")/(beta[key].shape[2]**2.0))
        out = jnp.transpose(out, axes=[0, 2, 3, 1])
        out = nn_flax.max_pool(out, window_shape=(2, 2), strides=(2, 2), padding='SAME')
        out = jnp.transpose(out, axes=[0, 3, 1, 2])

    # print(out.shape)

    out = jnp.resize(out, (out.shape[0], out.shape[1]*out.shape[2]*out.shape[3]))

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