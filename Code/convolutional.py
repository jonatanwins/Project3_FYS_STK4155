from jax import nn
from Code.utilities import MSELoss_method
from Code.descent_methods import SGD_adam

import numpy as np
import jax.numpy as jnp
from jax import grad
import jax.scipy as jsp


def _beta_init_conv(input_shape, window_size_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """

    beta0 = {}

    extra_nodes = 0

    # Add random initialisation
    for i in range(len(window_size_list)):
        # Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"conv{i}"] = np.random.normal(0, 1, (1, window_size_list[i], window_size_list[i]))

        extra_nodes += (window_size_list[i]-1)



    extra_nodes = 0


    beta0[f"W_out"] = np.random.normal(
            loc=0,
            scale=np.sqrt(2 / ((input_shape[0] + extra_nodes - 1)*(input_shape[1] + extra_nodes - 1) + 10)),
            size=((input_shape[0] + extra_nodes)*(input_shape[1] + extra_nodes), 10),
        )
        
    # Bias vector
    beta0[f"b_out"] = 0 * np.random.normal(
        loc=0,
        scale=np.sqrt(2 / ((input_shape[0] + extra_nodes - 1)*(input_shape[1] + extra_nodes - 1) + 10)),
        size=(10),
    )
    return beta0




def convolutional_model_old(beta, X):
    out = X.copy()
    i = 0
    for key in beta.keys():
        if i >= 2:
            out = jsp.signal.convolve(X, beta[key], mode="full")[:, :-beta[key].shape[1]+1, :-beta[key].shape[1]+1]/(beta[key].shape[1]**2.0)
        i += 1
    out = jnp.resize(out, (out.shape[0], out.shape[1]*out.shape[2]))
    out = (jnp.dot(out, beta["W_out"]) + beta["b_out"])
    return out

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