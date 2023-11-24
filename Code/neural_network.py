from jax import nn
from Code.utilities import MSELoss_method
from Code.descent_methods import SGD_adam

import numpy as np
import jax.numpy as jnp
from jax import grad, lax


def _beta_init2(layer_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """

    beta0 = {}

    # Add random initialisation
    for i in range(1, len(layer_list)):
        # Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"W{i}"] = np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (layer_list[i - 1] + layer_list[i])),
            size=(layer_list[i - 1], layer_list[i]),
        )

        # Bias vector
        beta0[f"b{i}"] = 0 * np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (layer_list[i - 1] + layer_list[i])),
            size=(layer_list[i]),
        )

    return beta0

def _beta_init(layer_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """

    beta0 = {}

    layers = []

    # Add random initialisation
    for i in range(1, len(layer_list)):
        # Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"W{i}"] = np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (layer_list[i - 1] + layer_list[i])),
            size=(layer_list[i - 1], layer_list[i]),
        )

        layers.append([np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (layer_list[i - 1] + layer_list[i])),
            size=(layer_list[i - 1], layer_list[i]),
        ), 0 * np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (layer_list[i - 1] + layer_list[i])),
            size=(layer_list[i]),
        )])

        # Bias vector
        beta0[f"b{i}"] = 0 * np.random.normal(
            loc=0,
            scale=np.sqrt(2 / (layer_list[i - 1] + layer_list[i])),
            size=(layer_list[i]),
        )

    beta0["layers"] = layers

    return beta0



def apply_layer(out, layer, activation_func = nn.sigmoid):
    # print(out.shape)
    # print(layer[0].shape)
    # print(jnp.add(jnp.dot(out, layer[0]), layer[1]))
    out_temp = activation_func(jnp.add(jnp.dot(out, layer[0]), layer[1]))
    return out_temp



def neural_flexible(beta, X, layer_func, number_of_layers, activation_func=nn.sigmoid, output_activation=(lambda x: x)):

    # out = activation_func(jnp.dot(X.copy(), beta["W1"]) + beta["b1"])
    out = X.copy()

    n = len(beta.keys())//2

    layers = jnp.zeros(shape=(number_of_layers - 1, beta["W1"].shape[1], beta["W1"].shape[1]))
    # print(layers.shape)

    biases = jnp.zeros(shape=(number_of_layers - 1, beta["b1"].shape[0]))
    # print(biases.shape)


    # for i in range(2,n):
    #     layers = layers.at[i, :, :].set(beta[f"W{i}"])
    #     biases = biases.at[i, :].set(beta[f"b{i}"])

    for layer in beta["layers"]:
        out = layer_func(out, layer)


    # outs = [out]

    # for i in range(2, number_of_layers):
    #     out.append(layer_func(out, (beta[f"W{i}"], beta[f"b{i}"])))
    


    # out_new, record = lax.scan(layer_func, out, layers_comb)
    # print(beta[f"W{n}"].shape)

    # out_new_2 = output_activation(jnp.dot(out[-1], beta[f"W{number_of_layers}"]) + beta[f"b{number_of_layers}"])

    return out
    

def get_flexible_network(beta, activation=nn.sigmoid, output_activation=(lambda x: x)):
    layer_func = lambda out, layer: apply_layer(out, layer, activation)
    n = len(beta.keys())//2
    return lambda beta, X: neural_flexible(beta, X, layer_func=layer_func, number_of_layers=n, activation_func=activation, output_activation=output_activation)



def get_neural_network_model(
    num_hidden, activation=nn.sigmoid, output_activation=(lambda x: x)
):
    """
    Due to issues with for loops and JAX, we have implemented functions for 0-6 layers

    input:
        beta:
        X:
        activation: activation for the hidden layers
        output_activation: function to shape output
    """
    if num_hidden == 0:
        return lambda beta, X: neural_0(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 1:
        return lambda beta, X: neural_1(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 2:
        return lambda beta, X: neural_2(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 3:
        return lambda beta, X: neural_3(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 4:
        return lambda beta, X: neural_4(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 5:
        return lambda beta, X: neural_5(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 6:
        return lambda beta, X: neural_6(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 7:
        return lambda beta, X: neural_7(
            beta, X, activation=activation, output_activation=output_activation
        )
    elif num_hidden == 8:
        return lambda beta, X: neural_8(
            beta, X, activation=activation, output_activation=output_activation
        )
    else:
        raise ValueError("num hidden must be 0, 1, ..., 6")


def neural_0(beta, X, activation, output_activation):
    out = output_activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    return out


def neural_1(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = output_activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    return out


def neural_2(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = output_activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    return out


def neural_3(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = output_activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    return out


def neural_4(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = output_activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    return out


def neural_5(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = output_activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    return out


def neural_6(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    out = output_activation(jnp.dot(out, beta[f"W7"]) + beta[f"b7"])
    return out


def neural_7(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    out = activation(jnp.dot(out, beta[f"W7"]) + beta[f"b7"])
    out = output_activation(jnp.dot(out, beta[f"W8"]) + beta[f"b8"])
    return out


def neural_8(beta, X, activation, output_activation):
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    out = activation(jnp.dot(out, beta[f"W7"]) + beta[f"b7"])
    out = activation(jnp.dot(out, beta[f"W8"]) + beta[f"b8"])
    out = output_activation(jnp.dot(out, beta[f"W9"]) + beta[f"b9"])
    return out
