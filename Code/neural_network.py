from Code.activation_functions import sigmoid
from Code.utilities import MSELoss_method
from Code.descent_methods import SGD_adam

import numpy as np
import jax.numpy as jnp
from jax import grad


def _beta_init(layer_list):
    """
    layer list, eg [2, 10, 1] for 2 input, 10 hidden neurons and 1 output
    """

    beta0 = {}

    # Add random initialisation
    for i in range(1, len(layer_list)):
        #Xavier Initialization #TODO Reference

        # Weight matrix
        beta0[f"W{i}"] = np.random.normal(loc=0, scale=np.sqrt(2/(layer_list[i-1] + layer_list[i])), size=(layer_list[i - 1], layer_list[i]))
        
        # Bias vector
        beta0[f"b{i}"] = 0*np.random.normal(loc=0, scale=np.sqrt(2/(layer_list[i-1] + layer_list[i])), size=(layer_list[i]))

    return beta0


def get_neural_network_model(num_hidden ,activation=sigmoid, output_activation = (lambda x : x)):

    """
        Due to issues with for loops and JAX, we have implemented functions for 0-6 layers

        input:
            beta: 
            X: 
            activation: activation for the hidden layers
            output_activation: function to shape output
    """        
    if num_hidden == 0:
        return lambda beta, X: neural_0(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 1:
        return lambda beta, X: neural_1(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 2:
        return lambda beta, X: neural_2(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 3:
        return lambda beta, X: neural_3(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 4:
        return lambda beta, X: neural_4(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 5:
        return lambda beta, X: neural_5(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 6:
        return lambda beta, X: neural_6(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 7:
        return lambda beta, X: neural_7(beta, X, activation=activation, output_activation=output_activation)
    elif num_hidden == 8:
        return lambda beta, X: neural_8(beta, X, activation=activation, output_activation=output_activation)
    else:
        raise ValueError("num hidden must be 0, 1, ..., 6")
        


def neural_0(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = output_activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    return out


def neural_1(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = output_activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    return out

def neural_2(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = output_activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    return out

def neural_3(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = output_activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    return out

def neural_4(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = output_activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    return out

def neural_5(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = output_activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    return out

def neural_6(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    out = output_activation(jnp.dot(out, beta[f"W7"]) + beta[f"b7"])
    return out


def neural_7(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
    out = activation(jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"])
    out = activation(jnp.dot(out, beta[f"W2"]) + beta[f"b2"])
    out = activation(jnp.dot(out, beta[f"W3"]) + beta[f"b3"])
    out = activation(jnp.dot(out, beta[f"W4"]) + beta[f"b4"])
    out = activation(jnp.dot(out, beta[f"W5"]) + beta[f"b5"])
    out = activation(jnp.dot(out, beta[f"W6"]) + beta[f"b6"])
    out = activation(jnp.dot(out, beta[f"W7"]) + beta[f"b7"])
    out = output_activation(jnp.dot(out, beta[f"W8"]) + beta[f"b8"])
    return out


def neural_8(beta, X, activation=sigmoid, output_activation = (lambda x: x)):
    
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