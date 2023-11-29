import jax.numpy as jnp
from jax import jit
import numpy as np
from matplotlib import pyplot as plt

##################################################
##################### Evaluation tools
##################################################
def predict(model, beta, X):
    """
    Returns the predicted number for each sample in X,
    We choose the number with largest "probability"
    """

    # Use the neural network
    y = model(beta, X)

    # Find best guess index
    predictions = jnp.eye(y.shape[1])[jnp.argmax(y, axis=1)]

    return predictions

# Define new accuracy function
def accuracy_func(model, beta, X, y):
    """
    ACCURACY = percentage guessed correctly    
    """

    predictions = predict(model, beta, X)

    # Do not div by 2, since errors are not countet twice
    return 1 - jnp.sum(predictions != y) / (2*y.shape[0])


def accuracy_func_method(model):
    return lambda beta, X, y : accuracy_func(model, beta, X, y)

##################################################
##################### Loss functions
##################################################
def MSELoss(y, y_pred):
    """MSE loss of prediction array.

    Args:
        y (ndarray): Target values
        y_pred (ndarray): Predicted values

    Returns:
        float: MSE loss
    """
    return jnp.sum(jnp.power(y - y_pred, 2)) / y.shape[0]

def cross_entropy_loss(y_pred, y_i):

    # Safe log clipping
    y_pred_new = jnp.clip(y_pred, 0.0000001, 0.9999999)

    # Divide by number of samples
    return - jnp.sum(y_i*jnp.log(y_pred_new)) / y_i.shape[0]

##################### JAX grad compatible
def ridge_term(beta):
    s = 0.0
    for key in beta.keys():
        s += jnp.sum(jnp.power(beta[key], 2))
    return s

def MSELoss_method(model, lam=0):
    if lam == 0:
        return lambda beta, X, y: MSELoss(model(beta, X), y)
    else:
        return lambda beta, X, y: MSELoss(model(beta, X), y) + lam*ridge_term(beta)

def cross_entropy_loss_method(model, lam=0):
    if lam == 0:
        return lambda beta, X, y: cross_entropy_loss(model(beta, X), y)
    else:
        return lambda beta, X, y: cross_entropy_loss(model(beta, X), y) + lam*ridge_term(beta)

