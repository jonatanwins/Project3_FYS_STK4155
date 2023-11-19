import jax.numpy as jnp
from jax import grad, jit
import pandas as pd
import numpy as np


### Logistic regression is a special case of the neural network where we have no hidden layers and the activation is sigmoid.
# model = jit(get_neural_network_model(0 ,activation=None, output_activation=nn.sigmoid))


def log_loss(y_pred, y_i):
    # Clip for save logarithm and reshape
    y_new_pred = jnp.clip(jnp.reshape(y_pred, (-1, 1)), 0.0000001, 0.9999999)
    y_i = jnp.reshape(y_i, (-1, 1))

    return (
        jnp.sum(-y_i * jnp.log(y_new_pred) - (1 - y_i) * jnp.log(1 - y_new_pred))
        / y_i.shape[0]
    )


def logistic_loss_func(model):
    return lambda beta, X, y: log_loss(model(beta, X), y)


def logistic_grad(model):
    return grad(logistic_loss_func(model))


def import_breast_cancer(filename="../Code/Data/breast-cancer-wisconsin.data"):
    """
    default filename assumes file is run from one folder deep from one folder outside Code ...
    """

    header = [
        "id",
        "thickness",
        "uni_cell_s",
        "uni_cell_sh",
        "marg_adh",
        "single_epithel",
        "bare_nuc",
        "bland_chromatin",
        "normal_nuc",
        "mitoses",
        "target",
    ]
    data = pd.read_csv(filename, names=header)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna()
    data["target"] = data["target"] == 4
    y = np.array(data["target"], dtype=int)
    X = np.array(data.to_numpy()[:, 1:-1], dtype=jnp.float64)
    return X, y


def true_positive_threshold(y_pred, y_i, th):
    y_pred_bool = jnp.squeeze(y_pred >= th)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum(y_i_squeezed * (y_pred_bool)) / jnp.count_nonzero(y_i_squeezed)


def false_positive_threshold(y_pred, y_i, th):
    y_pred_bool = jnp.squeeze(y_pred >= th)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum((1 - y_i_squeezed) * y_pred_bool) / (
        jnp.count_nonzero(1 - y_i_squeezed)
    )


def accuracy(y_pred, y_i):
    y_pred_bool = jnp.squeeze(y_pred >= 0.5)
    y_i_squeezed = jnp.squeeze(y_i)
    return jnp.sum(1 - jnp.abs(y_i_squeezed - y_pred_bool)) / y_i.shape[0]


def true_pos_variable_func(model, th):
    return lambda beta, X, y: true_positive_threshold(model(beta, X), y, th)


def false_pos_variable_func(model, th):
    return lambda beta, X, y: false_positive_threshold(model(beta, X), y, th)


def accuracy_func(model):
    return lambda beta, X, y: accuracy(model(beta, X), y)


def loss_func_creator(model, loss_compute):
    return lambda beta, X, y: loss_compute(model(beta, X), y)


def ridge_term(beta):
    s = 0.0
    for key in beta.keys():
        s += jnp.sum(jnp.power(beta[key], 2))
    return s


def log_loss_ridge(model, lam):
    log_loss_func = logistic_loss_func(model=model)
    return lambda beta, X, y: jnp.add(log_loss_func(beta, X, y), lam * ridge_term(beta))
