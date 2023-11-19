import jax.numpy as jnp
from jax import grad
import numpy as np
from matplotlib import pyplot as plt


# 1d
def feature_matrix(x, num_features):
    """
    x: array with x values
    num_features: the degree of polynomial 1+x+...+x^p

    returns:
    X: The feature matrix,a 2D numpy array with a column for each feature
    """

    x = x.squeeze()

    return jnp.array([x**i for i in range(num_features)]).T


def random_partition(X, y, n_batches):
    batch_size = int(y.shape[0] / n_batches)
    batches = []

    for i in range(n_batches):
        index = list(range(i * batch_size, (i + 1) * batch_size))
        batches.append((X[index, :], y[index]))

    return batches, batch_size


def train_test_split(X, Y, percentage, test_index=None):
    """
    X: Feature matrix
    Y: Label vector(size=(n, 1))
    Percentage: How much of the dataset should be used as a test set.
    """

    n = X.shape[0]
    if test_index is None:
        test_index = np.random.choice(n, round(n * percentage), replace=False)
    test_X = X[test_index]
    test_Y = Y[test_index]

    train_X = np.delete(X, test_index, axis=0)
    train_Y = np.delete(Y, test_index, axis=0)

    return train_X, train_Y, test_X, test_Y, test_index


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


# Defines loss function. Can use these with JAX grad
def MSELoss_method(model):
    return lambda beta, X, y: MSELoss(model(beta, X), y)


def _ridge_term(beta):
    s = 0.0
    for key in beta.keys():
        s += jnp.sum(jnp.power(beta[key], 2))
    return s


def ridge_loss_method(model, lam):
    return lambda beta, X, y: MSELoss(model(beta, X), y) + _ridge_term(beta) * lam


##########################################################
##################### Gradients for OLS and RIDGE
##########################################################


#### OLS Analytic
def _OLS_grad(beta, X, y, model):
    n = y.shape[0]
    return 2 * (np.dot(X.T, (model(beta, X) - y))) / n


def OLS_train_analgrad(model):
    return lambda beta, X, y: {"b": _OLS_grad(beta, X, y, model)}


#### OLS JAX
def OLS_train_autograd(model):
    return grad(MSELoss_method(model))


#### Ridge analytic
def _ridge_grad(beta, X, y, model, lam):
    mse_grad = _OLS_grad(beta, X, y, model)
    l2_grad = 2 * lam * beta["b"]
    return mse_grad + l2_grad


def ridge_train_analgrad(model, lam):
    return lambda beta, X, y: {"b": _ridge_grad(beta, X, y, model, lam)}


#### Ridge automatic
def ridge_train_autograd(model, lam):
    return grad(ridge_loss_method(model, lam))


##########################################################
##################### Some plotting ######################
##########################################################


def plot_test_results(test_loss_list, train_loss_list, num_batches, ylabel="MSE"):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))  # 1 row, 2 columns

    # Subplot 1
    axs[0].plot(test_loss_list, label="test")
    axs[0].plot(train_loss_list, label="train")
    axs[0].set_xlabel("Training step")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title("Over all sub-epochs")
    axs[0].legend()

    # Subplot 2
    axs[1].plot(test_loss_list[::num_batches], label="test")
    axs[1].plot(train_loss_list[::num_batches], label="train")
    axs[1].set_xlabel("Training step")
    axs[1].set_title("End of epoch error")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
