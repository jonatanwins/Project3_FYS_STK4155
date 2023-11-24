import jax.numpy as jnp
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
    predictions = jnp.array([jnp.argmax(y_sample) for y_sample in y])

    return predictions

# Define new accuracy function
def accuracy_func(model, beta, X, y):
    """
    ACCURACY = percentage guessed correctly    
    """
    # Find indeces corresponding to ground truth
    predictions_gt = np.array([np.argmax(y_sample) for y_sample in y])
    predictions    = predict(model, beta, X)

    # return 1-jnp.mean(jnp.abs(predictions_gt-predictions))
    return float(np.sum(predictions_gt == predictions) / predictions.shape[0])

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
