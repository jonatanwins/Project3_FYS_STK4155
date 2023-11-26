import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm


# Used to pick batches
def partition(X, y, batch_size):
    n_batches = int(y.shape[0] / batch_size)
    batches = []

    for i in range(n_batches):
        index = list(range(i * batch_size, (i + 1) * batch_size))
        batches.append((X[index, :], y[index]))

    return batches, n_batches

########################################################################################################################################
########################################################################################################################################
### General descent interface, specialised with various step methods in functions below (ONLY ADAM THIS TIME)
########################################################################################################################################
########################################################################################################################################


def _SGD_general(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    init_func,
    step_func,
    beta0: dict,
    n_epochs=50,
    batch_size=32,
    test_loss_func=None,
    gamma=0.0,
):
    # Get parameter keys
    keys = beta0.keys()

    # Initialise result storage
    result = {}
    if test_loss_func is not None:
        if type(test_loss_func) is list:
            result["train_loss_list"] = [
                [test_func(beta0, X_train, y_train)] for test_func in test_loss_func
            ]
            result["test_loss_list"] = [
                [test_func(beta0, X_test, y_test)] for test_func in test_loss_func
            ]
        else:
            result["train_loss_list"] = [test_loss_func(beta0, X_train, y_train)]
            result["test_loss_list"] = [test_loss_func(beta0, X_test, y_test)]

    # Initialise step
    v = {}
    for key in keys:
        v[key] = jnp.zeros_like(beta0[key])

    # Partition in batches
    batches, n_batches = partition(X_train, y_train, batch_size)

    # Store current beta
    beta_current = beta0.copy()

    # Perform training
    for epoch in tqdm(range(n_epochs)):

        # Accumulation variables
        tools = init_func(epoch, gamma, v)

        for i in tqdm(range(n_batches), leave=False):
            
            # Draw a batch and compute gradients for this sub-epoch
            X_b, y_b = batches[np.random.randint(n_batches)]
            # Divide by batch_size to get avg contribution from training samples
            gradients = grad_method(beta_current, X_b, y_b)
            for key in gradients.keys():
                gradients[key] = gradients[key] / batch_size

            # Perform a step with desired method
            beta_current, tools = step_func(beta_current, tools, gradients)

            if test_loss_func is not None:
                if type(test_loss_func) is list:
                    for i, test_func in enumerate(test_loss_func):
                        result["train_loss_list"][i].append(
                            test_func(beta_current, X_train, y_train)
                        )
                        result["test_loss_list"][i].append(
                            test_func(beta_current, X_test, y_test)
                        )
                else:
                    result["train_loss_list"].append(
                        test_loss_func(beta_current, X_train, y_train)
                    )
                    result["test_loss_list"].append(
                        test_loss_func(beta_current, X_test, y_test)
                    )

        gamma = tools["gamma"]
        v = tools["v"]

    # Add betas to result
    result["beta_final"] = beta_current

    return result

############################
####### Adam
############################
def init_adam(lr, weights, beta1, beta2, delta):
    tools = {
        "eta": lr,
        "s": {},
        "r": {},
        "beta1": beta1,
        "beta2": beta2,
        "delta": delta,
    }

    # Reset accumulation variables
    for key in weights.keys():
        tools["s"][key] = 0
        tools["r"][key] = 0

    return lambda epoch, gamma, v: tools | {"epoch": epoch, "gamma": gamma, "v": v}


def step_adam(beta_prev, adam_variables, gradients):
    new_beta = {}

    for key in beta_prev.keys():
        # Accumulate and compute firsr and second term
        adam_variables["s"][key] = (
            adam_variables["beta1"] * adam_variables["s"][key]
            + (1 - adam_variables["beta1"]) * gradients[key]
        )
        adam_variables["r"][key] = (
            adam_variables["beta2"] * adam_variables["r"][key]
            + (1 - adam_variables["beta2"]) * gradients[key] * gradients[key]
        )

        first_term = adam_variables["s"][key] / (
            1 - adam_variables["beta1"] ** (adam_variables["epoch"] + 1)
        )
        second_term = adam_variables["r"][key] / (
            1 - adam_variables["beta2"] ** (adam_variables["epoch"] + 1)
        )

        # Adam scaling
        update = (
            adam_variables["eta"]
            * first_term
            / (jnp.sqrt(second_term) + adam_variables["delta"])
        )  # safe division with delta

        # Perform step, if gamma != 0 it is done with momentum...
        adam_variables["v"][key] = (
            adam_variables["gamma"] * adam_variables["v"][key] + update
        )
        new_beta[key] = beta_prev[key] - adam_variables["v"][key]

    return new_beta, adam_variables


def SGD_adam(
    X_train,
    y_train,
    X_test,
    y_test,
    grad_method,
    beta0: dict,
    n_epochs: int = 50,
    batch_size: int = 32,
    test_loss_func=None,
    lr: float = 0.01,
    gamma: float = 0.0,
    delta: float = 1e-8,
    beta1: float = 0.9,
    beta2: float = 0.99,
):
    init_func = init_adam(lr, beta0, beta1, beta2, delta)

    return _SGD_general(
        X_train,
        y_train,
        X_test,
        y_test,
        grad_method,
        init_func,
        step_adam,
        beta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        test_loss_func=test_loss_func,
        gamma=gamma,
    )
