from Code.neural_network import _beta_init
import jax.numpy as jnp

##################################################
##################### Softmax tools
##################################################
def softmax_beta_init(input_shape, output_shape):
    """
    Input:
        input_shape: num features
        output_shape: num classes
    Returns:
        xavier initialised weights
    """

    return _beta_init([input_shape, output_shape])

def softmax(out):
    # Softmax to get probabilities TODO: MULIG SLICINGA FUCKER OPP
    return jnp.exp(out) / jnp.sum(jnp.exp(out), axis=-1)[:, None]


def softmax_model(beta, X):
    # Weighted sum of input + bias
    out = jnp.dot(X.copy(), beta[f"W1"]) + beta[f"b1"]
    return softmax(out)