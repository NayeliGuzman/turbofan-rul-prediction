import tensorflow as tf
import numpy as np

def make_asymmetric_mse(alpha=1.0, beta=2.0):
    def loss(y_true, y_pred):
        error = y_pred - y_true
        under_loss = alpha * tf.square(tf.minimum(error, 0))
        over_loss = beta * tf.square(tf.maximum(error, 0))
        return tf.reduce_mean(under_loss + over_loss)
    loss.__name__ = f"asymmetric_mse_a{alpha}_b{beta}"
    return loss

# S-Score function
def nasa_s_score(y_true, y_pred, alpha=13.0, beta=10.0):
    error = y_pred - y_true
    score = np.where(error < 0,
                     np.exp(-error / alpha) - 1,
                     np.exp(error / beta) - 1)
    return np.sum(score)
