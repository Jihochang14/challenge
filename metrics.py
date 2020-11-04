import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def D_direction(y_true, y_pred):
    kernel = np.array([0.05, 0.1, 0.7, 0.1, 0.05], dtype='float32')
    kernel = kernel.reshape(-1, 1, 1)

    wma_true = tf.nn.conv1d(tf.expand_dims(y_true, -1), kernel, 1, 'SAME')
    wma_pred = tf.nn.conv1d(tf.expand_dims(y_pred, -1), kernel, 1, 'SAME')

    return tf.reduce_sum(tf.square(wma_true - wma_pred), axis=(1, 2))


def D_class(y_true, y_pred):
    return tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)


def d_dir(y_true, y_pred):
    # y_true: [None, 3, 10, 1]
    # y_pred: [None, 3, 10, 16]
    y_pred = tf.math.round(y_pred)
    return D_direction(tf.reduce_sum(y_true, axis=1),
                       tf.reduce_sum(y_pred, axis=1))


def d_cls(y_true, y_pred):
    # y_true: [None, 3, 10, 1]
    # y_pred: [None, 3, 10, 16]
    y_pred = tf.math.round(y_pred)
    return D_class(tf.reduce_sum(y_true, axis=-1),
                   tf.reduce_sum(y_pred, axis=-1))


def d_total(y_true, y_pred):
    return 0.8 * d_dir(y_true, y_pred) + 0.2 * d_cls(y_true, y_pred)


def d_dir_zero(y_true, y_pred):
    # y_true: [None, 3, 10, 1]
    # y_pred: [None, 3, 10, 16]
    y_pred = K.zeros_like(y_pred)
    return D_direction(tf.reduce_sum(y_true, axis=1),
                       tf.reduce_sum(y_pred, axis=1))


def d_cls_zero(y_true, y_pred):
    # y_true: [None, 3, 10, 1]
    # y_pred: [None, 3, 10, 16]
    y_pred = K.zeros_like(y_pred)
    return D_class(tf.reduce_sum(y_true, axis=-1),
                   tf.reduce_sum(y_pred, axis=-1))


def d_total_zero(y_true, y_pred):
    return 0.8 * d_dir_zero(y_true, y_pred) + 0.2 * d_cls_zero(y_true, y_pred)
