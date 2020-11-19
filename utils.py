import numpy as np
import os
import pickle
import h5py
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.saving import hdf5_format

from train_density_full import Spec_Filter

EPSILON = 1e-8


''' 
UTILS FOR FRAMES AND WINDOWS 
'''
def seq_to_windows(seq, 
                   window, 
                   skip=1,
                   padding=True, 
                   **kwargs):
    '''
    INPUT:
        seq: np.ndarray
        window: array of indices
            ex) [-3, -1, 0, 1, 3]
        skip: int
        padding: bool
        **kwargs: params for np.pad

    OUTPUT:
        windows: [n_windows, window_size, ...]
    '''
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window) + 1
    windows = window[np.newaxis, :] \
            + np.arange(0, len(seq), skip)[:, np.newaxis]
    if padding:
        seq = np.pad(
            seq,
            [[win_size//2, (win_size-1)//2]] + [[0, 0]]*len(seq.shape[1:]),
            mode='constant',
            **kwargs)

    return np.take(seq, windows, axis=0)


def windows_to_seq(windows,
                   window,
                   skip=1):
    '''
    INPUT:
        windows: np.ndarray (n_windows, window_size, ...)
        window: array of indices
        skip: int

    OUTPUT:
        seq
    '''
    n_window = windows.shape[0]
    window = np.array(window - np.min(window)).astype(np.int32)
    win_size = max(window)

    seq_len = (n_window-1)*skip + 1
    seq = np.zeros([seq_len, *windows.shape[2:]], dtype=windows.dtype)
    count = np.zeros(seq_len)

    for i, w in enumerate(window):
        indices = np.arange(n_window)*skip - win_size//2 + w
        select = np.logical_and(0 <= indices, indices < seq_len)
        seq[indices[select]] += windows[select, i]
        count[indices[select]] += 1
    
    seq = seq / (count + EPSILON)
    return seq


'''
DATASET
'''
def list_to_generator(dataset: list):
    def _gen():
        if isinstance(dataset, tuple):
            for z in zip(*dataset):
                yield z
        else:
            for data in dataset:
                yield data
    return _gen


def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')


'''
MODEL
'''
def apply_kernel_regularizer(model, kernel_regularizer):
    model = tf.keras.models.clone_model(model)
    layer_types = (tf.keras.layers.Dense, tf.keras.layers.Conv2D, Spec_Filter)
    for layer in model.layers:
        if isinstance(layer, layer_types):
            layer.kernel_regularizer = kernel_regularizer

    model = tf.keras.models.clone_model(model)
    return model


'''
Load
'''
#/root/.keras/models/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5
#/root/.keras/models/efficientnet-b0_noisy-student_notop.h5

def load_imagenet(model, scale=0, train_type='imagenet'):
    filepath = '/root/.keras/models/'
    filepath += f'efficientnet-b{scale}'
    if train_type == 'imagenet':
        filepath += '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    elif train_type == 'noisy-student':
        filepath += '_noisy-student_notop.h5'
    

    if not model._is_graph_network and not model.built:
        raise ValueError(
            'Unable to load weights saved in HDF5 format into a subclassed '
            'Model which has not created its variables yet. Call the Model '
            'first, then load the weights.')
    #model._assert_weights_created()
    
    with h5py.File(filepath, 'r') as f:
        if train_type == 'imagenet':
            f = f['model_weights']

        layers = dict()
        for layer in model.layers:
            weights = hdf5_format._legacy_weights(layer)
            if weights:
                layers[layer.name] = weights

        weight_value_tuples = []
        for l_name in f.keys():
            if 'stem' in l_name:
                continue
            
            if l_name not in layers.keys():
                continue
            
            symbolic_weights = layers[l_name]

            w_name = [x.name.split('/')[-1] for x in symbolic_weights]
            weight_values = []
            for n in w_name:
                assert n in f[l_name][l_name], f'Layer: {l_name} should have the same kinds of weights between model and loaded weights.'
                weight_values.append(f[l_name][l_name][n])

            weight_value_tuples += zip(symbolic_weights, weight_values)
            #print([x.shape for x in symbolic_weights])
            #print([x.shape for x in weight_values])
        K.batch_set_value(weight_value_tuples)
