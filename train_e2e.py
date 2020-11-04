import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *

import efficientnet.model as model
from swa import SWA
from pipeline import *
from transforms import *
from utils import *
from metrics import *
from models import transformer_layer, encoder


args = argparse.ArgumentParser()
args.add_argument('--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dim', type=int, default=256)
args.add_argument('--n_heads', type=int, default=8)

# DATA
args.add_argument('--background_sounds', type=str,
                  default='/codes/generate_wavs/drone_normed_complex.pickle')
args.add_argument('--voices', type=str,
                  default='/codes/generate_wavs/voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default='/codes/generate_wavs/voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default='/codes/RDChallenge/tf_codes/sounds/noises_specs.pickle')
args.add_argument('--test_background_sounds', type=str,
                  default='/codes/generate_wavs/test_drone_normed_complex.pickle')
args.add_argument('--test_voices', type=str,
                  default='/codes/generate_wavs/test_voice_normed_complex.pickle')
args.add_argument('--test_labels', type=str,
                  default='/codes/generate_wavs/test_voice_labels_mfc.npy')
args.add_argument('--n_mels', type=int, default=128)

# TRAINING
args.add_argument('--optimizer', type=str, default='adam',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--lr_factor', type=float, default=0.7)
args.add_argument('--lr_patience', type=int, default=10)

args.add_argument('--epochs', type=int, default=500)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--n_frame', type=int, default=2048)
args.add_argument('--steps_per_epoch', type=int, default=200)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--snr', type=float, default=-15)
args.add_argument('--max_voices', type=int, default=10)
args.add_argument('--max_noises', type=int, default=5)


def minmax_log_on_mel(mel, labels=None):
    axis = tuple(range(1, len(mel.shape)))

    # MIN-MAX
    mel_max = tf.math.reduce_max(mel, axis=axis, keepdims=True)
    mel_min = tf.math.reduce_min(mel, axis=axis, keepdims=True)
    mel = (mel-mel_min) / (mel_max-mel_min+EPSILON)

    # LOG
    mel = tf.math.log(mel + EPSILON)

    if labels is not None:
        return mel, labels
    return mel


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=16, n_mask=6) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels


def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')


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


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    # strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = config.name if config.name.endswith('.h5') else config.name + '.h5'
    N_CLASSES = 30

    # with strategy.scope():
    """ MODEL """
    x = tf.keras.layers.Input(shape=(config.n_mels, config.n_frame, 2))
    model = getattr(model, config.model)(
        include_top=False,
        weights=None,
        input_tensor=x,
        backend=tf.keras.backend,
        layers=tf.keras.layers,
        models=tf.keras.models,
        utils=tf.keras.utils,
    )
    out = tf.transpose(model.output, perm=[0, 2, 1, 3])
    out = tf.keras.layers.Reshape([out.shape[-3], out.shape[-1]*out.shape[-2]])(out)
    # out = tf.keras.layers.Reshape([-1, out.shape[-1]*out.shape[-2]])(out)

    # Add Transformer Layer
    if config.n_layers > 0:
        out = tf.keras.layers.Dense(config.n_dim)(out)
        for i in range(config.n_layers):
            out = transformer_layer(config.n_dim, config.n_heads)(out)

    if config.pretrain:
        out2 = tf.keras.layers.Dense(N_CLASSES, activation='sigmoid')(out)
        model = tf.keras.models.Model(inputs=model.input, outputs=out2)
        model.load_weights(NAME)
        print('loaded pretrained model')

        for l in model.layers:
            l.trainable = False

    # Add Transformer Layer
    n_layers, n_dim, n_heads = 3, 64, 8
    out = tf.keras.layers.Dense(n_dim)(out)
    out = tf.keras.layers.Flatten()(out) # GlobalMaxPool1D()(out)
    for i in range(2):
        out = tf.keras.layers.Dense(n_dim * 8)(out)
        out = tf.keras.layers.Activation('sigmoid')(out) * out
    out = tf.keras.layers.Dense(30)(out)
    out = tf.keras.layers.Reshape((3, 10))(out)
    model = tf.keras.models.Model(inputs=model.input, outputs=out)

    if config.optimizer == 'adam':
        opt = Adam(config.lr, clipvalue=0.1) 
    elif config.optimizer == 'sgd':
        opt = SGD(config.lr, momentum=0.9)
    else:
        opt = RMSprop(config.lr, momentum=0.9)

    if config.l2 > 0:
        model = apply_kernel_regularizer(
            model, tf.keras.regularizers.l2(config.l2))
    model.compile(optimizer=opt, 
                  loss='mse', # 'binary_crossentropy',
                  metrics=[d_total, d_dir, d_cls])
    model.summary()
    

    """ DATA """
    # TRAINING DATA
    backgrounds = load_data(config.background_sounds)
    voices = load_data(config.voices)
    labels = load_data(config.labels)
    labels = np.eye(N_CLASSES, dtype='float32')[labels] # to one-hot vectors
    noises = load_data(config.noises)

    # PIPELINE
    pipeline = make_pipeline(backgrounds, 
                             voices, labels,
                             noises,
                             n_frame=config.n_frame,
                             max_voices=config.max_voices,
                             max_noises=config.max_noises,
                             n_classes=N_CLASSES,
                             snr=config.snr)
    pipeline = pipeline.map(to_class_labels)
    pipeline = pipeline.map(augment, num_parallel_calls=AUTOTUNE)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.batch(BATCH_SIZE, drop_remainder=False)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    pipeline = pipeline.prefetch(AUTOTUNE)

    # VAL
    backgrounds = load_data(config.test_background_sounds)
    voices = load_data(config.test_voices)
    labels = load_data(config.test_labels)
    labels = np.eye(N_CLASSES, dtype='float32')[labels] # to one-hot vectors
    noises = load_data(config.noises)

    val_pipe = make_pipeline(backgrounds, 
                             voices, labels,
                             noises,
                             n_frame=config.n_frame,
                             max_voices=config.max_voices,
                             max_noises=config.max_noises,
                             n_classes=N_CLASSES,
                             snr=config.snr)
    val_pipe = val_pipe.map(to_class_labels)
    val_pipe = val_pipe.map(complex_to_magphase)
    val_pipe = val_pipe.batch(BATCH_SIZE, drop_remainder=False)
    val_pipe = val_pipe.map(magphase_to_mel(config.n_mels))
    val_pipe = val_pipe.map(minmax_log_on_mel)
    val_pipe = val_pipe.prefetch(AUTOTUNE)

    """ TRAINING """
    from train_frame import custom_scheduler
    callbacks = [
        CSVLogger(NAME.replace('.h5', '.log'),
                  append=True),
        LearningRateScheduler(custom_scheduler(n_dim*8, TOTAL_EPOCH/10)),
        SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
        ModelCheckpoint(NAME,
                        monitor='d_total',
                        mode='max',
                        save_best_only=True),
        TerminateOnNaN()
    ]

    model.fit(pipeline,
              epochs=TOTAL_EPOCH,
              batch_size=BATCH_SIZE,
              steps_per_epoch=config.steps_per_epoch,
              validation_data=val_pipe,
              validation_steps=16,
              callbacks=callbacks)

    # TODO : BN 
    model.save(NAME.replace('.h5', '_SWA.h5'))
