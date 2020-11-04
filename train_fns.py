import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K

import efficientnet.model as model
from swa import SWA
from pipeline import *
from transforms import *
from utils import *
from metrics import *
from models import transformer_layer, encoder


args = argparse.ArgumentParser()
args.add_argument('-n', '--name', type=str, required=True)
args.add_argument('--model', type=str, default='EfficientNetB0')
args.add_argument('--pretrain', type=bool, default=False)
args.add_argument('--freeze-extractor', type=bool, default=False)
args.add_argument('--frame-wise', type=bool, default=False)
args.add_argument('--sample-wise', type=bool, default=True)
args.add_argument('--n_layers', type=int, default=0)
args.add_argument('--n_dims', type=int, default=64)
args.add_argument('--n_heads', type=int, default=8)

# DATA
base_path = '/root/volume/challenge/generate_wavs/codes/'
args.add_argument('--background_sounds', type=str,
                  default=base_path+'drone_normed_complex.pickle')
args.add_argument('--voices', type=str,
                  default=base_path+'voice_normed_complex.pickle')
args.add_argument('--labels', type=str,
                  default=base_path+'voice_labels_mfc.npy')
args.add_argument('--noises', type=str,
                  default=base_path+'noises_specs.pickle')
args.add_argument('--test_background_sounds', type=str,
                  default=base_path+'test_drone_normed_complex.pickle')
args.add_argument('--test_voices', type=str,
                  default=base_path+'test_voice_normed_complex.pickle')
args.add_argument('--test_labels', type=str,
                  default=base_path+'test_voice_labels_mfc.npy')
args.add_argument('--n_mels', type=int, default=128)
args.add_argument('--n_classes', type=int, default=30)

# TRAINING
args.add_argument('--optimizer', type=str, default='sgd',
                                 choices=['adam', 'sgd', 'rmsprop'])
args.add_argument('--lr', type=float, default=0.0001)
args.add_argument('--lr_factor', type=float, default=0.7)
args.add_argument('--lr_patience', type=int, default=10)

args.add_argument('--epochs', type=int, default=300)
args.add_argument('--batch_size', type=int, default=2)
args.add_argument('--n_frame', type=int, default=2000)
args.add_argument('--steps_per_epoch', type=int, default=20)
args.add_argument('--l2', type=float, default=1e-6)

# AUGMENTATION
args.add_argument('--alpha', type=float, default=0.75)
args.add_argument('--snr', type=float, default=-15)
args.add_argument('--max_voices', type=int, default=10)
args.add_argument('--max_noises', type=int, default=5)


def augment(specs, labels, time_axis=1, freq_axis=0):
    specs = mask(specs, axis=time_axis, max_mask_size=16, n_mask=6) # time
    specs = mask(specs, axis=freq_axis, max_mask_size=32) # freq
    specs = random_shift(specs, axis=freq_axis, width=8)
    return specs, labels


def preprocess_labels(x, y1, y2=None):
    # preprocess y
    for i in range(5):
        y1 = tf.nn.max_pool1d(y1, 2, strides=2, padding='SAME')

    if y2 is not None:
        for i in range(5):
            y2 = tf.nn.max_pool1d(y2, 2, strides=2, padding='SAME')
        return x, y1, y2
    return x, y1


def load_data(path):
    if path.endswith('.pickle'):
        return pickle.load(open(path, 'rb'))
    elif path.endswith('.npy'):
        return np.load(path)
    else:
        raise ValueError('invalid file format')


def make_dataset(config, training=True):
    # Load required datasets
    if training:
        backgrounds = load_data(config.background_sounds)
        voices = load_data(config.voices)
        labels = load_data(config.labels)
    else:
        backgrounds = load_data(config.test_background_sounds)
        voices = load_data(config.test_voices)
        labels = load_data(config.test_labels)
    
    labels = np.eye(N_CLASSES, dtype='float32')[labels] # to one-hot vectors
    noises = load_data(config.noises)

    # Make pipeline and process the pipeline
    pipeline = make_pipeline(backgrounds, 
                             voices, labels,
                             noises,
                             n_frame=config.n_frame,
                             max_voices=config.max_voices,
                             max_noises=config.max_noises,
                             n_classes=config.n_classes,
                             snr=config.snr)
    
    if config.frame_wise and not config.sample_wise:
        pipeline = pipeline.map(to_frame_labels)
    elif not config.frame_wise and config.sample_wise:
        pipeline = pipeline.map(to_class_labels)
    elif config.frame_wise and config.sample_wise:
        pipeline = pipeline.map(to_both_labels)

    if training:
        pipeline = pipeline.map(augment)
    pipeline = pipeline.batch(config.batch_size, drop_remainder=False)
    pipeline = pipeline.map(complex_to_magphase)
    pipeline = pipeline.map(magphase_to_mel(config.n_mels))
    pipeline = pipeline.map(minmax_log_on_mel)
    
    if config.frame_wise and not config.sample_wise:
        pipeline = pipeline.map(preprocess_labels)
    elif config.frame_wise and config.sample_wise:
        pipeline = pipeline.map(preprocess_labels)
    
    return pipeline.prefetch(AUTOTUNE)


if __name__ == "__main__":
    config = args.parse_args()
    print(config)

    # strategy = tf.distribute.MirroredStrategy()

    TOTAL_EPOCH = config.epochs
    BATCH_SIZE = config.batch_size
    NAME = 'logs/' + (config.name if config.name.endswith('.h5') else config.name + '.h5')
    N_CLASSES = config.n_classes

    # with strategy.scope():
    """ MODEL """
    x = tf.keras.layers.Input(shape=(config.n_mels, None, 2))
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
    out = tf.keras.layers.Reshape([-1, out.shape[-1] * out.shape[-2]])(out)

    # Add Transformer Layer
    if config.n_layers > 0:
        out = tf.keras.layers.Dense(config.n_dims)(out)
        for i in range(config.n_layers):
            out = transformer_layer(config.n_dims, config.n_heads)(out)

    if config.frame_wise:
        out = tf.keras.layers.Dense(N_CLASSES, activation='sigmoid')(out)
        model1 = tf.keras.models.Model(inputs=model.input, outputs=out)

    if config.sample_wise:
        # Add Transformer Layer
        out = tf.keras.layers.Dense(config.n_dims)(out)
        for i in range(1):
            out = transformer_layer(config.n_dims, config.n_heads)(out)
        out = out[:, -1, :]
        out = tf.keras.layers.Dense(30, activation="relu")(out)
        out = tf.keras.layers.Reshape((3, 10))(out)
        model2 = tf.keras.models.Model(inputs=model.input, outputs=out)

    # load
    if config.pretrain:
        model1.load_weights(NAME)
        print('loaded pretrained model')
    # freeze
    if config.freeze_extractor:
        for l in model1.layers:
            l.trainable = False

    # optimizer
    if config.optimizer == 'adam':
        opt = Adam(config.lr, clipvalue=0.1) 
    elif config.optimizer == 'sgd':
        opt = SGD(config.lr, momentum=0.9)
    else:
        opt = RMSprop(config.lr, momentum=0.9)

    # weight decay
    if config.l2 > 0:
        model = apply_kernel_regularizer(
            model, tf.keras.regularizers.l2(config.l2))

    # compile
    if config.frame_wise and not config.sample_wise:
        model = model1
        model.compile(optimizer=opt, 
                      loss=focal_loss, # 'binary_crossentropy',
                      metrics=['AUC'])
        monitor = 'val_auc'
    elif not config.frame_wise and config.sample_wise:
        model = model2
        model.compile(optimizer=opt, 
                      loss='mean_absolute_error', # 'binary_crossentropy',
                      metrics=[d_total, d_dir, d_cls, d_total_zero, d_dir_zero, d_cls_zero])
        monitor = 'd_total'
    elif config.frame_wise and config.sample_wise:
        model = model2
        model.compile(optimizer=opt,
                      loss=[focal_loss, 'mse'],
                      metrics=[['accuracy', 'AUC'], [d_total, d_dir, d_cls]])
        monitor = 'd_total'
    else:
        assert False, 'Wrong configuration!'
    model.summary()

    """ DATA """
    train_set = make_dataset(config, training=True)
    #test_set = make_dataset(config, training=False)

    """ TRAINING """
    callbacks = [
        CSVLogger(NAME.replace('.h5', '.log'),
                  append=True),
        #LearningRateScheduler(custom_scheduler(config.n_dims*8, TOTAL_EPOCH/10)),
        SWA(start_epoch=TOTAL_EPOCH//2, swa_freq=2),
        ModelCheckpoint(NAME,
                        monitor=monitor,
                        mode='max',
                        save_best_only=True),
        TerminateOnNaN()
    ]

    model.fit(train_set,
              epochs=TOTAL_EPOCH,
              batch_size=BATCH_SIZE,
              steps_per_epoch=config.steps_per_epoch,
              validation_data=None,
              validation_steps=24,
              callbacks=callbacks)

    # TODO : BN 
    model.save(NAME.replace('.h5', '_SWA.h5'))
