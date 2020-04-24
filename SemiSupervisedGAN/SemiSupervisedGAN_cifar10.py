from utils import generate_and_save_images, get_random_z
import tensorflow as tf
from tensorflow.keras import layers

import os
import sys
import numpy as np
from datetime import datetime

import wandb
wandb.init(project='tf2-gans', name='SemiSupervisedGAN')

# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

# hyper-parameters
ITERATION = 100000
Z_DIM = 100
BATCH_SIZE = 512
BUFFER_SIZE = 60000
D_LR = 0.0004
G_LR = 0.0004
IMAGE_SHAPE = (32, 32, 3)
RANDOM_SEED = 42
D_NUM_CLASSES = 10 + 1  # category + fake
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

test_z = tf.random.normal([36, Z_DIM])


def make_discriminaor(input_shape):  # define discriminator
    return tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                      input_shape=input_shape),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(D_NUM_CLASSES, activation='softmax')
    ])


def make_generator(input_shape):  # define generator
    return tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(
            1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(
            2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False, activation='tanh')
    ])


def get_loss_fn():  # define loss function
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, real_labels, fake_logits):
        fake_labels = tf.one_hot(
            tf.fill(real_labels.shape, 10), D_NUM_CLASSES)  # 10 is fake category
        real_labels = tf.one_hot(real_labels, D_NUM_CLASSES)

        real_loss = criterion(real_labels, real_logits)
        fake_loss = criterion(fake_labels, fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        fake_labels = tf.one_hot(tf.random.uniform(
            (BATCH_SIZE,), 0, 10, dtype=tf.int32), D_NUM_CLASSES)  # 0 ~ 9 is real category
        return criterion(fake_labels, fake_logits)

    return d_loss_fn, g_loss_fn


sys.path.append('..')

# data load & preprocessing
(train_x, train_y), (_, _) = tf.keras.datasets.cifar10.load_data()
train_x = train_x.reshape(train_x.shape[0], 32, 32, 3).astype('float32')
train_x = (train_x - 127.5) / 127.5
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
)

# generator & discriminator
G = make_generator((Z_DIM,))
D = make_discriminaor(IMAGE_SHAPE)

# optimizer
g_optim = tf.keras.optimizers.Adam(G_LR, beta_1=0.5, beta_2=0.999)
d_optim = tf.keras.optimizers.Adam(D_LR, beta_1=0.5, beta_2=0.999)

# loss function
d_loss_fn, g_loss_fn = get_loss_fn()


@tf.function
def train_step(real_images, labels):
    z = get_random_z(Z_DIM, BATCH_SIZE)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = G(z, training=True)

        fake_logits = D(fake_images, training=True)
        real_logits = D(real_images, training=True)

        d_loss = d_loss_fn(real_logits, labels, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss, d_loss


def train(ds, log_freq=20, test_freq=1000):  # training loop
    ds = iter(ds)
    for step in range(ITERATION):
        images, labels = next(ds)
        g_loss, d_loss = train_step(images, labels)

        g_loss_metrics(g_loss)
        d_loss_metrics(d_loss)
        total_loss_metrics(g_loss + d_loss)

        if step % log_freq == 0:
            template = '[{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
            print(template.format(step, ITERATION, d_loss_metrics.result(),
                                  g_loss_metrics.result(), total_loss_metrics.result()))
            wandb.log({
                'G_Loss': g_loss_metrics.result(),
                'D_Loss': d_loss_metrics.result(),
                'Total_Loss': total_loss_metrics.result()
            })
            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()

        if step % test_freq == 0:
            # generate result images
            generate_and_save_images(
                G, step, test_z, IMAGE_SHAPE, name='ssgan_cifar10', max_step=ITERATION)


train(train_ds)
