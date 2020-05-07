import tensorflow as tf
from tensorflow.keras import layers

import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# metrics setting
g_loss_metrics = tf.metrics.Mean(name='g_loss')
d_loss_metrics = tf.metrics.Mean(name='d_loss')
total_loss_metrics = tf.metrics.Mean(name='total_loss')

# hyper-parameters
ITERATION = 10000
Z_DIM = 100
BATCH_SIZE = 512
BUFFER_SIZE = 60000
IMAGE_SIZE = 28*28
D_LR = 0.0004
G_LR = 0.0004
IMAGE_SHAPE = (28, 28, 1)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

test_z = tf.random.normal([36, Z_DIM])


def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)


# define discriminator
def make_discriminaor(input_shape):
    return tf.keras.Sequential([
        layers.Input(IMAGE_SHAPE),
        layers.Flatten(),
        layers.Dense(256,  activation=None, input_shape=input_shape),
        layers.LeakyReLU(0.2),
        layers.Dense(256,  activation=None),
        layers.LeakyReLU(0.2),
        layers.Dense(1, activation='sigmoid')
    ])


# define generator
def make_generator(input_shape):
    return tf.keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(784, activation='tanh'),
        layers.Reshape(IMAGE_SHAPE)
    ])


# define loss function
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        return -tf.reduce_mean(tf.math.log(real_logits + 1e-10) + tf.math.log(1. - fake_logits + 1e-10))

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(tf.math.log(fake_logits + 1e-10))

    return d_loss_fn, g_loss_fn


# data load & preprocessing
(train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_x = (train_x - 127.5) / 127.5
train_ds = (
    tf.data.Dataset.from_tensor_slices(train_x)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
)

# generator & discriminator
G = make_generator((Z_DIM,))
D = make_discriminaor((IMAGE_SIZE,))

# optimizer
g_optim = tf.keras.optimizers.Adam(G_LR)
d_optim = tf.keras.optimizers.Adam(D_LR)

# loss function
d_loss_fn, g_loss_fn = get_loss_fn()


@tf.function
def train_step(real_images):
    z = get_random_z(Z_DIM, BATCH_SIZE)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_images = G(z, training=True)

        fake_logits = D(fake_images, training=True)
        real_logits = D(real_images, training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)

    d_gradients = d_tape.gradient(d_loss, D.trainable_variables)
    g_gradients = g_tape.gradient(g_loss, G.trainable_variables)

    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))
    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))

    return g_loss, d_loss


# training loop
def train(ds, log_freq=20):
    ds = iter(ds)
    for step in range(ITERATION):
        images = next(ds)
        g_loss, d_loss = train_step(images)

        g_loss_metrics(g_loss)
        d_loss_metrics(d_loss)
        total_loss_metrics(g_loss + d_loss)

        if step % log_freq == 0:
            template = '[{}/{}] D_loss={:.5f} G_loss={:.5f} Total_loss={:.5f}'
            print(template.format(step, ITERATION, d_loss_metrics.result(),
                                  g_loss_metrics.result(), total_loss_metrics.result()))
            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()


if __name__ == "__main__":
    train(train_ds)
