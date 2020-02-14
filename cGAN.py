import tensorflow as tf
from tensorflow.keras import layers

import os
import numpy as np
from datetime import datetime
from utils import generate_and_save_images, get_random_z
from functools import partial

# tensorboard setting
log_dir = 'logs/gan/' + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

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
IMAGE_SHAPE = (28, 28, 1)
RANDOM_SEED = 42
NUM_CLASSES = 10  # 10 is number of MNIST clategory

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

test_z = tf.random.normal([100, Z_DIM])
test_labels = tf.concat([tf.fill([10, ], x) for x in range(10)], axis=0)


# define discriminator
class Discriminator(tf.keras.Model):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.image_shape = image_shape
        self.label_embedding = tf.keras.Sequential([
            layers.Dense(image_shape[0]*image_shape[1],
                         input_shape=(NUM_CLASSES,)),
            layers.Reshape((image_shape[0], image_shape[1], 1))
        ])
        self.conv1 = tf.keras.Sequential([
            # IMAGE_SHAPE[2] + 1 is image channels + label condition
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                          input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2] + 1)),
            layers.LeakyReLU(),
            layers.Dropout(0.3)
        ])
        self.conv2 = tf.keras.Sequential([
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3)
        ])
        self.out = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1)
        ])

    def call(self, images, labels, training=False):
        labels = tf.cast(tf.one_hot(labels, NUM_CLASSES), dtype=tf.float32)
        embedded_label = self.label_embedding(labels)
        x = tf.concat([images, embedded_label], axis=3)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return self.out(x, training=training)


# define generator
class Generator(tf.keras.Model):
    def __init__(self, z_shape):
        super(Generator, self).__init__()
        self.input_layer = tf.keras.Sequential([
            layers.Dense(7*7*256, use_bias=False,
                         input_shape=(z_shape[0] + NUM_CLASSES,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Reshape((7, 7, 256))
        ])
        self.conv_tp_1 = tf.keras.Sequential([
            layers.Conv2DTranspose(128, (5, 5), strides=(
                1, 1), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])
        self.conv_tp_2 = tf.keras.Sequential([
            layers.Conv2DTranspose(64, (5, 5), strides=(
                2, 2), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU()
        ])
        self.out = layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False, activation='tanh')

    def call(self, z, labels, training=False):
        labels = tf.cast(tf.one_hot(labels, NUM_CLASSES), dtype=tf.float32)
        x = tf.concat([z, labels], axis=1)
        x = self.input_layer(x, training=training)
        x = self.conv_tp_1(x, training=training)
        x = self.conv_tp_2(x, training=training)
        return self.out(x, training=training)


def get_loss_fn():  # define loss function
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        return criterion(tf.ones_like(fake_logits), fake_logits)

    return d_loss_fn, g_loss_fn


# data load & preprocessing
(train_x, train_y), (_, _) = tf.keras.datasets.mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype('float32')
train_x = (train_x - 127.5) / 127.5
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .repeat()
)

# generator & discriminator
G = Generator((Z_DIM,))
D = Discriminator(IMAGE_SHAPE)

# optimizer
g_optim = tf.keras.optimizers.Adam(G_LR, beta_1=0.5, beta_2=0.999)
d_optim = tf.keras.optimizers.Adam(D_LR, beta_1=0.5, beta_2=0.999)

# loss function
d_loss_fn, g_loss_fn = get_loss_fn()


@tf.function
def train_step(real_images, real_labels):
    z = get_random_z(Z_DIM, BATCH_SIZE)
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        fake_labels = tf.random.uniform(
            real_labels.shape, 0, 10, dtype=tf.int32)
        fake_images = G(z, fake_labels, training=True)

        fake_logits = D(fake_images, fake_labels, training=True)
        real_logits = D(real_images, real_labels, training=True)

        d_loss = d_loss_fn(real_logits, fake_logits)
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

            # write the log on the tensorboard
            with writer.as_default():
                tf.summary.scalar('g_loss', g_loss_metrics.result(), step=step)
                tf.summary.scalar('d_loss', d_loss_metrics.result(), step=step)
                tf.summary.scalar(
                    'total_loss', total_loss_metrics.result(), step=step)

            g_loss_metrics.reset_states()
            d_loss_metrics.reset_states()
            total_loss_metrics.reset_states()

        if step % test_freq == 0:
            # generate result images
            generate_and_save_images(
                partial(G, labels=test_labels), step, test_z, IMAGE_SHAPE, name='cgan', max_step=ITERATION, figsize=(10, 10))


train(train_ds)
