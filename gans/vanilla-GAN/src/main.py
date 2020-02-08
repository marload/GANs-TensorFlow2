import tensorflow as tf
from tensorflow.keras import layers

from model import make_discriminaor, make_generator
from utils import compute_disc_loss, compute_gen_loss

# HyperParameters
EPOCHS = 100
Z_DIM = 100
BATCH_SIZE = 256
BUFFER_SIZE = 6000

(train_x, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_x = train_x.reshape(train_x.shape[0], 28*28).astype('float32')
train_x -= 127.5
train_ds = tf.data.Dataset.from_tensor_slices(
    train_x).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

G = make_generator()
D = make_discriminaor()

g_optim = tf.keras.optimizers.Adam(1e-4)
d_optim = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(real_images):
    z = tf.random.uniform([BATCH_SIZE, Z_DIM])  # latent vector
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = G(z, training=True)

        fake_outputs = D(fake_images, training=True)
        real_outputs = D(real_images, training=True)

        g_loss = compute_gen_loss(fake_outputs)
        d_loss = compute_disc_loss(real_outputs, fake_outputs)

    g_gradients = g_tape.gradient(d_loss, G.trainable_variables)
    d_gradients = d_tape.gradient(g_loss, D.trainable_variables)

    g_optim.apply_gradients(zip(g_gradients, G.trainable_variables))
    d_optim.apply_gradients(zip(d_gradients, D.trainable_variables))

    return g_loss, d_loss


def train():
    for epoch in range(EPOCHS):
        for step, images in enumerate(train_ds):
            g_loss, d_loss = train_step(images)

            if step % 30 == 0:
                print('[{}/{}] [{}] D_loss={} G_loss={} total_loss={}'.format(epoch +
                                                                              1, EPOCHS, step, d_loss, g_loss, d_loss+g_loss))


train()
