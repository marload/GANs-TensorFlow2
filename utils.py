import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime


def generate_and_save_images(generator, step, z_input, image_shape, name):
    assert type(image_shape) == tuple
    predictions = generator(z_input, training=False)
    if len(predictions.shape) <= 2:
        predictions = tf.reshape(
            predictions, (-1, image_shape[0], image_shape[1], image_shape[2]))

    fig = plt.figure(figsize=(6, 6))
    for idx in range(predictions.shape[0]):

        plt.subplot(6, 6, idx+1)
        plt.imshow(predictions[idx, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('result/{}/{}.png'.format(name, step), bbox_inches='tight')


def get_random_z(z_dim, batch_size):
    return tf.random.uniform([batch_size, z_dim], minval=-1, maxval=1)
