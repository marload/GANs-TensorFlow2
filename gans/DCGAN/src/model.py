import tensorflow as tf
from tensorflow.keras import layers


def make_discriminaor():
    return tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2),
                      padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])


def make_generator():
    def BatchNormLeakyReLU():
        return tf.keras.Sequential([
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2)
        ])

    return tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        BatchNormLeakyReLU(),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5),
                               strides=(1, 1), padding='same', use_bias=False),
        BatchNormLeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5),
                               strides=(2, 2), padding='same', use_bias=False),
        BatchNormLeakyReLU(),
        layers.Conv2DTranspose(1, (5, 5),
                               strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
