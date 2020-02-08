import tensorflow as tf
from tensorflow.keras import layers


def make_discriminaor():
    return tf.keras.Sequential([
        layers.Dense(128,  activation='relu', input_shape=(784,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])


def make_generator():
    return tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(784, activation='tanh')
    ])
