import tensorflow as tf


def get_loss_fn():
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def compute_disc_loss(real_output, fake_output):
        real_loss = loss_object(tf.ones_like(real_output), real_output)
        fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def compute_gen_loss(fake_output):
        return loss_object(tf.ones_like(fake_output), fake_output)

    return compute_disc_loss, compute_gen_loss
