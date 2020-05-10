![TF Depend](https://img.shields.io/badge/TensorFlow-2.1-orange) ![License Badge](https://img.shields.io/badge/license-Apache%202-green)<br>

<p align="center">
  <img width="150" src="./assets/logo.png">
</p>

<h2 align=center>Generative Adversarial Nets in TensorFlow2</h2>

[GANs-TensorFlow2](https://github.com/marload/GANs-TensorFlow2) is a repository that implements a variety of popular Generative Adversarial Network algorithms using [TensorFlow2](https://tensorflow.org). The key to this repository is an easy-to-understand code. Therefore, if you are a student or a researcher studying Deep Reinforcement Learning, I think it would be the **best choice to study** with this repository. One algorithm relies only on one python script file. So you don't have to go in and out of different files to study specific algorithms. This repository is constantly being updated and will continue to add a new Generative Adversarial Network algorithm.

## Algorithms

- [GAN](#gan)
- [DCGAN](#dcgan)
- [LSGAN](#lsgan)
- [WGAN](#wgan)
- [WGAN-GP](#wgan-gp)
- [DRAGAN](#dragan)

<hr>

<a name='gan'></a>

### GAN

**Paper** [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)<br>
**Author** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio<br>
**Publish** NIPS 2014

#### Animation Results

<p align="center">
  <img width="350" src="./assets/gan.gif">
</p>

#### Loss Function

```python
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        return -tf.reduce_mean(tf.math.log(real_logits + 1e-10) + tf.math.log(1. - fake_logits + 1e-10))

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(tf.math.log(fake_logits + 1e-10))

    return d_loss_fn, g_loss_fn
```

#### Getting Start

```bash
$ python GAN/GAN.py
```

<hr>

<a name='dcgan'></a>

### DCGAN

**Paper** [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)<br>
**Author** Alec Radford, Luke Metz, Soumith Chintala<br>
**Publish** ICLR 2016

#### Animation Results
<p align="center">
  <img width="350" src="./assets/dcgan.gif">
</p>

#### Loss Function
```python
def get_loss_fn():
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def d_loss_fn(real_logits, fake_logits):
        real_loss = criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        return criterion(tf.ones_like(fake_logits), fake_logits)

    return d_loss_fn, g_loss_fn
```

#### Getting Start
```bash
$ python DCGAN/DCGAN.py
```

<hr>


<a name='lsgan'></a>

### LSGAN

**Paper** [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)<br>
**Author** Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley<br>
**Publish** ICCV 2017

#### Animation Results
<p align="center">
  <img width="350" src="./assets/lsgan.gif">
</p>

#### Loss Function
```python
def get_loss_fn():
    criterion = tf.keras.losses.MeanSquaredError()

    def d_loss_fn(real_logits, fake_logits):
        real_loss = criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss_fn(fake_logits):
        return criterion(tf.ones_like(fake_logits), fake_logits)

    return d_loss_fn, g_loss_fn
```

#### Getting Start
```bash
$ python LSGAN/LSGAN.py
```

<hr>

<a name='wgan'></a>

### WGAN

**Paper** [Wasserstein GAN](https://arxiv.org/abs/1701.07875)<br>
**Author** Martin Arjovsky, Soumith Chintala, Léon Bottou<br>
**Publish** arXiv 2017

#### Animation Results
<p align="center">
  <img width="350" src="./assets/wgan.gif">
</p>

#### Loss Function
```python
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(fake_logits)

    return d_loss_fn, g_loss_fn
```

#### Getting Start
```bash
$ python WGAN/WGAN.py
```

<hr>

<a name='wgan-gp'></a>

### WGAN-GP

**Paper** [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)<br>
**Author** Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville<br>
**Publish** NIPS 2017

#### Animation Results
<p align="center">
  <img width="350" src="./assets/wgan-gp.gif">
</p>

#### Loss Function
```python
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(fake_logits)

    return d_loss_fn, g_loss_fn

def gradient_penalty(generator, real_images, fake_images):
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)
    alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0., 1.)
    diff = fake_images - real_images
    inter = real_images + (alpha * diff)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        predictions = generator(inter)
    gradients = tape.gradient(predictions, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    return tf.reduce_mean((slopes - 1.) ** 2)
```

#### Getting Start
```bash
$ python WGAN-GP/WGAN-GP.py
```

<hr>

<a name='dragan'></a>

### DRAGAN

**Paper** [On Convergence and Stability of GANs](https://arxiv.org/abs/1705.07215)<br>
**Author** Naveen Kodali, Jacob Abernethy, James Hays, Zsolt Kira<br>
**Publish** ICLR 2018

#### Animation Results
<p align="center">
  <img width="350" src="./assets/dragan.gif">
</p>

#### Loss Function
```python
def get_loss_fn():
    def d_loss_fn(real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def g_loss_fn(fake_logits):
        return -tf.reduce_mean(fake_logits)

    return d_loss_fn, g_loss_fn

def gradient_penalty(generator, real_images):
    real_images = tf.cast(real_images, tf.float32)
    def _interpolate(a):
        beta = tf.random.uniform(tf.shape(a), 0., 1.)
        b = a + 0.5 * tf.math.reduce_std(a) * beta
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape, 0., 1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter
    
    x = _interpolate(real_images)
    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = generator(x, training=True)
    grad = tape.gradient(predictions, x)
    slopes = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    return tf.reduce_mean((slopes - 1.) ** 2)
```

#### Getting Start
```bash
$ python DRAGAN/DRAGAN.py
```

<hr>

## Reference

* https://github.com/tensorflow/tensorflow
* https://github.com/hwalsuklee/tensorflow-generative-model-collections
* https://www.tensorflow.org/tutorials/
* https://github.com/drewszurko/tensorflow-WGAN-GP
