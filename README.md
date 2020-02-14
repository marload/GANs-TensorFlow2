# TensorFlow2 GANs

This repo uses [TensorFlow2](https://github.com/tensorflow/tensorflow) to implement and store variants of GANs. I will continue to update this repo and my goal is to implement all the GANs that exist in the world. I was very inspired by the [repo of Hwalsuk Lee](https://github.com/hwalsuklee/tensorflow-generative-model-collections). I implemented it along the code style of the [TensorFlow2 Offical Tutorial](https://www.tensorflow.org/tutorials/). All GAN scrips do not rely on any external files except utils. If you are studying GAN, I believe that reference to this repo is the best choice. I sincerely thank you Ian Goodfellow for opening the GAN era and Google for creating the best framework, TensorFlow.


### GAN


|  Name  |                                                              GAN                                                               |
| :----: | :----------------------------------------------------------------------------------------------------------------------------: |
| Paper  |                               [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)                               |
| Author | Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio |
|  Code  |                             🔥[Implemented!](https://github.com/marload/TensorFlow2-GANs/GAN.py)🔥                             |

### DCGAN

|  Name  |                                                              DCGAN                                                               |
| :----: | :------------------------------------------------------------------------------------------------------------------------------: |
| Paper  | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) |
| Author |                                            Alec Radford, Luke Metz, Soumith Chintala                                             |
|  Code  |                             🔥[Implemented!](https://github.com/marload/TensorFlow2-GANs/DCGAN.py)🔥                             |

### LSGAN

|  Name  |                                       LSGAN                                        |
| :----: | :--------------------------------------------------------------------------------: |
| Paper  | [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)  |
| Author | Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley |
|  Code  |      🔥[Implemented!](https://github.com/marload/TensorFlow2-GANs/LSGAN.py)🔥      |

### WGAN

|  Name  |                                  WGAN                                   |
| :----: | :---------------------------------------------------------------------: |
| Paper  |           [Wasserstein GAN](https://arxiv.org/abs/1701.07875)           |
| Author |             Martin Arjovsky, Soumith Chintala, Léon Bottou              |
|  Code  | 🔥[Implemented!](https://github.com/marload/TensorFlow2-GANs/WGAN.py)🔥 |


### WGAN

|  Name  |                                      WGAN-GP                                      |
| :----: | :-------------------------------------------------------------------------------: |
| Paper  |     [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)     |
| Author | Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville |
|  Code  |    🔥[Implemented!](https://github.com/marload/TensorFlow2-GANs/WGAN-GP.py)🔥     |


### cGAN

|  Name  |                                    cGAN                                    |
| :----: | :------------------------------------------------------------------------: |
| Paper  | [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) |
| Author |                        Mehdi Mirza, Simon Osindero                         |
|  Code  |  🔥[Implemented!](https://github.com/marload/TensorFlow2-GANs/cGAN.py)🔥   |


### Semi supervised GAN

|  Name  |                            Semi supervised GAN                             |
| :----: | :------------------------------------------------------------------------: |
| Paper  | [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) |
| Author |                               Augustus Odena                               |
|  Code  |            🔥[Implemented!](https://arxiv.org/abs/1606.01583)🔥            |

#### ref

* https://github.com/tensorflow/tensorflow
* https://github.com/hwalsuklee/tensorflow-generative-model-collections
* https://www.tensorflow.org/tutorials/
* https://github.com/drewszurko/tensorflow-WGAN-GP