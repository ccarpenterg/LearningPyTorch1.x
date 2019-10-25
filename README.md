# Learning PyTorch 1.x Notebooks

<img src="https://pytorch.org/assets/images/logo-icon.svg" title="TF-logo" width="100" />


### 1) Getting Started with PyTorch: Training a Neural Network on MNIST

In this notebook, we will train a neural network (1 input layer, 1 hidden layer, 1 output layer) to classify handwritten digits. We use the **Module** as our container from the **torch.nn** module. We'll be training our neural network  on the MNIST, which is the "hello world" of Machine learning and Deep learning algoriths:

<img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" title="MNIST" width="375" />

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningPyTorch1.x/blob/master/01_getting_started_with_pytorch.ipynb)) ([github](https://github.com/ccarpenterg/LearningPyTorch1.x/blob/master/01_getting_started_with_pytorch.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningPyTorch1.x/blob/master/01_getting_started_with_pytorch.ipynb))

### 2) Introduction to Convolutional Neural Networks and Deep Learning

Similar to artificial neural networks, convolutional neural networks consist of artificial neurons organized in layers. Convolutional neural networks (convnets or CNNs for short), however, introduced the concept of convolution to automatically extract features that then are fed to a classifier of fully connected neurons.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Typical_cnn.png/800px-Typical_cnn.png" 
title="CNN" width="500" />

In this notebook, we build a convolutional neural network from zero and train it on the MNIST dataset.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningPyTorch1.x/blob/master/02_introduction_to_convnets_with_pytorch.ipynb)) ([github](https://github.com/ccarpenterg/LearningPyTorch1.x/blob/master/02_introduction_to_convnets_with_pytorch.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningPyTorch1.x/blob/master/02_introduction_to_convnets_with_pytorch.ipynb))

### 3) Plotting Accuracy and Loss for CNNs with PyTorch

Part of the work that involves designing and training deep neural networks, consists in plotting the various parameters and metrics generated in the process of training. In this notebook we will design and train our Convnet from scratch, and will plot the training vs. test accuracy, and the training vs. test loss.

These are very important metrics, since they will show us how well is doing our neural network.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningPyTorch1.x/blob/master/03_plotting_loss_and_accuracy.ipynb)) ([github](https://github.com/ccarpenterg/LearningPyTorch1.x/blob/master/03_plotting_loss_and_accuracy.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningPyTorch1.x/blob/master/03_plotting_loss_and_accuracy.ipynb))

### 4) CIFAR-10: A More Challenging Dataset for CNNs

So far we have trained our neural networks on the MNIST dataset, and have achieved high acurracy rates for both the training and test datasets. Now we train our Convnet on the CIFAR-10 dataset, which contains 60,000 images of 32x32 pixels in color (3 channels) divided in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png" title="CIFAR-10" width="295" />

As we'll see in this notebook, the CIFAR-10 dataset will prove particularly challenging for our very basic Convnet, and from this point we'll start exploring the world of pretrained neural networks.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningPyTorch1.x/blob/master/04_cifar_10_challenging_convnets.ipynb)) ([github](https://github.com/ccarpenterg/LearningPyTorch1.x/blob/master/04_cifar_10_challenging_convnets.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningPyTorch1.x/blob/master/04_cifar_10_challenging_convnets.ipynb))

### 5) Pretrained Convolutional Neural Networks

Pretrained Convnets have been trained on large datasets such as [Imagenet](https://en.wikipedia.org/wiki/ImageNet) and are available for downloading (including the architecture/layers and its parameters/weights).

We will see that it's possible to tweak a pretrained convolutional neural network and transfer its "learning" to our problem.

<img src="https://d2l.ai/_images/residual-block.svg" title="ResNet" width="480" />

In this notebook, we will explore the convnet ResNet18 and we will take a look at its structure and number of parameters. Also we will discuss what it means to freeze and unfreeze a layer.

notebook: ([nbviewer](https://nbviewer.jupyter.org/github/ccarpenterg/LearningPyTorch1.x/blob/master/05_pretrained_convnets_and_transfer_learning.ipynb)) ([github](https://github.com/ccarpenterg/LearningPyTorch1.x/blob/master/05_pretrained_convnets_and_transfer_learning.ipynb)) ([colab](https://colab.research.google.com/github/ccarpenterg/LearningPyTorch1.x/blob/master/05_pretrained_convnets_and_transfer_learning.ipynb))
