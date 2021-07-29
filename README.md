# Quantum-Classical Hyrid Neural Network for binary image classification using PyTorch-Qiskit pipeline

This project involved developing a Hybrid Quantum Neural Network using the amalgamation of PyTorch and Qiskit , i.e intergrating the classical ML tools and features of PyTorch with the Quantum Computing framework of Qiskit. Hold-out validation was carried out, the model was tested on a validation set before noting the test accuracy. Hyperparameter tuning , effect of changing learning rates, optimizer and loss function on the model and Layer architectures were studied. The Quantum Layer involved a Parametrized Quantum circuit analogous to using a Variational Circuit as a classifier. This project was carried out under the guidance of Dr Elias F Combarro, Professor of Computer Science at  Universidad de Oviedo, Spain  and Advisor CERN QTI to whom I am extremely indebted to for his help and mentorship.


## Pre-requisites 
The following are the pre-requisites for running the notebook on a local machine (Google Colab was used throughout this project along with IBM Quantum Experience owing to the ease in integrating PyTorch and Qiskit without the local installation of additional dependencies).

* Python3
* Qiskit
* PyTorch
* Matplotlib
* Numpy
* torchvision

The Google Colab notebooks can be accessed under the [Notebook](https://github.com/DRA-chaos/Quantum-Convolutional-Neural-Network/tree/main/Notebooks) folder of this repository.

## Dataset:
The CIFAR-10 dataset outsourced from the torchvision datasets under PyTorch has been used for this project.
The CIFAR-10 dataset that can be accessed [here](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
The project encompassed building a variational classifier to distinguish between Aeroplanes and Automobiles.
![image](https://user-images.githubusercontent.com/68393451/127451266-36669b90-bacb-4c38-afb5-ee2d2199d9f9.png)

[image source](https://www.cs.toronto.edu/~kriz/cifar.html)

Here are a few sample observations from the study :
![image](https://user-images.githubusercontent.com/68393451/127451652-5e70a45d-80c9-4c1e-b51d-09bb34fa8ed9.png)

## Flowchart depicting the Quantum Layer

![image](https://user-images.githubusercontent.com/68393451/127452166-2361aad8-817d-4fbb-a626-01086588ee8c.png)


## References:
[1]Crooks, Gavin. (2019). Gradients of parameterized quantum gates using the parameter-shift rule and gate decomposition. 

[2]A. Asfaw, L. Bello, Y. Ben-Haim, S. Bravyi, L. Capelluto, A. C. Vazquez, J. Ceroni, J. Gambetta, S. Garion, L. Gil, et al., Learn quantum computation using qiskit (2020),
URL:https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html

[3]Krizhevsky, Alex. (2012). Learning Multiple Layers of Features from Tiny Images. University of Toronto. 

[4]Oh, Seunghyeok & Choi, Jaeho & Kim, Joongheon. (2020). A Tutorial on Quantum Convolutional Neural Networks (QCNN).

[5]Farhi, Edward & Neven, Hartmut. (2018). Classification with Quantum Neural Networks on Near Term Processors. 

[6]Kulkarni, Viraj & Kulkarni, Milind & Pant, Aniruddha. (2020). Quantum Computing Methods for Supervised Learning. 

[7]Beer, K., Bondarenko, D., Farrelly, T. et al. Training deep quantum neural networks. Nat Commun 11, 808 (2020). https://doi.org/10.1038/s41467-020-14454-2

Enjoy your journey through the quantum realm !!

Rita Abani 



