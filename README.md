# Image Classification: MLP vs. CNN (LeNet-5)

This project implements and compares two deep learning architectures for image classification tasks: a custom Multi-Layer Perceptron (MLP) for the MNIST dataset and the LeNet-5 Convolutional Neural Network for the CIFAR-10 dataset.

## 📌 Project Overview

The goal of this project is to analyze how different hyperparameters affect model convergence, accuracy, and feature representation. 

### Geometries Implemented
1.  **MLP (MNIST)**: 
    * Input: 784 (28x28 flattened)
    * Hidden Layer 1: 250 neurons
    * Hidden Layer 2: 100 neurons
    * Output: 10 classes
2.  **LeNet-5 (CIFAR-10)**:
    * Standard Yann LeCun architecture adapted for 3-channel input.

### Experiments
For both architectures, we iteratively test the following variations:
* **Activation Functions**: ReLU vs. Sigmoid
* **Batch Sizes**: 8 vs. 64
* **Loss Functions**: Cross Entropy vs. Mean Squared Error (MSE)
* **Optimizers**: SGD vs. Adam

## 📂 Repository Structure

- `MultilayerPerceptron.ipynb`: Code for the MNIST experiments. Includes class definition, training loops, and PCA visualization for the 100-neuron hidden layer.
- `LeNet_5.ipynb`: Code for the CIFAR-10 experiments. Includes dynamic model creation and performance heatmaps.
- `Project_Report.pdf`: (Optional) Consolidated report containing all plots and interpretations.

## 🛠️ Requirements

* Python 3.x
* PyTorch
* Torchvision
* Scikit-Learn (for PCA and Metrics)
* Matplotlib & Seaborn (for Visualization)

## 📊 Key Results

The project generates the following visualizations for every experimental case:
1.  **Confusion Matrix**: To visualize classification performance per class.
2.  **PCA Projection**: A 2D visualization of the internal feature representations (Hidden Layer 2 for MLP, FC2 for LeNet), showing how well the network separates classes in the latent space.

## 🚀 Usage

Open the notebooks in Jupyter or Google Colab. The scripts are self-contained and will download the necessary datasets (MNIST/CIFAR-10) automatically upon execution.
