# Machine_learning_project

Deep Learning with PyTorch
This repository contains code for deep learning experiments using PyTorch. The code is organized into two main files:

main.py: This file includes the main program for training and testing deep learning models on the MNIST and CIFAR-10 datasets.

functions.py: This file contains utility functions and classes for training, testing, and hyperparameter tuning of deep learning models.

Before running the code, make sure you have the following dependencies installed:

Python 3.x
PyTorch
torchvision
scikit-learn (for hyperparameter tuning)
You can install these dependencies using pip:

pip install torch torchvision scikit-learn


Usage:
Training and Testing a Model
To train and test a logistic regression model on either the MNIST or CIFAR-10 dataset, use the following command:

python main.py --dataset <DATASET_NAME> --mode logistic --gpu 1


Replace <DATASET_NAME> with either MNIST or CIFAR10 depending on the dataset you want to use. The --gpu flag is optional and specifies whether to use GPU (1 for GPU, 0 for CPU).

Hyperparameter Tuning
To perform hyperparameter tuning on the selected dataset, use the following command:

python main.py --dataset <DATASET_NAME> --mode tune --target_metric <METRIC> --gpu 1

Replace <DATASET_NAME> with the dataset name (MNIST or CIFAR10), <METRIC> with the target metric (acc for accuracy or loss for loss), 
and --gpu to specify GPU usage.

Output
The program will display the training and validation progress, including loss and accuracy metrics. 
After training, it will show the final test accuracy.

Hyperparameter Tuning
The tune_hyper_parameter function in functions.py performs hyperparameter tuning using grid search over various hyperparameters,
including learning rate, batch size, number of epochs, weight decay, and optimizer type (Adam or SGD). It selects the best
hyperparameters based on the specified target metric.
