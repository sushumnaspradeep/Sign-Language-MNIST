# Sign-Language-MNIST
Hand Gesture Recognition using CNN on Sign Language MNIST Dataset

Introduction

This project aims to evaluate the performance of Convolutional Neural Networks (CNNs) in recognizing hand gestures from the Sign Language MNIST dataset. The dataset consists of grayscale images representing American Sign Language (ASL) letters, excluding 'J' and 'Z' due to their motion-dependent nature. The objective is to preprocess the dataset, train CNN models, optimize parameters, and analyze their performance.

Dataset Overview

The Sign Language MNIST dataset contains:

Training set: 27,455 grayscale images (28x28 pixels)

Test set: 7,712 grayscale images (28x28 pixels)

Total classes: 24 (letters 'A' to 'I' and 'K' to 'Z')

Format: Similar to the original MNIST dataset

Data Visualization

To understand the distribution of samples across different classes, we used multiple visualization techniques:

Histogram Plot - Displays the count of images for each alphabet.

Seaborn Count Plot - Another way to represent the class distribution.

Pie Chart (Plotly) - Provides an interactive breakdown of alphabet occurrences.

Data Preprocessing

Preprocessing was carried out to standardize the dataset and enhance model performance:

Grayscale Conversion: Reduces computational load by converting images to single-channel.

Histogram Equalization: Adjusts image contrast for better feature extraction.

Label Encoding: One-hot encoding using the Label Binarizer from Scikit-learn.

Normalization: Each pixel value is scaled between 0 and 1 by dividing by 255.

Dataset Splitting: Training data was further split into 80% training and 20% validation sets.

Model Implementation and Evaluation

Understanding Convolution

Convolution in CNNs is the process of applying filters (kernels) to input images to extract spatial features. In this project, we used 3x3 filters in all models.

Evaluation Metric Selection

We chose accuracy as the primary evaluation metric since all classes are equally important. It is calculated as:


CNN Model Experiments

Baseline Model (Model 1)

Two convolutional layers (filters: 32, 64)

Max pooling for down-sampling

Dropout layer for regularization

Adam optimizer

Test Accuracy: 90.51%

Improved Model with Batch Normalization (Model 2)

Added batch normalization to stabilize training

Helps in faster convergence and prevents overfitting

Test Accuracy: 91.39%

Overfitting Experiment (Model 3)

Removed a dropout layer

Test Accuracy: 94.91% (indicating overfitting)

Best Model Selection

While Model 3 had the highest accuracy, it overfitted the training data. Therefore, Model 2 (with batch normalization) was chosen as the best model due to its balance between accuracy and generalization.

Optimizer Experiments

To analyze the impact of different optimizers, we replaced Adam in Model 2 with:

Stochastic Gradient Descent (SGD) – 90.28%

Adagrad – 82.93%

Adamax – 90.88%

Adam (Baseline) – 91.39%

Adam performed the best, likely due to its adaptive learning rate properties, making it more effective for this dataset.

Further Model Improvements

Experiment 1: Adding an Extra Convolutional Layer

Introduced a third convolutional layer

Test Accuracy Increased to 94.75%

Experiment 2: Increasing Epochs & Batch Size

Increased epochs from 10 to 15 and batch size from 128 to 1024

Test Accuracy Improved to 92.67%

Key Takeaways

Batch normalization improved performance by stabilizing gradients and accelerating training.

Dropout prevents overfitting, and its removal led to high but unreliable accuracy.

Adam optimizer performed best for this dataset, followed by SGD.

Adding more convolutional layers enhanced feature extraction, leading to higher accuracy.

Increasing training epochs and batch size improved accuracy, but requires balancing with computation time.

Grayscale conversion speeds up training but may cause minor information loss.

Conclusion

This project explored CNN architectures for hand gesture recognition using ASL alphabets. Various preprocessing techniques, optimizers, and architectural changes were tested. The best model (CNN with batch normalization and Adam optimizer) achieved 91.39% accuracy, while further improvements led to 94.75% accuracy. Future work may involve experimenting with deeper networks, data augmentation, or transfer learning.

This study highlights the trade-offs between accuracy, computational efficiency, and model complexity in deep learning-based image classification.
