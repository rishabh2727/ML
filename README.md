# Machine learning project
The provided code comprises a machine learning project that focuses on image classification using logistic regression. It is designed to work with two datasets, namely MNIST and CIFAR-10, with the primary goal of training and evaluating a logistic regression model's performance.

The logistic regression model is utilized for image classification tasks. For MNIST, it operates on 28x28 pixel grayscale images, while for CIFAR-10, it handles 32x32 pixel color images. The code includes functions for training the model, allowing users to specify batch size, learning rate, and optimizer (Adam or SGD), along with the number of training epochs.

In terms of evaluation, the project calculates accuracy, loss, and relevant metrics on a separate validation dataset. For MNIST, a predefined accuracy score benchmark is used for performance evaluation. Additionally, the code offers functionality for hyperparameter tuning, seeking the optimal combination of hyperparameters (e.g., learning rate, batch size) to enhance model performance. Both accuracy and loss can be used as target metrics during the tuning process.

The project is adaptable to different computing devices, supporting both CPU and GPU utilization if available. It handles data loading and preprocessing tasks, applying essential transformations such as normalization to the input data. The code's modularity enables users to switch between datasets (MNIST or CIFAR-10) and modes (training or hyperparameter tuning) with ease.

Overall, this machine learning project provides a versatile framework for building, training, and evaluating logistic regression models for image classification tasks, empowering users to work with various datasets and optimize model performance through hyperparameter tuning.
