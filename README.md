# -image-classification-model

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: KONGARAPU ASHA 

*INTERN ID*: CT04DF84

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

📌 Project Description – Image Classification Model (Task 3)
As part of my internship at CODTECH IT SOLUTIONS, I was assigned Task 3: Image Classification Model, where the objective was to build and evaluate a Convolutional Neural Network (CNN) using a deep learning framework. The purpose of this task was to design a model capable of classifying images into multiple categories, train it on a dataset, and evaluate its performance on unseen test data. I used Python programming language, the TensorFlow and Keras deep learning libraries, and the Visual Studio Code (VS Code) environment for developing and executing the code.

🔹 Objective:
The primary aim of this task was to build a CNN-based image classification model that could effectively recognize and classify images from a predefined dataset. I chose the CIFAR-10 dataset, a well-known benchmark dataset that consists of 60,000 32x32 color images across 10 different classes such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. This dataset is widely used in the deep learning community for training and evaluating image classification models.

🔹 Tools & Platform Used:
Programming Language: Python 3.12

Libraries: TensorFlow, Keras, NumPy, Matplotlib

IDE/Platform: Visual Studio Code (VS Code)

Dataset: CIFAR-10 (from Keras datasets)

System: Windows 10, CPU-based execution

🔹 Task Implementation:
I implemented the CNN using the Sequential API provided by Keras. The model consisted of the following layers:

An Input layer defining the input shape of (32, 32, 3)

Three Conv2D layers with ReLU activation functions to extract spatial features from the images

Two MaxPooling2D layers to reduce the dimensionality and retain essential features

A Flatten layer to convert the 2D matrix into a 1D feature vector

A Dense layer with 64 units and ReLU activation

A final Dense layer with 10 output neurons using Softmax activation for multi-class classification

The model was compiled using the Adam optimizer and the sparse categorical crossentropy loss function, which is suitable for multi-class classification with integer labels. I trained the model for 10 epochs using the CIFAR-10 training dataset and evaluated it using the test dataset to observe generalization performance.

🔹 Evaluation:
During training, I monitored both training accuracy and validation accuracy. A graphical plot was generated to visualize the model’s learning progress over epochs. The training accuracy increased steadily with each epoch, reaching approximately 77.8%, while the validation accuracy stabilized around 70%. The loss values also decreased consistently, indicating that the model was learning effectively without overfitting. The final evaluation on the test set confirmed that the model had generalized well to unseen data.

🔹 Outcome:
This task helped me gain practical experience in building deep learning models using TensorFlow and Keras. It strengthened my understanding of CNNs, overfitting, activation functions, loss evaluation, and model optimization. I also learned how to properly structure a model, monitor training progress, and interpret performance metrics. All implementation and model training were carried out in VS Code, which provided a flexible and efficient development environment.

🔹 Conclusion:
Successfully completing this image classification task allowed me to apply theoretical knowledge of deep learning into practice. The final model demonstrated good performance on the test data and met the requirements stated in the internship instructions. This task has enhanced my skills in machine learning, model development, and problem-solving using Python and TensorFlow, and has laid a strong foundation for future projects involving computer vision and AI.

