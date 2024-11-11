Handwritten Digit Classifier

This project is a Convolutional Neural Network (CNN) model designed to classify handwritten digits from 0 to 9. It processes grayscale images, learns from labeled training data, and predicts labels for a test set. The model was implemented using Python, TensorFlow, and Keras.

Table of Contents
1. Project Overview
2. Data Preprocessing
3. Model Architecture
4. Training the Model
5. Evaluating Model Performance
6. Generating Predictions
7. Submission
8. Requirements

Project Overview

This project involves training a CNN to classify 28x28 grayscale images of handwritten digits. Each image corresponds to one of the 10 possible digits (0–9). We use a supervised learning approach with training and validation sets, then generate predictions on a test set for submission.

Data Preprocessing

The code assumes a labeled training dataset (train.csv) and a test dataset (test.csv) in CSV format. Here's how the data is processed:

1. Load Dataset: Load train.csv into a DataFrame and separate features (pixel values) from labels.
2. Reshape and Normalize: Reshape images to (28, 28, 1) for grayscale and normalize pixel values to the range [0, 1].
3. One-Hot Encode Labels: Convert the labels to categorical format for multi-class classification.
4. Train-Validation Split: Split the dataset into training and validation sets (80-20 split).

Model Architecture

The CNN model consists of three convolutional layers, each followed by max pooling, and ends with two fully connected layers for classification. Here is a summary of the architecture:

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # Output layer with 10 classes (digits 0-9)
])

Key Layers:
- Convolutional Layers: Feature extraction with filters.
- MaxPooling Layers: Reduce spatial dimensions, lowering computational load and controlling overfitting.
- Flatten Layer: Convert the 2D matrix to a 1D vector.
- Dense Layers: Fully connected layers, with the final layer outputting probabilities for each digit class.

Training the Model

After compiling the model with categorical_crossentropy as the loss function and adam as the optimizer, we train it for 10 epochs. Validation data is provided to monitor accuracy on unseen data:

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

Evaluating Model Performance

The model's validation accuracy is computed by comparing predicted labels on the validation set with true labels:

accuracy = np.mean(y_pred_val == y_val_true)
print(f"Validation Accuracy: {accuracy:.4f}")

This helps verify that the model is generalizing well on data it hasn't seen before.

Generating Predictions

For the test data, we reshape and normalize the images, then generate predictions:

test_predictions = model.predict(test_data).argmax(axis=1)

The predictions are stored in a submission-ready DataFrame.

Submission

The submission.csv file is generated with two columns:
- ImageId: The index of each test sample, starting from 1.
- Label: The predicted digit label for each sample.

submission.to_csv('submission.csv', index=False)

Requirements

To run this code, you'll need:
- Python 3.6+
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- scikit-learn

To install these dependencies, you can use:
pip install numpy pandas tensorflow scikit-learn

This README serves as a guide to understanding, running, and submitting predictions for this handwritten digit classification project.
