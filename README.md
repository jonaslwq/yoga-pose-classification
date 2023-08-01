# Pose Classification with MoveNet and TensorFlow

This repository contains code for building and training a pose classification model using TensorFlow and the MoveNet pose estimation model. The goal is to classify yoga poses based on the landmarks detected by MoveNet.

**Report:** https://publuu.com/flip-book/200279/484470/page/1

## Overview

The code consists of several components:

1. **Pose Estimation with MoveNet:** We use the MoveNet Thunder model from TensorFlow Hub to detect landmarks (keypoints) of yoga poses from input images.

2. **Preprocessing:** The `MoveNetPreprocessor` class is responsible for loading the images, running pose estimation, and saving the detected landmarks into CSV files. It also splits the dataset into training and test sets.

3. **Model Architecture:** We define a pose classification model using TensorFlow/Keras. The model takes in the detected landmarks and outputs the class probabilities for each yoga pose.

4. **Training:** The model is trained using the training dataset generated from the preprocessed landmarks. We use Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

5. **Model Conversion to TensorFlow Lite:** After training, the model is converted to TensorFlow Lite (TFLite) format for deployment on mobile and edge devices. TFLite models are optimized for inference on resource-constrained devices.

## Dependencies

- Python 3.x
- TensorFlow 2.x
- Numpy
- OpenCV (cv2)
- Pandas
- tqdm
- PyInquirer (for command-line interaction)

## How to Use

1. **Dataset Preparation:** Place your yoga pose images in the following directory structure:

```
yoga_poses/
|__ downdog/
    |______ 00000128.jpg
    |______ 00000181.jpg
    |______ ...
|__ goddess/
    |______ 00000243.jpg
    |______ 00000306.jpg
    |______ ...
...
```

Make sure to replace `yoga_poses/` with the actual path to your dataset.

2. **Preprocessing and Pose Estimation:** Run the `MoveNetPreprocessor` class to preprocess the images, detect pose landmarks, and save them into CSV files:

```python
# Replace 'dataset_in' with the path to your dataset.
# Set 'dataset_is_split' to False if you want to split the dataset into train and test sets.
from preprocess import MoveNetPreprocessor, split_into_train_test

use_custom_dataset = True  # Set this to True if using a custom dataset.
dataset_is_split = True    # Set this to False if you need to split the dataset.
dataset_in = 'yoga_poses/' # Replace this with your dataset path.

if use_custom_dataset:
    if not os.path.isdir(dataset_in):
        raise Exception("dataset_in is not a valid directory")

    if dataset_is_split:
        IMAGES_ROOT = dataset_in
    else:
        dataset_out = 'split_' + dataset_in
        split_into_train_test(dataset_in, dataset_out, test_split=0.2)
        IMAGES_ROOT = dataset_out

preprocessor = MoveNetPreprocessor(images_in_folder=IMAGES_ROOT,
                                   images_out_folder='output/images_with_landmarks',
                                   csvs_out_path='output/landmarks.csv')
preprocessor.process()
```

3. **Model Training:** Run the training code to build and train the pose classification model:

```python
from pose_classifier import build_pose_classifier_model, load_pose_landmarks

# Replace 'landmarks.csv' with the CSV file containing the detected landmarks.
X, y, class_names, _ = load_pose_landmarks('output/landmarks.csv')

# Replace this with your model architecture, or use the default pose_classifier model.
model = build_pose_classifier_model(class_names)

# Compile the model with appropriate optimizer, loss, and metrics.
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with your training and validation data.
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(X_val, y_val))
```

4. **Model Conversion to TFLite:** After training, convert the model to TFLite format for deployment on mobile and edge devices:

```python
import tensorflow as tf

# Convert the Keras model to TFLite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model to a file.
with open('pose_classifier.tflite', 'wb') as f:
  f.write(tflite_model)
```

5. **Evaluation with TFLite Model:** Load the TFLite model and evaluate its accuracy on the test dataset:

```python
from sklearn.metrics import accuracy_score

# Load the TFLite model.
classifier_interpreter = tf.lite.Interpreter(model_content=tflite_model)
classifier_interpreter.allocate_tensors()

# Replace X_test and y_test with your test dataset.
accuracy = evaluate_model(classifier_inter
