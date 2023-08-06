# Yoga Pose Classification with Deep Learning

## Introduction

This repository contains code for a yoga pose classification project using deep learning. The project aims to classify yoga poses based on landmark data obtained from images. The models used in this project are implemented using TensorFlow and PyTorch frameworks. The goal is to develop accurate and efficient models for classifying yoga poses, which can have applications in fitness tracking, yoga training, and pose analysis.

## Dataset

The dataset used in this project contains landmark information extracted from images of various yoga poses. Each image is associated with a specific yoga pose, and the landmark data captures the key points of the pose. The dataset is split into training and testing sets to evaluate the performance of the trained models accurately.

## Requirements

To run the code in this repository, you need the following:

- Python (>= 3.6)
- TensorFlow (>= 2.0)
- PyTorch (>= 1.7)
- NumPy
- Matplotlib
- OpenCV
- scikit-learn
- tqdm
- TensorFlow Hub

Install the required packages using:

```bash
pip install tensorflow torch numpy matplotlib opencv-python scikit-learn tqdm tensorflow-hub
```

## Usage

To train and evaluate the models, follow these steps:

1. Clone this repository:

```bash
git clone https://github.com/your_username/yoga-pose-classification.git
cd yoga-pose-classification
```

2. Prepare the data:

The dataset should be placed in the `./DATASETNEW/yoga_set` directory. Make sure that the dataset is organized into subdirectories, with each subdirectory representing a different yoga pose and containing images of that pose.

3. Training the Base CNN model:

The initial model used in this project is the Base CNN model. Run the following code to train the Base CNN model:

```python
train_test_draw(model, "base_cnn", 200, 16)
```

4. Hyperparameter Tuning:

The project explores hyperparameter tuning using L2 regularization and changing batch numbers. For example:

- Using L2 regularization
- Changing Batch Number

5. Analysis Comparison with other SOTA models:

The project also compares the performance of the Base CNN model with other state-of-the-art models like DenseNet and EfficientNet. For example:

- DenseNet
- EfficientNet with PyTorch

6. Export Model

The trained pose classification model can be converted to TensorFlow Lite format ('pose_classifier.tflite') for efficient deployment on resource-constrained devices such as mobile phones and embedded systems. The model size, measured in kilobytes, provides an idea of the file's compactness, which is crucial for optimizing memory usage on these platforms.

To export the model to TensorFlow Lite, follow these steps:

```python
model = load_model("base_cnn")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print('Model size: %dKB' % (len(tflite_model) / 1024))

# Save the TFLite model to a file
with open('pose_classifier.tflite', 'wb') as f:
  f.write(tflite_model)

# Save the class labels to a file (optional)
with open('pose_labels.txt', 'w') as f:
  f.write('\n'.join(class_names))
```

After exporting the model, you can use the `evaluate_model` function to assess its accuracy on a test dataset. This function takes a TensorFlow Lite interpreter, the input test data (`X_test`), and the corresponding ground truth labels (`y_test`) as inputs.

```python
def evaluate_model(interpreter, X, y_true):
  """Evaluates the given TFLite model and returns its accuracy."""
  # ... (code continues as in the original segment)
```

Finally, you can evaluate the accuracy of the converted TFLite model as follows:

```python
classifier_interpreter = tf.lite.Interpreter(model_content=tflite_model)
classifier_interpreter.allocate_tensors()
print('Accuracy of TFLite model: %s' % evaluate_model(classifier_interpreter, X_test, y_test))
```

By exporting the model to TensorFlow Lite, you can deploy it efficiently on mobile and edge devices to perform real-time yoga pose classification.

## Results

Our proposed CNN achieved an accuracy of 96.7% and a loss of 0.288 on the Kaggle dataset. When tested on our collected Singapore dataset, the accuracy was 93.2%. Comparing our model with DenseNet and EfficientNet, our proposed model outperformed them in both accuracy and latency.

## Limitations:

One limitation is the lack of a specific dataset for Singaporeans doing yoga, which may affect the model's generalization to the local context. The model is currently optimized for able-bodied individuals and may not be suitable for individuals with physical disabilities.

## Proposed Modifications/Improvements:
To improve the model's accuracy and scalability, we propose data augmentation techniques, including image flipping and rotation. Additionally, we suggest generating a user-specific joint profile to accommodate differently-abled individuals. Obtaining a Singapore-labeled dataset would further enhance the model's accuracy for local user

## Conclusion
Our proposed system for yoga pose classification using deep learning models shows promising results, with high accuracy and low latency. The application can be a valuable tool for individuals practicing yoga at home, providing real-time feedback and guidance. With further improvements and inclusion of more diverse datasets, the model can cater to a broader audience and contribute to promoting a healthy lifestyle.
