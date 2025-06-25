# Sketch Classification System

This project implements a sketch classification system that can identify hand-drawn sketches using multiple feature extraction techniques and machine learning models.

## Overview

The sketch classification pipeline consists of:
1. **Data loading** - Working with a PNG sketch dataset
2. **Image preprocessing** - Denoising, contrast enhancement, and filtering
3. **Feature extraction** - HOG, LBP, and CNN features
4. **Dimensionality reduction** - Using PCA
5. **Model training and evaluation** - Various ML classifiers

## Dataset

The project uses the "sketches-png-dataset" which contains hand-drawn sketches organized in categories. Each sketch is processed as a grayscale image and resized to 128x128 pixels before feature extraction.

## Features

The system extracts three types of features:

1. **HOG (Histogram of Oriented Gradients)** - Captures the distribution of gradient directions
2. **LBP (Local Binary Pattern)** - Captures texture patterns
3. **CNN Features** - Deep features extracted using a pre-trained ResNet-34

These features are then combined to create a robust representation of each sketch.

## Models

The project implements and compares several classification algorithms:

- **Logistic Regression**: Simple and fast linear classifier
- **Random Forest**: Ensemble of decision trees
- **SVM**: Support Vector Machine with RBF kernel
- **KNN**: K-Nearest Neighbors
- **MLP**: Multi-Layer Perceptron neural network
- **Naive Bayes**: Probabilistic classifier

## Results

The Logistic Regression model achieved the highest accuracy of 47.68% on the test set using the combined feature set without PCA. The SVM model also performed well with 43.65% accuracy.

### Performance Comparison:

- Logistic Regression: 47.68% accuracy
- SVM: 43.65% accuracy
- Naive Bayes: 34.65% accuracy
- Random Forest: 32.18% accuracy

## Feature Importance

The combination of CNN features (extracted from ResNet-34) and LBP features provided the best representation for sketch classification. The CNN features capture high-level semantic information while LBP features focus on local texture patterns.

## Visualization

The project includes t-SNE visualization to show how well the extracted features separate different sketch classes in a 2D space.

## Prerequisites

- Python 3.x
- NumPy
- OpenCV
- scikit-learn
- PyTorch
- TorchVision
- scikit-image
- Matplotlib

## Usage

1. **Load features**:
```python
import pickle
import numpy as np

# Load features
with open("lbp_features.pkl", "rb") as f:
    lbp_features = pickle.load(f)
    
with open("cnn_features.pkl", "rb") as f:
    cnn_features = pickle.load(f)
```

2. **Combine features**:
```python
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler_lbp = StandardScaler()
scaler_cnn = StandardScaler()

lbp_scaled = scaler_lbp.fit_transform(lbp_features)
cnn_scaled = scaler_cnn.fit_transform(cnn_features)

# Combine
combined_features = np.concatenate([lbp_scaled, cnn_scaled], axis=1)
```

3. **Train and use a model**:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.2, random_state=42)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
```

## Future Improvements

- Implement data augmentation to increase dataset size
- Try more advanced feature extraction techniques
- Experiment with ensemble methods combining multiple classifiers
- Implement a deep learning approach using CNNs directly on the images
- Add a web interface for real-time sketch classification

## Acknowledgments

- The sketch dataset used in this project
- PyTorch for providing pre-trained models for feature extraction
