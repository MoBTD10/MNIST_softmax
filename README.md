# MNIST Handwritten Digit Recognition using TensorFlow & Keras

This repository contains a Deep Learning pipeline built with **TensorFlow** and **Keras** to classify handwritten digits (0-9) from the classic **MNIST dataset**. The project implements a Multi-Layer Perceptron (MLP) Neural Network with regularization callbacks to prevent overfitting.

## 🧠 Neural Network Architecture
The model is built sequentially using dense layers:
1. **Input Layer:** `Flatten` layer that reshapes the $28 \times 28$ pixel grayscale images into a 1D vector of 784 features.
2. **Hidden Layer 1:** Fully connected `Dense` layer with **128 units** and `ReLU` activation.
3. **Hidden Layer 2:** Fully connected `Dense` layer with **32 units** and `ReLU` activation for deeper feature extraction.
4. **Output Layer:** `Dense` layer with **10 units** (one for each digit from 0 to 9) using `Softmax` activation to output class probabilities.

## ⚙️ Training Strategy & Regularization
* **Data Normalization:** Pixels were scaled from intensity values of `[0, 255]` down to a float range of `[0.0, 1.0]` by dividing by `255.0` to ensure faster optimization convergence.
* **Optimization:** Compiled using the **Adam** optimizer and **Sparse Categorical Crossentropy** loss function.
* **Overfitting Prevention (Early Stopping):** Implemented a `Keras` callback to monitor validation loss. With a `patience=3`, training safely halted at **Epoch 11** out of 20, preserving the weights with the best generalization performance.

## 📊 Performance Results
* **Test Accuracy:** `97.47%`
* **Test Loss:** `0.0873`
* The training script also includes loss visualization plots (`train_loss` vs `val_loss`) to evaluate the convergence behavior.

## 🔮 Model Persistence & Live Inference
* **Inference Pipeline:** Features a custom sample prediction workflow that reshapes a single image slice, runs it through the trained network, extracts the highest probability class using `.argmax()`, and visualizes it alongside the image.
* **Saving:** The final trained architecture and weights are exported locally using the native Keras format (`mnist_model.keras`).
