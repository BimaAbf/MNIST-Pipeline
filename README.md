# MNIST Digit Classifier

An intelligent handwritten digit classification system built from scratch using the MNIST dataset (70,000 images, 10 classes). The project combines deep learning feature extraction with classical machine learning classification, achieving **~98.4% test accuracy**.

## Pipeline Architecture

```
Input Image (28x28) --> LeNet-5 CNN --> 84-d Feature Vector --> 10 Binary LR Models --> Softmax --> Predicted Digit
```

1. **LeNet-5 CNN** is pretrained on MNIST for 10 epochs, then its classification head is removed
2. The frozen backbone extracts an **84-dimensional feature vector** per image
3. **10 One-vs-Rest Logistic Regression** models (built from scratch) each predict P(digit = k)
4. The 10 probabilities are passed through **softmax** to produce a final prediction

## Project Structure

```
.
|-- phase2.ipynb          # Main notebook: training, evaluation, diagnostics
|-- digit_gui.py          # Live drawing GUI for real-time digit prediction
|-- Metrics.py            # Custom metrics class (replaces sklearn.metrics)
|-- lenet5_mnist.pth      # Trained LeNet-5 weights
|-- lr_models.pkl         # Trained logistic regression models (pickled)
|-- feedback_log.csv      # User feedback log from the GUI
|-- phase1/
|   |-- phaseone.ipynb    # Phase 1 baseline notebook
```

## Key Features

- **CNN Feature Extraction**: LeNet-5 pretrained on MNIST serves as a fixed feature extractor for classical ML classifiers
- **Regularization Analysis**: L1 and L2 regularization sweep across multiple C values with train vs. validation comparison
- **Bias-Variance Diagnosis**: Systematic overfitting/underfitting analysis across regularization strengths
- **Learning Curves**: Training size vs. accuracy plots with 3-fold cross-validation to visualize model capacity
- **Live Deployment GUI**: Tkinter-based drawing interface with real-time inference, probability bar charts, and user feedback logging

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- Pillow (for the GUI)

## Usage

### Training the pipeline

Open and run `phase2.ipynb` from top to bottom. It will:
- Download MNIST automatically
- Train LeNet-5 and extract features
- Train the 10 logistic regression models
- Run regularization sweeps and diagnostic plots
- Save model weights to `lenet5_mnist.pth` and `lr_models.pkl`

### Running the live GUI

```bash
python digit_gui.py
```

Draw a digit on the canvas and the model predicts in real time. Use the feedback buttons to log whether predictions are correct.

## Results

| Split | Accuracy |
|-------|----------|
| Train | 99.09%   |
| Val   | 98.36%   |
| Test  | 98.36%   |

The bias-variance analysis confirms a **good fit** with a train-val gap of ~0.7%, indicating the model generalizes well without significant overfitting.
