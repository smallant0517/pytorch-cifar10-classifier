# CIFAR-10 CNN Classifier
A simple CIFAR-10 classifier built with PyTorch.


## 🧱 Model Architecture
Conv2d(3 → 32) → ReLU → MaxPool2d
Conv2d(32 → 64) → ReLU → MaxPool2d
Flatten → Linear(4096 → 128) → ReLU
Linear(128 → 10)

## 📁 Project Structure
project/
│
├── src/
│ ├── dataset.py # Data loading + transforms
│ ├── model.py # CNN model definition
│
├── train.py # Training script
├── test.py # Evaluation script
├── README.md
└── requirements.txt

## 🚀 Training
python train.py

## 🧪 Testing
python test.py

## 📦 Requirements
pip install -r requirements.txt

## 📊 Result Example
Epoch [5/5] Loss: 1.1023
Test Accuracy: 72.34%