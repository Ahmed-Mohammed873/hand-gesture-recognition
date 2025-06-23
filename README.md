# Hand-gesture-recognition

This project implements a hand gesture recognition system using a Convolutional Neural Network (CNN) trained on the LeapGestRecog dataset. A simple Tkinter GUI is used for real-time webcam inference.

# Dataset

LeapGestRecog from Kaggle

10 Gesture Classes:

'palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down'

Grayscale .png images, grouped by subject and gesture

# Model Architecture

CNN with:

Conv2D → MaxPooling2D × 2

Flatten → Dense → Dropout

Final Dense layer with Softmax

Trained for 5 epochs with categorical_crossentropy loss and Adam optimizer

# Features

 Loads gesture image data from nested folders

 Normalizes and reshapes images (128×128 grayscale)

 Trains a CNN model to classify 10 hand gestures

 Saves the trained model as hand_gesture_model.h5

 Real-time webcam detection using OpenCV + Tkinter GUI

# File Structure

gesture_project/

│

├── #train_model.py       # Load data + train + save model

├── GUI.py               # Tkinter GUI for live webcam detection

├── hand_gesture_model.h5        # Saved trained model

└── README.md

# Usage

# Install dependencies:

pip install -r requirements.txt


# Train the model:

python # train_model.py

# Launch the GUI:

python gesture_gui.py

Make sure your webcam is connected and the dataset is in the correct structure before training.

