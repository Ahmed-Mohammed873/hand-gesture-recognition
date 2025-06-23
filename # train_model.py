# train_model.py (Corrected to be location-aware)
# This script loads the dataset, trains a CNN model, and saves it.

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_data(data_dir):
    """
    Loads image data from the leapgestrecog dataset directory structure.
    Expected structure: data_dir/subject_id/gesture_name/image.png
    """
    images = []
    labels = []
    # Define a fixed mapping for gestures to ensure consistency.
    gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
    gesture_map = {name: i for i, name in enumerate(gestures)}
    print(f"Gesture mapping used for training: {gesture_map}")
    print(f"Loading data from: {data_dir}")

    # Iterate through subject directories (e.g., '00', '01', ...)
    for subject_dir in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        # Iterate through gesture directories (e.g., '01_palm', '02_l', ...)
        for gesture_dir in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture_dir)
            if not os.path.isdir(gesture_path):
                continue
            
            try:
                # Extract gesture name from directory name (e.g., '01_palm' -> 'palm')
                gesture_name = '_'.join(gesture_dir.split('_')[1:])
                if gesture_name not in gesture_map:
                    continue
                gesture_label = gesture_map[gesture_name]
            except IndexError:
                continue

            # Load all images within the gesture directory
            for image_name in os.listdir(gesture_path):
                if image_name.endswith('.png'):
                    image_path = os.path.join(gesture_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        image = cv2.resize(image, (128, 128))
                        images.append(image)
                        labels.append(gesture_label)

    return np.array(images), np.array(labels), gesture_map

def main():
    """ Main function to run the data loading, model training, and saving. """
    try:
        # --- FIX: Make the script location-aware ---
        # Get the absolute path of the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the base directory for the dataset relative to the script's location
        base_dir = os.path.join(script_dir, 'leapgestrecog')
        data_root = base_dir
        
        # Automatically detect and handle common nested directories
        if os.path.exists(base_dir):
            dir_contents = os.listdir(base_dir)
            if len(dir_contents) == 1 and os.path.isdir(os.path.join(base_dir, dir_contents[0])):
                nested_dir = os.path.join(base_dir, dir_contents[0])
                print(f"Detected a nested directory. Using '{nested_dir}' as the data root.")
                data_root = nested_dir
        else:
            raise FileNotFoundError(f"Dataset directory '{base_dir}' not found.")


        # Load the data using the determined data_root
        images, labels, gesture_map = load_data(data_root)
        
        if images.size == 0:
             print("\nError: No images were loaded. Please check the 'leapgestrecog' directory structure.")
             return

        print(f"\nSuccessfully loaded {len(images)} images.")
        
        # --- Data Preprocessing ---
        images = images.astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)
        labels = to_categorical(labels, num_classes=len(gesture_map))

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

        # --- Build the CNN Model ---
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(len(gesture_map), activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # --- Train the Model ---
        print("\nTraining the model...")
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        # --- Save the Model ---
        model.save('hand_gesture_model.h5')
        print("\nModel trained and saved as hand_gesture_model.h5")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the 'leapgestrecog' folder is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
