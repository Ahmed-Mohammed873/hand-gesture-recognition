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
    gestures = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
    gesture_map = {name: i for i, name in enumerate(gestures)}
    print(f"Gesture mapping used for training: {gesture_map}")
    print(f"Loading data from: {data_dir}")

    for subject_dir in os.listdir(data_dir):
        subject_path = os.path.join(data_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        for gesture_dir in os.listdir(subject_path):
            gesture_path = os.path.join(subject_path, gesture_dir)
            if not os.path.isdir(gesture_path):
                continue
            
            try:
                gesture_name = '_'.join(gesture_dir.split('_')[1:])
                if gesture_name not in gesture_map:
                    continue
                gesture_label = gesture_map[gesture_name]
            except IndexError:
                continue

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
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, 'leapgestrecog')
        data_root = base_dir
        
        if os.path.exists(base_dir):
            dir_contents = os.listdir(base_dir)
            if len(dir_contents) == 1 and os.path.isdir(os.path.join(base_dir, dir_contents[0])):
                nested_dir = os.path.join(base_dir, dir_contents[0])
                print(f"Detected a nested directory. Using '{nested_dir}' as the data root.")
                data_root = nested_dir
        else:
            raise FileNotFoundError(f"Dataset directory '{base_dir}' not found.")


        images, labels, gesture_map = load_data(data_root)
        
        if images.size == 0:
             print("\nError: No images were loaded. Please check the 'leapgestrecog' directory structure.")
             return

        print(f"\nSuccessfully loaded {len(images)} images.")
        
        images = images.astype('float32') / 255.0
        images = np.expand_dims(images, axis=-1)
        labels = to_categorical(labels, num_classes=len(gesture_map))

        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

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

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        print("\nTraining the model...")
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

        model.save('hand_gesture_model.h5')
        print("\nModel trained and saved as hand_gesture_model.h5")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the 'leapgestrecog' folder is in the same directory as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
