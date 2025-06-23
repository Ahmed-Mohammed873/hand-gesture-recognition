# gesture_gui.py
import tkinter as tk
from tkinter import font
import cv2
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

class GestureApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Load the trained model
        try:
            self.model = load_model('hand_gesture_model.h5')
            print("Model loaded successfully.")
        except IOError:
            print("Model 'hand_gesture_model.h5' not found. Please run the training script first.")
            self.window.destroy()
            return
            
        # Define the gesture names based on the training script's mapping
        self.gesture_names = {0: 'palm', 1: 'l', 2: 'fist', 3: 'fist_moved', 4: 'thumb', 
                              5: 'index', 6: 'ok', 7: 'palm_moved', 8: 'c', 9: 'down'}

        # Start the video source (webcam)
        self.vid = cv2.VideoCapture(0)

        # Create a canvas that can fit the video source
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Create a label for the predicted gesture
        self.prediction_label = tk.Label(window, text="Predicted Gesture: -", font=("Helvetica", 20))
        self.prediction_label.pack(pady=20)
        
        # Button to start/stop
        self.btn_text = tk.StringVar()
        self.btn_text.set("Start Prediction")
        self.btn_toggle = tk.Button(window, textvariable=self.btn_text, width=20, command=self.toggle_prediction, font=("Helvetica", 14))
        self.btn_toggle.pack(pady=10)

        self.is_predicting = False
        self.update()

        self.window.mainloop()

    def toggle_prediction(self):
        self.is_predicting = not self.is_predicting
        self.btn_text.set("Stop Prediction" if self.is_predicting else "Start Prediction")


    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # If prediction is on, process the frame
            if self.is_predicting:
                # Preprocess the frame for the model
                processed_image = self.preprocess_frame(frame)
                
                # Make a prediction
                prediction = self.model.predict(processed_image)
                predicted_gesture_index = np.argmax(prediction)
                predicted_gesture_name = self.gesture_names.get(predicted_gesture_index, "Unknown")
                
                # Update the prediction label
                self.prediction_label.config(text=f"Predicted Gesture: {predicted_gesture_name}")
            else:
                 self.prediction_label.config(text="Prediction Paused")


            # Convert the frame to a format that tkinter can use
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Repeat every 15 milliseconds
        self.window.after(15, self.update)
        
    def preprocess_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to match model's expected input
        resized = cv2.resize(gray, (128, 128))
        # Normalize
        normalized = resized / 255.0
        # Reshape for the model
        reshaped = np.reshape(normalized, (1, 128, 128, 1))
        return reshaped

    def __del__(self):
        # Release the video source when the object is destroyed
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
if __name__ == "__main__":
    GestureApp(tk.Tk(), "Hand Gesture Recognition")