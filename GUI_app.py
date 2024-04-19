import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from keras.models import load_model

# Class to hold parameters and model for emotion detection
class Parameters:

    def __init__(self):
        # List of emotion labels
        self.labels = ['Angry', 'Happy', 'Neutral', 'Sad']
        # Haar cascade for detecting faces
        self.face = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        # Load pre-trained emotion detection model
        self.model = load_model('./emotions.h5')

# Class for detecting emotions from live camera feed
class Detect_emotions(Parameters):

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)  # Open default camera

    def predeict_emotions(self):
        while True:
            ret, frame = self.cap.read()  # Read frame from camera
            faces = self.face.detectMultiScale(frame)  # Detect faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Draw rectangle around face
                roi_ = frame[y:y+h, x:x+w]
                roi_ = cv2.resize(roi_, (48, 48), interpolation=cv2.INTER_AREA)  # Resize image for model
                if np.sum([roi_]) != 0:
                    roi = roi_.astype('float')/255.0
                    roi = np.expand_dims(roi, axis=0)
                    prediction = self.model.predict(roi)[0]  # Make prediction using model
                    label = self.labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Emotion Detector', frame)  # Display frame with detected emotions
            cv2.moveWindow('Emotion Detector', 400, 200)  # Move window to specified location
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
                break
        self.cap.release()  # Release camera
        cv2.destroyAllWindows()  # Close all windows

# Class for loading and predicting emotions from a single image
class Load_Image(Parameters):

    def __init__(self, image):
        super().__init__()
        self.image = cv2.imread(image)  # Load image

    def predict_emotion(self):
        faces = self.face.detectMultiScale(self.image)  # Detect faces in the image
        for (x, y, w, h) in faces:
            cv2.rectangle(self.image, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Draw rectangle around face
            roi_ = self.image[y:y+h, x:x+w]
            roi_ = cv2.resize(roi_, (48, 48), interpolation=cv2.INTER_AREA)  # Resize image for model
            if np.sum([roi_]) != 0:
                roi = roi_.astype('float')/255.0
                roi = np.expand_dims(roi, axis=0)
                prediction = self.model.predict(roi)[0]  # Make prediction using model
                label = self.labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(self.image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(self.image, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', cv2.resize(self.image, (700, 700)))  # Display image with detected emotions
        cv2.moveWindow('Emotion Detector', 400, 80)  # Move window to specified location
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            cv2.destroyAllWindows()

# Class for creating GUI to open camera or select an image for emotion detection
class Outer_Canvas:

    def __init__(self):
        self.outer = tk.Tk()  # Create main window
        self.canvas = tk.Canvas(self.outer, width=400, height=200, background="light blue")
        self.outer.eval('tk::PlaceWindow . center')  # Center the window
        tk.Label(self.outer, text="Emotion Detection", font=("Helvetica", 14), background="light blue").place(x=120, y=50)
        self.canvas.pack()

    # Method to open image or camera
    def open_image(self):
        # Function to display image selection window
        def display_image():
            build_window = Image_canvas()
            build_window.run_app()
        # Function to open camera
        def display_video():
            cc = Camera_canvas()
            cc.run_app()

        tk.Button(self.outer, text="Select an Image", bg='white', fg='Black', command=display_image).place(x=100, y=90)
        tk.Button(self.outer, text="Open Webcam", bg='white', fg='Black', command=display_video).place(x=220, y=90)
        self.outer.mainloop()

# Class for GUI to select an image
class Image_canvas:

    def __init__(self):
        self.top = tk.Tk()  # Create window
        self.canvas = tk.Canvas(self.top, width=400, height=200, background="light blue")
        self.top.eval('tk::PlaceWindow . center')  # Center the window
        tk.Label(self.top, text="Select an Image", font=("Helvetica", 14), background="light blue").place(x=120, y=50)
        tk.Label(self.top, text="Enter Path for Image : ", font=("Helvetica", 10), background="light blue").place(x=50, y=87)
        self.entry_widget = tk.Entry(self.top)  # Entry widget for entering image path
        self.canvas.create_window(250, 100, window=self.entry_widget)
        self.canvas.pack()

    # Method to run the GUI
    def run_app(self):
        # Function to run code for selected image
        def run_code():
            entry_value = self.entry_widget.get()
            if os.path.exists(entry_value):
                la = Load_Image(entry_value)
                la.predict_emotion()
            else:
                messagebox.showwarning(title="Error", message="image not found")
        tk.Button(self.top, text="open camera", bg='white', fg='Black', command=run_code).place(x=160, y=120)
        self.top.mainloop()

# Class for GUI to open camera
class Camera_canvas:

    def __init__(self):
        self.top = tk.Tk()  # Create window
        self.canvas = tk.Canvas(self.top, width=400, height=200, background="light blue")
        self.top.eval('tk::PlaceWindow . center')  # Center the window
        tk.Label(self.top, text="Emotion Detection", font=("Helvetica", 14), background="light blue").place(x=120, y=50)
        self.canvas.pack()

    # Method to run the GUI
    def run_app(self):
        # Function to run code for camera feed
        def run_code():
            try:
                de = Detect_emotions()
                de.predeict_emotions()
            except Exception:
                messagebox.showwarning(title="Error", message="camera not found")
        tk.Button(self.top, text="open camera", bg='white', fg='Black', command=run_code).place(x=160, y=90)
        self.top.mainloop()

if __name__ == "__main__":
    oc = Outer_Canvas()  # Create main GUI window
    oc.open_image()  # Display main GUI window
