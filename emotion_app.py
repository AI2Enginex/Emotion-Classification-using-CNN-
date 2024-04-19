import os
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, Response, flash

# Class to load model and required variables
class Load_Var:

    def __init__(self):
        # Load emotion detection model
        self.model = load_model("./emotions.h5")
        # Load Haar cascade for face detection
        self.face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        # Emotion labels
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

# Class for processing single image
class Image(Load_Var):

    def __init__(self, img_path):
        super().__init__()
        self.img = cv2.imread(img_path)  # Read image

    # Method to predict emotion from image
    def predict_result(self):
        try:
            faces = self.face_classifier.detectMultiScale(self.img)  # Detect faces
            for (x, y, w, h) in faces:
                roi_ = self.img[y:y+h, x:x+w]  # Extract region of interest
                roi_ = cv2.resize(roi_, (48, 48), interpolation=cv2.INTER_AREA)  # Resize for model
                if np.sum([roi_]) != 0:
                    roi = roi_.astype('float')/255.0
                    roi = np.expand_dims(roi, axis=0)
                    prediction = self.model.predict(roi)[0]
                    label = self.emotion_labels[prediction.argmax()]  # Get predicted emotion label
            return label
        except:
            return "error"

# Class for processing live video feed
class Camera_Frame(Load_Var):

    def __init__(self):
        super().__init__()
        self.camera = cv2.VideoCapture(0)  # Initialize camera

    # Method to capture frames from camera
    def frames(self):
        try:
            while True:
                ret, frame = self.camera.read()  # Read frame from camera
                faces = self.face_classifier.detectMultiScale(frame)  # Detect faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Draw rectangle around face
                    roi_ = frame[y:y+h, x:x+w]  # Extract region of interest
                    roi_ = cv2.resize(roi_, (48, 48), interpolation=cv2.INTER_AREA)  # Resize for model
                    if np.sum([roi_]) != 0:
                        roi = roi_.astype('float')/255.0
                        roi = np.expand_dims(roi, axis=0)
                        prediction = self.model.predict(roi)[0]
                        label = self.emotion_labels[prediction.argmax()]  # Get predicted emotion label
                        label_position = (x, y)
                        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield frame as byte stream
        except:
            return None

# Class for Flask application
class Flask_app:

    def __init__(self):
        self.app = Flask(__name__, template_folder='templates')  # Initialize Flask app
        self.app.secret_key = 'emotions'  # Set secret key for session management

# Class for predicting emotions from image using Flask
class Image_predict(Flask_app):

    def __init__(self):
        super().__init__()

    # Method to render index page
    def index(self):
        return render_template('index.html')

    # Method to handle prediction request
    def predict(self):
        if request.method == 'POST':
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)  # Save uploaded image
            lv = Image(file_path)
            preds = lv.predict_result()  # Predict emotion
            return preds  # Return predicted emotion
        return None

# Class for streaming live video using Flask
class Live_Video(Flask_app):

    def __init__(self):
        super().__init__()
        self.web_cam = Camera_Frame()  # Initialize camera frame

    # Method to render live camera page
    def live_camera(self):
        return render_template('live_camera.html')

    # Method to stream live video
    def video(self):
        cam_frm = self.web_cam.frames()
        if cam_frm:
            return Response(cam_frm, mimetype='multipart/x-mixed-replace; boundary=frame')  # Return video stream
        return render_template('live_camera.html')  # Render live camera page

    # Method to close camera
    def close_camera(self):
        while self.cam.isOpened():
           self.cam.release()  # Release camera object
        flash("Camera Object Killed....please restart the server")  # Flash message
        return render_template('live_camera.html')  # Render live camera page

# Class to run Flask application
class Run_Template(Image_predict, Live_Video):

    def __init__(self):
        super().__init__()

    # Method to run the Flask application
    def run_application(self):
        # Add URL rules for different routes
        self.app.add_url_rule('/', methods=['GET', 'POST'], view_func=self.index)
        self.app.add_url_rule('/predict', methods=['GET', 'POST'], view_func=self.predict)
        self.app.add_url_rule('/webcam', methods=['GET', 'POST'], view_func=self.live_camera)
        self.app.add_url_rule('/release', methods=['GET', 'POST'], view_func=self.close_camera)
        self.app.add_url_rule('/video', methods=['GET', 'POST'], view_func=self.video)
        self.app.run(debug=True)  # Run Flask app in debug mode

# Main block
if __name__ == '__main__':
    run = Run_Template()  # Create instance of Run_Template class
    run.run_application()  # Run the Flask application
