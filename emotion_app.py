import os
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, Response , flash


class Load_Var:

    def __init__(self):

        self.model = load_model("./emotions.h5")
        self.face_classifier = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']


class Image(Load_Var):

    def __init__(self, img_path):

        super().__init__()
        self.img = cv2.imread(img_path)

    def predict_result(self):

        try:

            faces = self.face_classifier.detectMultiScale(self.img)

            for (x, y, w, h) in faces:
                roi_ = self.img[y:y+h, x:x+w]
                roi_ = cv2.resize(roi_, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_]) != 0:
                    roi = roi_.astype('float')/255.0
                    roi = np.expand_dims(roi, axis=0)
                    prediction = self.model.predict(roi)[0]
                    label = self.emotion_labels[prediction.argmax()]

            return label
        except:
            return "error"


class Camera_Frame(Load_Var):

    def __init__(self):

        super().__init__()

        self.camera = cv2.VideoCapture(0)

    def frames(self):
        
        try:
            while True:
                ret, frame = self.camera.read()
            
                

                faces = self.face_classifier.detectMultiScale(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    roi_ = frame[y:y+h, x:x+w]
                    roi_ = cv2.resize(roi_, (48, 48),
                                        interpolation=cv2.INTER_AREA)

                    if np.sum([roi_]) != 0:
                        roi = roi_.astype('float')/255.0

                        roi = np.expand_dims(roi, axis=0)

                        prediction = self.model.predict(roi)[0]

                        label = self.emotion_labels[prediction.argmax()]

                        label_position = (x, y)

                        cv2.putText(frame, label, label_position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, 'No Faces', (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()


                yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except:
            return None
                
class Flask_app:

    def __init__(self):

        self.app = Flask(__name__, template_folder='templates')
        self.app.secret_key = 'emotions'


class Image_predict(Flask_app):

    def __init__(self):

        super().__init__()

    def index(self):

        return render_template('index.html')

    def predict(self):

        if request.method == 'POST':
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            lv = Image(file_path)
            preds = lv.predict_result()
            return preds
        return None

class Live_Video(Flask_app):

    def __init__(self):

        super().__init__()

        self.web_cam = Camera_Frame()
        self.cam = self.web_cam.camera

    def live_camera(self):

        return render_template('live_camera.html')

    def video(self):
        
        cam_frm = self.web_cam.frames()

        if cam_frm:
            return Response(cam_frm, mimetype='multipart/x-mixed-replace; boundary=frame')
        return render_template('live_camera.html')
    
    def close_camera(self):

        while self.cam.isOpened():
           self.cam.release()
       
        flash("Camera Objeck Killed....please restart the server")
        return render_template('live_camera.html')

   

class Run_Template(Image_predict,Live_Video):

    def __init__(self):
        super().__init__()
    
    def run_application(self):

        self.app.add_url_rule(
            '/', methods=['GET', 'POST'], view_func=self.index)
        self.app.add_url_rule(
            '/predict', methods=['GET', 'POST'], view_func=self.predict)
        self.app.add_url_rule(
            '/webcam', methods=['GET', 'POST'], view_func=self.live_camera)
        self.app.add_url_rule(
            '/release', methods=['GET', 'POST'], view_func=self.close_camera)
        self.app.add_url_rule(
            '/video', methods=['GET', 'POST'], view_func=self.video)
        
        
        
        self.app.run(debug=True)

        


if __name__ == '__main__':

    run = Run_Template()
    run.run_application()
