import os
import cv2
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array
from flask import Flask, request, render_template


class Load_var:

    def __init__(self):

        self.model = load_model("./emotions.h5")
        self.face_classifier = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')
        self.emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

    def predict_result(self, img_path):

        try:
            img = cv2.imread(img_path)
            faces = self.face_classifier.detectMultiScale(img)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                roi_ = img[y:y+h, x:x+w]
                roi_ = cv2.resize(roi_, (48, 48),interpolation=cv2.INTER_AREA)
                
                if np.sum([roi_]) != 0:
                    roi = roi_.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = self.model.predict(roi)[0]
                    label = self.emotion_labels[prediction.argmax()]

            return label
        except:
            return "error"


class Flask_app:

    def __init__(self):

        self.lv = Load_var()
        self.app = Flask(__name__, template_folder='templates')

    def index(self):
        return render_template('index.html')

    def upload(self):
        if request.method == 'POST':
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            preds = self.lv.predict_result(file_path)
            return preds
        return None

    def run_application(self):

        
        self.app.add_url_rule(
            '/', methods=['GET', 'POST'], view_func=self.index)
        self.app.add_url_rule(
            '/predict', methods=['GET', 'POST'], view_func=self.upload)
        self.app.run(debug=True)


if __name__ == '__main__':

    fa = Flask_app()
    fa.run_application()
