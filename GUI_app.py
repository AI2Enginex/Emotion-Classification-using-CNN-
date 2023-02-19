
"""
python GUI application for detecting Emotions for Image and Live Camera

model built using ---- > CNN
code for model building check ------> emotions_prediction.py 

"""


import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
from keras.preprocessing.image import img_to_array



class Detect_emotions:

    def __init__(self):

        self.emotion_labels = ['Angry','Happy','Neutral', 'Sad']
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.classifier = load_model('D:\CNN_Projects\Emotions\emotions.h5')
        self.cap = cv2.VideoCapture(0)

    def predeict_emotions(self):

        while True:
            _, frame = self.cap.read()
            faces = self.face_classifier.detectMultiScale(frame)
            for (x,y,w,h) in faces:

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = frame[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
                
                    prediction = self.classifier.predict(roi)[0]
                    label=self.emotion_labels[prediction.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                else:
                    cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            cv2.imshow('Emotion Detector',frame)
            cv2.moveWindow('Emotion Detector' , 400,200)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


class Load_Image:

    def __init__(self,image):
        
        self.image = cv2.imread(image)
        self.emotion_labels = ['Angry','Happy','Neutral', 'Sad']
        self.classifier = load_model('D:\CNN_Projects\Emotions\emotions.h5')
        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def predict_emotion(self):

        faces = self.face_classifier.detectMultiScale(self.image)
        for (x,y,w,h) in faces:

            cv2.rectangle(self.image,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = self.image[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                prediction = self.classifier.predict(roi)[0]
                label=self.emotion_labels[prediction.argmax()]
                label_position = (x,y)
                cv2.putText(self.image,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(self.image,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    def display_image(self):
        
        self.predict_emotion()
        resized_image = cv2.resize(self.image , (700,700))
        cv2.imshow('Emotion Detector',resized_image)
        cv2.moveWindow('Emotion Detector' , 400,60)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class Camera_canvas:

    def __init__(self):

        self.top = tk.Tk()
        self.canvas = tk.Canvas(self.top, width=400, height=200 , background="light blue")
        self.top.eval('tk::PlaceWindow . center')
        tk.Label(self.top, text="Emotion Detection",font=("Helvetica", 14) , background="light blue").place(x=120, y=50)
        self.canvas.pack()

    def run_app(self):

        def run_code():
            de = Detect_emotions()
            de.predeict_emotions()
        
        tk.Button(self.top, text="open camera", bg='white',
                  fg='Black', command=run_code).place(x=160, y=90)
        self.top.mainloop()

class Image_canvas:

    def __init__(self):

        self.top = tk.Tk()
        self.canvas = tk.Canvas(self.top, width=400, height=200 , background="light blue")
        self.top.eval('tk::PlaceWindow . center')
        tk.Label(self.top, text="Select an Image",font=("Helvetica", 14) , background="light blue").place(x=120, y=50)
        tk.Label(self.top, text="Enter Path for Image : ",font=("Helvetica", 10) , background="light blue").place(x=50, y=87)
        self.entry_widget = tk.Entry(self.top)
        self.canvas.create_window(250, 100, window=self.entry_widget)
        self.canvas.pack()

    def run_app(self):

        def run_code():

            entry_value = self.entry_widget.get()
            if os.path.exists(entry_value):
                la = Load_Image(entry_value)
                la.display_image()
            else:

                messagebox.showwarning(title="Error",message="image not found")
                
                
        
        tk.Button(self.top, text="open camera", bg='white',
                  fg='Black', command=run_code).place(x=160, y=120)
        self.top.mainloop()

class Outer_Canvas:

    def __init__(self):
        
        
        self.outer = tk.Tk()
        self.canvas = tk.Canvas(self.outer, width=400, height=200 , background="light blue")
        self.outer.eval('tk::PlaceWindow . center')
        tk.Label(self.outer, text="Emotion Detection",font=("Helvetica", 14) , background="light blue").place(x=120, y=50)
        self.canvas.pack()
        

    def open_image(self):

        
        def display_image():

            build_window = Image_canvas()
            build_window.run_app()

        def display_video():

            cc = Camera_canvas()
            cc.run_app()

        tk.Button(self.outer, text="Select an Image", bg='white',
                  fg='Black', command=display_image).place(x=100, y=90)
        tk.Button(self.outer, text="Open Webcam", bg='white',
                  fg='Black', command=display_video).place(x=220, y=90)
        self.outer.mainloop()





if __name__ == "__main__":

    oc = Outer_Canvas()
    oc.open_image()




        