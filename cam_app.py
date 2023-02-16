import cv2
import numpy as np
import tkinter as tk
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


class Build_canvas:

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
                  fg='Black', command=run_code).place(x=120, y=80)
        self.top.mainloop()





if __name__ == "__main__":

    canvas = Build_canvas()
    canvas.run_app()

   
    
    
    



    

    

    

