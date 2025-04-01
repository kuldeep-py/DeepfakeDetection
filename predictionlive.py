
from flask import Flask, render_template, Response
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

app = Flask(__name__)

model1 = load_model(r'E:\\DeepFake\\MAJOR PROJECT\\25_3.h5')

def predict_image(image):
    resized_img = cv2.resize(image, (224, 224))  
    input_img = resized_img / 255.0  
    input_img = np.expand_dims(input_img, axis=0) 
  
    predictions1 = model1.predict(input_img)
    pred = "REAL" if predictions1 >= 0.5 else "FAKE"
    return pred


def webcam_feed():
    camera = cv2.VideoCapture(0) 
    while True:
      
        ret, frame = camera.read()
        if not ret:
            break
        
      
        pred = predict_image(frame)
        
        cv2.putText(frame, f'Prediction: {pred}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('redirectcamera.html')

@app.route('/video_feed')
def video_feed():
    return Response(webcam_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(use_reloader=False)
