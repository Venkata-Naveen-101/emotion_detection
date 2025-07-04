import os
import cv2
import numpy as np
import warnings
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

warnings.filterwarnings("ignore")

# Optional: Force TensorFlow to run on CPU (helpful on AMD systems)
tf.config.set_visible_devices([], 'GPU')

# Load the pre-trained model
model = load_model("best_model.h5")

# Load Haar cascade classifier
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue

    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224))

        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial Emotion Analysis', resized_img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
