# Get webcam
import cv2
import numpy as np
from keras.models import load_model

face_cascade = cv2.CascadeClassifier(r"D:\Machine Learning Stuff\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
model = load_model(r"D:\vs code\python\DeepLearning\Projects\Facial_landmark_detection\facial_detection_model_2.h5")


camera = cv2.VideoCapture(0)

while True:
    # Read data from the webcam
    _, image = camera.read() 
    image_copy = np.copy(image)   
    
    # Convert RGB image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    # Identify faces in the webcam using haar cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_landmarks = []

    for (x,y,w,h) in faces:
        cropped_image = gray[y:y+h, x:x+w]
        resized_image = cv2.resize(cropped_image, (224, 224))
        img = np.array(resized_image)
        img = np.array(img)
        img = img.astype(np.float32) / 255.0

        img = np.expand_dims(img,axis=-1)
        img = np.expand_dims(img,axis=0)
        coordinates = model.predict(img)[0]
        for point in range(0, len(coordinates), 2):
            # Extract x and y coordinates from predictions
            x = int(coordinates[point])
            y = int(coordinates[point + 1])

    # Draw a circle at (x, y) with radius 2 and blue color
            cv2.circle(image, (x, y), 2, (255, 255, 0), -1)

        cv2.imshow("image",image)
    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break