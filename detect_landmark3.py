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
    faces_keypoints = []
    
    # Loop through faces
    for (x,y,w,h) in faces:
        
        # Crop Faces
        face = gray[y:y+h, x:x+w]
        # Scale Faces to 96x96
        scaled_face = cv2.resize(face, (224,224), 0, 0, interpolation=cv2.INTER_AREA)

        # Normalize images to be between 0 and 1
        input_image = scaled_face / 255

        # Format image to be the correct shape for the model
        input_image = np.expand_dims(input_image, axis = 0)
        input_image = np.expand_dims(input_image, axis = -1)

        # Use model to predict keypoints on image
        face_points = model.predict(input_image)[0]

        # Adjust keypoints to coordinates of original image
        face_points[0::2] = face_points[0::2] * w/2 + w/2 + x
        face_points[1::2] = face_points[1::2] * h/2 + h/2 + y
        faces_keypoints.append(face_points)
        # Plot facial keypoints on image
    for point in range(len(faces_keypoints)):
        print(faces_keypoints[point])
        cv2.circle(image_copy, (int(face_points[point]/20),int(face_points[point + 1]/20)), 2, (255, 255, 0), -1)

    cv2.imshow('Screen with facial Keypoints predicted',image_copy)        
           
    if cv2.waitKey(1) & 0xFF == ord("q"):   
        break