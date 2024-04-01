import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

face_cascade = cv2.CascadeClassifier(r"D:\Machine Learning Stuff\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
model = load_model(r"D:\vs code\python\DeepLearning\Projects\Facial_landmark_detection\facial_detection_model_2.h5")

img_main = Image.open(r"D:\Datasets\facial_landmark_dataset\train\00000.png")
img_gray = img_main.convert('L')
img = img_gray.resize((224,224))
img_main = np.array(img_main)
img = np.array(img)
img = img.astype(np.float32) / 255.0

img = np.expand_dims(img,axis=-1)
img = np.expand_dims(img,axis=0)

print(type(img))
predictions = model.predict(img)[0]

print(len(predictions))

for point in range(0, len(predictions), 2):
    # Extract x and y coordinates from predictions
    x = int(predictions[point])
    y = int(predictions[point + 1])

    # Draw a circle at (x, y) with radius 2 and blue color
    cv2.circle(img_main, (x, y), 2, (255, 255, 0), -1)

# Display the image with circles drawn
cv2.imshow("main image", img_main)
cv2.waitKey(0)
cv2.destroyAllWindows()