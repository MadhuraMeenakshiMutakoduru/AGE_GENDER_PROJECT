import cv2
import numpy as np

# Paths to models
age_model = 'age_net.caffemodel'
age_proto = 'deploy_age.prototxt'
gender_model = 'gender_net.caffemodel'
gender_proto = 'deploy_gender.prototxt'

# Load models
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels
AGE_LABELS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LABELS = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(0)

# Mean values used during training (for Caffe)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LABELS[gender_preds[0].argmax()]

        # Age Prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LABELS[age_preds[0].argmax()]

        # Drawing
        label = f'{gender}, {age}'
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Age & Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
