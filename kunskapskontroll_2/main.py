from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(
    r'haarcascade_frontalface_default.xml')

emotion_classifier = load_model(
    r'model.h5', compile=False)

emotion_classifier.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

gender_classifier = load_model(
    r'age_gender_model.h5', compile=False)

gender_classifier.compile(
    loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']

gender_dict = {0: "Male", 1: "Female"}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            emo_prediction = emotion_classifier.predict(roi)[0]
            emo_label = emotion_labels[emo_prediction.argmax()]

            gender_prediction = gender_classifier.predict(roi)

            prediction_text = emo_label + " " + \
                gender_dict[round(gender_prediction[0][0][0])]

            label_position = (x, y-10)
            cv2.putText(frame, prediction_text, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Emotion Detector', frame)
    # Press Esc to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # Click "X" button to exit
    if cv2.getWindowProperty('Emotion Detector', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
