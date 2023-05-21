
# .\venv\Scripts\activate
# python -m flask --app .\app.py run

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import mediapipe as mp
import base64

mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# variables
# Landmark points corresponding to left eye
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
# flatten and remove duplicates
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))

# Landmark points corresponding to right eye
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))

# Combined for plotting - Landmark points for both eye
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)

# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Load the model
model = load_model('./model.h5')

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    return []
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files.get('image')
    # Assuming the input data is a base64 encoded image
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    plt.imshow(image)
    plt.show()
    plt.close()
    preprocessed_image = preprocess_image(image)
    if (preprocessed_image is None):
        return []

    predictions = model.predict(preprocessed_image)
    print(predictions)
    # Return the prediction as json
    return jsonify(predictions.tolist())


def preprocess_image(image):
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    land_face_array = None
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = img[y:y + h, x:x + w]
        land_face_array = landmarks(roi_color)

    if (land_face_array is None):
        return None

    plt.imshow(land_face_array)
    plt.show()
    plt.close()

    IMG_SIZE = 145
    resized_array = cv2.resize(land_face_array, (IMG_SIZE, IMG_SIZE))
    preprocessed_image = np.array(resized_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    preprocessed_image = preprocessed_image / 255.0

    return preprocessed_image


imgH, imgW, _ = 0, 0, 0


def landmarks(image):
    resized_array = []
    IMG_SIZE = 145
    image = np.ascontiguousarray(image)
    imgH, imgW, _ = image.shape

    with mp_facemesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1, 
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5, ) as face_mesh:

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                resized_array = draw(img_dt=image.copy(), face_landmarks=face_landmarks)
    return resized_array


IMG_SIZE = 145
i = 0


def draw(
        *,
        img_dt,
        img_eye_lmks=None,
        img_eye_lmks_chosen=None,
        face_landmarks=None,
        ts_thickness=1,
        ts_circle_radius=2,
        lmk_circle_radius=3,
        name="1",
):
    # For plotting Face Tessellation
    image_drawing_tool = img_dt

    # For plotting all eye landmarks
    image_eye_lmks = img_dt.copy() if img_eye_lmks is None else img_eye_lmks

    # For plotting chosen eye landmarks
    img_eye_lmks_chosen = img_dt.copy() if img_eye_lmks_chosen is None else img_eye_lmks_chosen

    # Initializing drawing utilities for plotting face mesh tessellation
    connections_drawing_spec = mp_drawing.DrawingSpec(
        thickness=ts_thickness,
        circle_radius=ts_circle_radius,
        color=(255, 255, 255)
    )

    # Draw landmarks on face using the drawing utilities.
    mp_drawing.draw_landmarks(
        image=image_drawing_tool,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=connections_drawing_spec,
    )

    # Get the object which holds the x, y, and z coordinates for each landmark
    landmarks = face_landmarks.landmark

    # Iterate over all landmarks.
    # If the landmark_idx is present in either all_idxs or all_chosen_idxs,
    # get the denormalized coordinates and plot circles at those coordinates.

    for landmark_idx, landmark in enumerate(landmarks):
        if landmark_idx in all_idxs:
            pred_cord = denormalize_coordinates(landmark.x,
                                                landmark.y,
                                                imgW, imgH)
            cv2.circle(image_eye_lmks,
                       pred_cord,
                       lmk_circle_radius,
                       (255, 255, 255),
                       -1
                       )
        if landmark_idx in all_chosen_idxs:
            pred_cord = denormalize_coordinates(landmark.x,
                                                landmark.y,
                                                imgW, imgH)
            cv2.circle(img_eye_lmks_chosen,
                       pred_cord,
                       lmk_circle_radius,
                       (255, 255, 255),
                       -1
                       )

    # resized_array = cv2.resize(image_drawing_tool, (IMG_SIZE, IMG_SIZE))
    return image_drawing_tool

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
