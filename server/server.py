# from flask import Flask, Response, jsonify
# from flask_cors import CORS, cross_origin
import os
import cv2 as cv
import numpy as np
import dlib
import torch
import time
from scipy.spatial import distance
from imutils import face_utils
from model import NeuralNetwork
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'best_model.pth')
model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([ # transform the live feed data
  transforms.Resize((28, 28)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# global variables

landmark_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
if not os.path.exists(landmark_path):
  raise FileNotFoundError(f"Landmark file not found at {landmark_path}")
facial_landmark_detector = dlib.shape_predictor(landmark_path)

is_tracking = False
live_video = None
is_drowsy = False
# duration = 0
facial_feature_detector = dlib.get_frontal_face_detector()
# facial_landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
ear_threshold = 0.18
duration_threshold = 47
ear_consec_frames = 15  # how many frames in a row indicates drowsiness
ear_counter = 0  # frame counter

# new face/eye detector
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# EAR formula

def eye_aspect_ratio(eye):
  point_2_minus_6 = distance.euclidean(eye[1], eye[5])
  point_3_minus_5 = distance.euclidean(eye[2], eye[4])
  point_1_minus_4 = distance.euclidean(eye[0], eye[3])

  ratio = (point_2_minus_6 + point_3_minus_5) / (2.0 * point_1_minus_4)

  return ratio

# new check drowsiness

def predict(face):
  if face is None or face.size == 0:
    return -1, 0.0
  try:
    img = Image.fromarray(cv.cvtColor(face, cv.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
      output = model(input_tensor)
      probability = torch.softmax(output, dim=1)
      prediction = torch.argmax(probability, dim=1).item()
      confidence = probability[0][prediction].item()
    return prediction, confidence
  
  except Exception as e:
    print("Error in prediction:", e)
  return -1, 0.0


      # old logic
      
      # for face in faces:
      #   landmarks = facial_landmark_detector(gray, face)
      #   landmarks = face_utils.shape_to_np(landmarks)

      #   (x, y, w, h) = face_utils.rect_to_bb(face)
      #   cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      #   for (x, y) in landmarks:
      #     cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
      #   if len(landmarks) > 0:
      #     left_eye = landmarks[42:48]
      #     right_eye = landmarks[36:42]
  
      #     left_ear = eye_aspect_ratio(left_eye)
      #     right_ear = eye_aspect_ratio(right_eye)
          
      #     ear = (left_ear + right_ear) / 2.0

      #     if ear < ear_threshold:
      #       duration += 1
      #     else:
      #       duration = 0

      #     if duration > duration_threshold:
      #       is_drowsy = True
      
def webcam_display():
  global live_video, is_tracking, is_drowsy

  if live_video is None:
    live_video = cv.VideoCapture(0)

  last_reading = 0
  prediction_interval = 0.5  # seconds between predictions
  ear_counter = 0
  model_drowsy_score = 0
  model_drowsy_max = 15
  model_drowsy_min = 0
  model_drowsy_increase = 3
  model_drowsy_decrease = 5
  model_drowsy_threshold = 10

  while is_tracking:
    ret, frame = live_video.read()
    if not ret:
      break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facial_feature_detector(gray, 0)
    current_reading = time.time()
    
    for face in faces:
      x, y, w, h = face.left(), face.top(), face.width(), face.height()
      # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      crop = frame[y: y+h, x: x+w]
      crop_resized = cv.resize(crop, (128, 128))  # match training resolution

      if current_reading - last_reading > prediction_interval:
        prediction, confidence = predict(crop_resized)
        last_reading = current_reading
      else:
        prediction, confidence = -1, 0.0
    
      landmarks = facial_landmark_detector(gray, face)
      landmarks = face_utils.shape_to_np(landmarks)
      
      left_eye = landmarks[42:48]
      right_eye = landmarks[36:42]
      left_ear = eye_aspect_ratio(left_eye)
      right_ear = eye_aspect_ratio(right_eye)
      
      ear = (left_ear + right_ear) / 2.0
      
      for (x_pt, y_pt) in np.concatenate((left_eye, right_eye), axis=0):
        cv.circle(frame, (x_pt, y_pt), 1, (0, 0, 255), -1)
        
      if ear < ear_threshold:
        ear_counter += 1
      else:
        ear_counter = 0
        
      ear_dec = ear_counter >= ear_consec_frames
      
      # prediction
      if prediction in [0, 3] and confidence > 0.3:
        model_drowsy_score += model_drowsy_increase
      elif confidence < 0.4:
        model_drowsy_score += 1
      else:
        model_drowsy_score -= 1

      model_drowsy_score = max(model_drowsy_min, min(model_drowsy_score, model_drowsy_max))
      # print(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Score: {model_drowsy_score}")

      model_dec = model_drowsy_score >= model_drowsy_threshold
      is_drowsy = model_dec or ear_dec
      label = f"{'Drowsy' if is_drowsy else 'Awake'} ({confidence*100:.1f}%)"
      color = (0, 0, 255) if is_drowsy else (0, 255, 0)
      
      cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
      cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_drowsy else (0, 255, 0), 2)

    ret, jpeg = cv.imencode('.jpg', frame)
    if ret:
      frame_bytes = jpeg.tobytes()
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    else:
      yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'  # empty frame  
  
  if live_video is not None:
    live_video.release()
    live_video = None
          
@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.get('/webcam')
def start_tracking():
    global is_tracking
    is_tracking = True
    return StreamingResponse(webcam_display(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/stop_webcam')
def stop_tracking():
    global is_tracking
    is_tracking = False
    return JSONResponse(content={"status": "Webcam Stopped"})

@app.get('/check_drowsiness')
def check_drowsiness():
    global is_drowsy
    return JSONResponse(content={"is_drowsy": is_drowsy})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=4000, debug=True)
