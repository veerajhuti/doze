import os
import cv2 as cv
import numpy as np
import dlib
import torch
import time
import base64
from scipy.spatial import distance
from imutils import face_utils
from model import NeuralNetwork
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

VERSION = "2026-06-27-1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# model

model_path = os.path.join(BASE_DIR, 'best_model.pth')
model = NeuralNetwork()
model.load_state_dict(torch.load(model_path))
model.eval()

transform = transforms.Compose([ # transform the live feed data
  transforms.Resize((28, 28)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

landmark_path = os.path.join(BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
if not os.path.exists(landmark_path):
  raise FileNotFoundError(f"Landmark file not found at {landmark_path}")

facial_landmark_detector = dlib.shape_predictor(landmark_path)
facial_feature_detector = dlib.get_frontal_face_detector()

ear_threshold = 0.18

# facial_landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
is_tracking = False
live_video = None
is_drowsy = False
duration_threshold = 47
ear_consec_frames = 15  # how many frames in a row indicates drowsiness

ear_counter = 0         # frame counter
model_drowsy_score = 0  # accumulated drowsiness score
last_reading = 0        # timestamp of last ML inference
# duration = 0

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

@app.get("/")
def read_root():
  return {
    "message": "API is running",
    "version": VERSION
  }

@app.post("/predict")
async def predict_route(request: Request):
  """
  Accepts JSON: { "image": "data:image/jpeg;base64,..." }
  Returns: { "is_drowsy": bool, "confidence": float }
  """

  global ear_counter, model_drowsy_score, last_reading

  prediction_interval = 0.5  # seconds between predictions
  model_drowsy_max = 15
  model_drowsy_min = 0
  model_drowsy_increase = 3
  model_drowsy_decrease = 5
  model_drowsy_threshold = 10
  
  current_reading = time.time()

  data = await request.json()
  image_b64 = data.get("image", "")

  if not image_b64.startswith("data:image"):
    return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

  # Decode base64 image
  try:
    header, encoded = image_b64.split(",", 1)
    img_bytes = base64.b64decode(encoded)
  except Exception:
    return JSONResponse(content={"error": "Base64 decode error"}, status_code=400)

  nparr = np.frombuffer(img_bytes, np.uint8)
  frame = cv.imdecode(nparr, cv.IMREAD_COLOR)

  if frame is None:
    return JSONResponse(content={"error": "Could not decode image"}, status_code=400)

  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  faces = facial_feature_detector(gray, 0)

  is_drowsy = False
  confidence = 0.0
  
  for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    crop = frame[y: y+h, x: x+w]
    crop_resized = cv.resize(crop, (128, 128))  # match training resolution
    
    # os.makedirs("debug_eyes", exist_ok=True)
    # timestamp = int(time.time() * 1000)
    # cv.imwrite(f"debug_eyes/{timestamp}_face.jpg", crop)
    # cv.imwrite(f"debug_eyes/{timestamp}_face_128.jpg", crop_resized)
    
    if current_reading - last_reading > prediction_interval:
      prediction, confidence = predict(crop_resized)
      print("Prediction:", prediction)
      print("Confidence:", confidence)
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
    print(f"EAR: {ear:.3f}")
    
    for (x_pt, y_pt) in np.concatenate((left_eye, right_eye), axis=0):
      cv.circle(frame, (x_pt, y_pt), 1, (0, 0, 255), -1)
      
    if ear < ear_threshold:
      ear_counter += 1
    else:
      ear_counter = 0
      
    ear_dec = ear_counter >= ear_consec_frames
    
    # CLASSES: 0=closed, 1=no_yawn, 2=open, 3=yawn  →  0 and 3 are drowsy states
    if prediction == 0 and confidence > 0.65:
      if ear < ear_threshold:
        model_drowsy_score += model_drowsy_increase  # closed + low EAR = confident drowsy
      else:
        model_drowsy_score -= 1  # model says closed but eyes look open
    elif prediction == 2 and confidence > 0.6:
      model_drowsy_score -= model_drowsy_decrease
    elif prediction == 1 and confidence > 0.6:  # no_yawn = clearly awake
      model_drowsy_score -= model_drowsy_decrease
    else:
      model_drowsy_score -= 1

    model_drowsy_score = max(model_drowsy_min, min(model_drowsy_score, model_drowsy_max))
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}, EAR: {ear:.2f}, Score: {model_drowsy_score}/{model_drowsy_threshold}, EAR_frames: {ear_counter}/{ear_consec_frames}")

    model_dec = model_drowsy_score >= model_drowsy_threshold
    if model_dec:
      model_drowsy_score = model_drowsy_threshold // 2
    is_drowsy = model_dec or ear_dec
    label = f"{'Drowsy' if is_drowsy else 'Awake'} ({confidence*100:.1f}%)"
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    
    cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_drowsy else (0, 255, 0), 2)
          
  return {"is_drowsy": is_drowsy, "confidence": confidence}

# LOCAL
if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=4000)
    
# if __name__ == "__main__":
#   import uvicorn
#   port = int(os.environ.get("PORT", 4000))
#   uvicorn.run("server:app", host="0.0.0.0", port=port)