from flask import Flask, Response
from flask_cors import CORS
import cv2 as cv
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils

app = Flask(__name__)
CORS(app, origins=["https://veerajhuti.github.io"])

# global variables

is_tracking = False
live_video = None
is_drowsy = False
duration = 0
facial_feature_detector = dlib.get_frontal_face_detector()
facial_landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
ear_threshold = 0.18
duration_threshold = 47

# EAR formula

def eye_aspect_ratio(eye):
    point_2_minus_6 = distance.euclidean(eye[1], eye[5])
    point_3_minus_5 = distance.euclidean(eye[2], eye[4])
    point_1_minus_4 = distance.euclidean(eye[0], eye[3])

    ratio = (point_2_minus_6 + point_3_minus_5) / (2.0 * point_1_minus_4)

    return ratio

def webcam_display():
    global live_video
    while True:

# face is being tracked
        if is_tracking:
            if live_video is None:
                live_video = cv.VideoCapture(0)

            ret, frame = live_video.read()
            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = facial_feature_detector(gray, 2)
            
            for face in faces:
                landmarks = facial_landmark_detector(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                for (x, y) in landmarks:
                    cv.circle(frame, (x, y), 1, (0, 0, 255), -1)
                
                if len(landmarks) > 0:
                    left_eye = landmarks[42:48]
                    right_eye = landmarks[36:42]
            
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    
                    ear = (left_ear + right_ear) / 2.0

                    if ear < ear_threshold:
                        duration += 1
                    else:
                        duration = 0

                    if duration > duration_threshold:
                        is_drowsy = True

            ret, jpeg = cv.imencode('.jpg', frame)
            if ret:
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# face is not being tracked

        else:
            if live_video is not None:
                live_video.release()
                live_video = None
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'  # empty frame  

@app.route('/webcam')
def start_tracking():
    global is_tracking
    is_tracking = True
    return Response(webcam_display(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam')
def stop_tracking():
    global is_tracking
    is_tracking = False
    return Response(webcam_display(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_drowsiness')
def check_drowsiness():
    global is_drowsy
    return jsonify({"is_drowsy": is_drowsy})
    
if __name__ == '__main__':
    app.run(debug=True)