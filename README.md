# Doze

<img width="2870" height="1560" alt="landing" src="https://github.com/user-attachments/assets/37c96553-cec1-4a75-90de-31d221c93a2b" />

## Description

**Doze** is a real-time drowsiness detector that watches your eyes through your webcam and alerts you if it notices signs of tiredness or prolonged eye closure. It uses a neural network I built and trained to recognize when your eyes are closing or open.

The model is built with PyTorch and trained on a dataset of eye images. It uses techniques like batch normalization, dropout, and the Adam optimizer to get good accuracy. Everything runs locally on your computer—no data leaves your device.

This project combines deep learning and computer vision to help keep you alert and safe.

## Features

- Real-time eye closure detection using the Eye Aspect Ratio (EAR) algorithm.
- Webcam footage is processed locally — no personal data is uploaded or stored.
- Alerts the user when prolonged eye closure is detected, helping prevent drowsiness-related risks.
- Simple and intuitive React frontend with FastAPI backend for processing.

## Usage

The app is deployed and ready to use here:

[https://client-djb4.onrender.com](https://client-djb4.onrender.com)

1. Open the URL in your browser.
2. Click **Start Tracking** to begin eye closure detection.
3. If drowsiness is detected, an alert will notify you.
4. Click **Stop Tracking** to stop detection and release the webcam.

No personal data is collected or stored by this application.

## Roadmap

Future improvements may include:

- Fine tuning the model.
- Integrate health tips using Healthline API or similar resources to provide personalized health tips based on detected eye closure and fatigue.
- Add physical and/or wearable assistive devices to get a more holistic view of the user’s sleep patterns, movement, and fatigue levels.
- Incorporate real-time data scraping that pulls updated health information, research papers, or tips based on detected user behaviour.

## References

This project leveraged resources and knowledge from:

- [Eye Aspect Ratio (EAR) for Blink Detection](https://www.mdpi.com/2079-9292/11/19/3183)
- [Sleep Apnea Detection and Technology](https://www.sciencedirect.com/science/article/pii/S2667241322000039)
- [dlib Face Landmark Detector](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [Driver Drowsiness Dataset (Kaggle)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [Lukas’ Machine Learning Class](https://github.com/lukas/ml-class)
- [Towards Data Science tutorials](https://towardsdatascience.com/building-a-neural-network-from-scratch-8f03c5c50adc/)
- [PyTorch official documentation](https://pytorch.org/get-started/locally/)
  
## Author

[Veera Jhuti](https://github.com/veerajhuti)
