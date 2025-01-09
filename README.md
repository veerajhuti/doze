# Baymax AI
<img width="1315" alt="image" src="https://github.com/user-attachments/assets/96882264-4649-4516-916e-a912a2f8a1af" />

## Description

**Baymax AI** is a personal project inspired by the character Baymax from *Big Hero 6*. This app detects prolonged eye closure using real-time webcam footage, a potential sign of sleep apnea or fatigue. When prolonged eye closure is detected, the application alerts the user. 

This app aims to improve personal health monitoring by providing a fun, interactive way to keep track of eye closure and alertness. It’s an opportunity to explore computer vision, machine learning, and AI principles, specifically applied in healthcare-related applications.

## Features

The application uses OpenCV to process webcam footage and detect eye closure in real-time using the Eye Aspect Ratio (EAR). The EAR compares the vertical and horizontal distances of eye landmarks to determine whether the user’s eyes are open or closed.

All webcam footage is processed locally on your device, and no personal data is collected or stored.

### Installation

1. Start by cloning the repository to your local machine:

    `git clone https://github.com/veerajhuti/BaymaxAI.git`
   
    `cd BaymaxAI`

3. Make sure **Python 3.7+** is installed. Then, create a virtual environment and install the necessary Python dependencies:

    `python -m venv venv`

    `source venv/bin/activate`  Or on Windows: `venv\Scripts\activate`

    `pip install -r server/requirements.txt`

4. Download `shape_predictor_68_face_landmarks.dat`. You will need the dlib face landmark model for detecting facial landmarks. Download it from here and extract it into the `server/` directory.

5. Once the dependencies are installed, you can start the Flask backend server:

    `cd server`
   
    `flask run`

The backend server will run on `http://127.0.0.1:5000`.

5. Ensure Node.js and npm or yarn are installed. Go to the frontend directory and install the necessary npm packages:

    `cd frontend`
   
    `npm/yarn install`

7. Once the dependencies are installed, start the React development server:

    `npm start/yarn run start`

The frontend application will be accessible at `http://localhost:3000`.

## Usage

1.  Start both the backend and frontend servers.
2.  Open the frontend in your browser at `http://localhost:3000`.
3.  Click **Start Tracking** to begin detecting eye closure.
4.  If prolonged eye closure is detected, a drowsiness alert will pop up.
5.  Click **Stop Tracking** to stop the detection.

## Roadmap

Future updates may include enhancements to:

1. Integrate health tips using Healthline API or similar resources to provide personalized health tips based on detected eye closure and fatigue.

2. Add physical and/or wearable assistive devices to get a more holistic view of the user’s sleep patterns, movement, and fatigue levels.

4. Incorporate real-time data scraping that pulls updated health information, research papers, or tips based on detected user behaviour.

## References

1. [Eye Aspect Ratio (EAR) for Blink Detection](https://www.mdpi.com/2079-9292/11/19/3183)

2. [Sleep Apnea Detection and Technology](https://www.sciencedirect.com/science/article/pii/S2667241322000039)
   
4. [Baymax Animation](https://cssanimation.rocks/baymax/)

5. [Baymax Audio Expressions](https://www.myinstants.com/en/search/?name=baymax)
   
## Author

[Veera Jhuti](https://github.com/veerajhuti)
