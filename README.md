# Advanced Driver Drowsiness Detection (ADDD)

## Overview

Advanced Driver Drowsiness Detection (ADDD) is a computer vision–based safety system designed to detect driver fatigue in real time. The system uses a webcam to monitor the driver's face and eye movements using OpenCV and Dlib.

The system detects facial landmarks and calculates the Eye Aspect Ratio (EAR) to determine whether the driver's eyes are closed for a prolonged period. If drowsiness is detected, the system displays warning messages such as **"STAY ALERT"** and **"DON'T SLEEP"**, and triggers a buzzer/alarm sound to alert the driver.

This system helps prevent accidents caused by driver fatigue.

---

## Demo of Advanced Driver Drowsiness Detection

<p align="center">
  <img src="images/demo.png" width="900">
</p>

---

## Facial Landmark Detection

<p align="center">
  <img src="images/demo1.png" width="900">
</p>

---

## Features

* Real-time face detection
* Facial landmark detection using Dlib
* Eye Aspect Ratio (EAR) calculation
* Drowsiness detection based on eye closure
* On-screen warning messages
* Buzzer/alarm alert system
* Real-time webcam monitoring
* FPS display for performance tracking

---

## Technologies Used

* Python
* OpenCV
* Dlib
* Imutils
* NumPy
* SciPy

---

## Project Structure

```bash id="v2ncbt"
ADDD project/
│
├── detect.py
├── blinkFatigue.csv
├── shape_predictor_68_face_landmarks.dat
├── images/
│   ├── demo.png
│   └── demo1.png
└── README.md
```

---

## Installation

### Create Environment

```bash id="slp0qr"
conda create -n ADDD python=3.10
conda activate ADDD
```

### Install Dependencies

```bash id="oiv7t6"
pip install opencv-python imutils numpy scipy
conda install -c conda-forge dlib
```

---

## Running the Project

Run the following command:

```bash id="y9o6re"
python detect.py
```

The webcam will start and monitor the driver's eye movements. If the system detects that the driver's eyes remain closed for several frames, a warning message and alarm sound will be triggered.

---

## How the System Works

1. The webcam captures real-time video frames.
2. OpenCV detects the driver's face.
3. Dlib detects 68 facial landmark points.
4. Eye landmarks are used to calculate the Eye Aspect Ratio (EAR).
5. If the EAR remains below a threshold for multiple frames, drowsiness is detected.
6. Warning messages appear and a buzzer alert is activated.

---

## Applications

* Driver monitoring systems
* Smart vehicle safety systems
* Accident prevention technology
* Transportation safety monitoring

---

## Future Improvements

* Integration with IoT vehicle systems
* Mobile alert notifications
* AI-based fatigue prediction
* Dashboard camera integration

---


