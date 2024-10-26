

# Smart Attendance Using Face Recognition

This project is a smart attendance system that uses real-time face recognition to identify and log attendance. The system captures facial data using a webcam, detects faces, recognizes them using a pre-trained model, and logs the information in a CSV file.

## Features
    Real-Time Face Recognition: Utilizes a webcam to capture and process images.
  High-Accuracy Embedding Model: Uses InceptionResNet for embedding generation and a KNN classifier for identifying known faces.
  Face Detection: Employs a YOLO (You Only Look Once) model for fast and accurate face detection.
  CSV Logging: Records attendance details with timestamps in a CSV file for easy access.

## Technologies Used
  TensorFlow: For the InceptionResNet-based face embedding model.
  OpenCV: For handling webcam inputs and image processing.
  YOLO: For face detection during real-time capture.
  K-Nearest Neighbors (KNN)**: For face classification.
  Python: The primary language for implementing the system.

## Installation

1. Clone this Repository:
 
   git clone  https://github.com/SoundharBalaji/Attendance-System-using-Face-Recognition
 

2. Install Requirements:
   Install all dependencies by running:
  
   pip install -r requirements.txt
  
   
3. Download Required Files:
   - Pre-trained models for face embeddings and YOLO face detection 

4. Organize Dataset:
   - Structure your dataset.

## Usage

1. Train Face Embeddings:
   - Run the `model.ipynb` script to train the InceptionResNet model on your structured dataset:
    
     python model.ipynb.py

   
2. Run Face Recognition for Attendance:
   - Execute the main attendance script to start the webcam and log recognized faces:
 
     python attendance.py


