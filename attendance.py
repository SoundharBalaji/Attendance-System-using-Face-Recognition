import os
import numpy as np
import cv2
import joblib
import torch
import csv
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
attendance=set()
# Initialize the device and face embedding model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_embedding = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load the pre-trained KNN model
knn_model = joblib.load('allknn.pkl')

# Load the YOLO model for face detection
yolo_model_path = r"E:\attendance\yolov8n-face.pt"
model = YOLO(yolo_model_path)

# Threshold for unknown faces (this needs to be tuned based on the embeddings)
unknown_threshold = 0.8  # Adjust this value based on experiment

# Voting threshold: The number of frames where a person needs to be recognized to mark present
voting_threshold = 5

# Dictionary to store the count of each person's recognition over frames
attendance_counter = defaultdict(int)

# List to hold the final attendance with name, time, and date
attendance_list = []

# Function to extract embeddings from face
def extract_embeddings(face):
    """Extract face embeddings from a given face image."""
    face_image = Image.fromarray(face).convert("RGB")
    face_image = face_image.resize((160, 160))

    face_tensor = torch.tensor(np.array(face_image)).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0  # Normalize
    face_tensor = face_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        embeddings = model_embedding(face_tensor)
    
    return embeddings.detach().cpu().numpy()

# Function to predict whether the person is known or unknown
def predict_person(embeddings, knn_model, threshold):
    """Predict whether the person is known or unknown."""
    embeddings_reshaped = embeddings.flatten().reshape(1, -1)
    distances, indices = knn_model.kneighbors(embeddings_reshaped)
    
    # Check the distance of the closest neighbor
    if distances[0][0] > threshold:
        return "Unknown"
    else:
        return knn_model.predict(embeddings_reshaped)[0]

# Function to export the attendance to a CSV file
def export_attendance(attendance_list, filename="attendance.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time", "Status"])
        for entry in attendance_list:
            writer.writerow(entry)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Perform face detection using YOLO
    results = model(frame, task='detect', conf=0.5, imgsz=1280)

    if results and len(results) > 0:
        for r in results:
            for box in r.boxes.xyxy:  # For each detected face
                x1, y1, x2, y2 = [int(coord) for coord in box[:4].cpu().numpy()]
                
                if (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1):
                    # Extract the face from the frame
                    face = frame[y1:y2, x1:x2]
                    
                    if face.size > 0:
                        embeddings = extract_embeddings(face)

                        if embeddings is not None and embeddings.size > 0:
                            name = predict_person(embeddings, knn_model, unknown_threshold)

                            # Increment the count for the recognized person (excluding "Unknown")
                            if name != "Unknown":
                                attendance_counter[name] += 1

                                # If person recognized for at least `voting_threshold` times, mark them present
                                if attendance_counter[name] == voting_threshold:
                                    current_time = datetime.now()
                                    date_str = current_time.strftime("%Y-%m-%d")
                                    time_str = current_time.strftime("%H:%M:%S")
                                    attendance_list.append([name, date_str, time_str, "Present"])

                                # Draw bounding box and predicted name
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                            else:
                                # Mark the face as "Unknown" on the frame without saving
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the webcam feed
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the display window
cap.release()
cv2.destroyAllWindows()

# Export attendance to a CSV file after the webcam feed ends
export_attendance(attendance_list)
print("Attendance exported to 'attendance.csv'.")
