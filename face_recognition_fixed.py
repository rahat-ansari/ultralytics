import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from ultralytics import YOLO
# import smtplib
import pygame
import os
import time
from datetime import datetime

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_faces_mediapipe(frame):
    """
    Detect faces using MediaPipe
    Returns list of face bounding boxes in format [x, y, w, h]
    """
    face_boxes = []
    
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_boxes.append([x, y, width, height])
    
    return face_boxes

def draw_detections(frame, yolo_results, face_boxes):
    """
    Draw YOLO detections and face detections on frame
    """
    result_frame = frame.copy()
    
    # Draw YOLO detections
    if yolo_results and len(yolo_results) > 0:
        for result in yolo_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Get class name
                    class_name = result.names[int(cls)] if hasattr(result, 'names') else f"Class {int(cls)}"
                    
                    # Draw bounding box
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(result_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Draw face detections
    for face_box in face_boxes:
        x, y, w, h = face_box
        # Draw face bounding box in red
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Draw face label
        cv2.rectangle(result_frame, (x, y - 25), (x + 60, y), (0, 0, 255), -1)
        cv2.putText(result_frame, "Face", (x + 5, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_frame

# Initialize YOLO model for object detection
# Using YOLOv8n which can detect 80 different object classes
# model = YOLO("runs/detect/train2/weights/best.pt")
model = YOLO("yolo11m.pt")

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load known faces
known_face_encodings = []
known_face_names = []

# Setup for known faces - replace with your implementation
known_faces_dir = "images"  # Create this directory and add images
if os.path.exists(known_faces_dir):
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        known_face_names.append(person_name)
                        print(f"Loaded face: {person_name} from {image_name}")
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")

# Print summary of loaded faces
if known_face_encodings:
    print(f"Successfully loaded {len(known_face_encodings)} face encodings for {len(set(known_face_names))} people")
else:
    print("Warning: No face encodings loaded. Face recognition will not work.")

# Setup alarm sound
pygame.mixer.init()
alarm_file = "pols-aagyi-pols.mp3"
if os.path.exists(alarm_file):
    pygame.mixer.music.load(alarm_file)
else:
    print(f"Warning: Alarm file {alarm_file} not found")

# Create log directory
log_dir = "security_logs"
os.makedirs(log_dir, exist_ok=True)

def log_event(event_type, details=""):
    """Log security events to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(log_dir, f"security_log_{datetime.now().strftime('%Y-%m-%d')}.txt")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - {event_type}: {details}\n")

def send_email_alert(person_name="Unknown", objects_detected=None):
    """Function to send email alert when a person is detected."""
    if objects_detected is None:
        objects_detected = []
    
    objects_str = ", ".join(objects_detected) if objects_detected else "None"
    log_event("ALERT_TRIGGERED", f"Person: {person_name}, Objects: {objects_str}")
    print(f"Alert triggered: {person_name} detected with objects: {objects_str}")

# Start Video Capture
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./Test-Video-And-Images/istockphoto-2174886250-640_adpp_is.mp4")
# cap = cv2.VideoCapture("./Test-Video-And-Images/istockphoto-1456638008-640_adpp_is.mp4")
# cap = cv2.VideoCapture("https://www.youtube.com/watch?v=wswxrDiSiHI")
if not cap.isOpened():
    print("Error: Could not open video capture device")
    exit()

# Performance optimization variables
frame_count = 0
face_recognition_interval = 5  # Process face recognition every 5 frames
last_alert_time = 0
alert_cooldown = 10  # Seconds between alerts

# Define objects of interest (subset of COCO classes that YOLO can detect) - FIXED MISSING COMMA
objects_of_interest = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "mouse",  # Added missing comma here
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "cell phone", "laptop", "book", "scissors", "knife", "face"
]

print("Security monitoring started. Press 'q' to quit.")
log_event("SYSTEM_START")

try:
    while True:
        # Start a high-resolution timer to measure performance or track elapsed time using OpenCV's getTickCount() method
        timer = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        frame_count += 1
        # Determine whether to process faces based on frame count interval
        # Helps optimize performance by reducing face recognition computations
        process_faces = frame_count % face_recognition_interval == 0
        current_time = time.time()
        
        # YOLO Detection for all objects
        results = model.track(frame, device='cuda:0', persist=True)
        
        # Track detected objects in this frame
        detected_objects = []
        detected_persons = []

        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue  # Skip invalid boxes
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls]
                
                # Handle NON-PERSON objects - draw bounding boxes as usual
                if class_name in objects_of_interest and class_name != "person":
                    detected_objects.append(class_name)
                    
                    # Draw bounding box for object
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    # Display object name and confidence
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Handle PERSON detection with face recognition
                elif class_name == "person":
                    detected_persons.append((x1, y1, x2, y2))
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # Initialize variables for this person
                    person_name = "UNKNOWN"
                    face_detected = False
                    
                    # Only proceed if person ROI is valid
                    if person_roi.size > 0 and person_roi.shape[0] > 0 and person_roi.shape[1] > 0:
                        
                        # Face Recognition - only process every few frames for performance
                        if process_faces:
                            # Resize for faster processing
                            small_frame = cv2.resize(person_roi, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                            
                            # Detect face locations using face_recognition library
                            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
                            
                            if face_locations:
                                face_detected = True
                                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                                
                                # Process each detected face
                                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                                    top *= 4
                                    right *= 4
                                    bottom *= 4
                                    left *= 4
                                    
                                    # Adjust face coordinates to original frame coordinates
                                    face_x1 = x1 + left
                                    face_y1 = y1 + top
                                    face_x2 = x1 + right
                                    face_y2 = y1 + bottom
                                    
                                    # Ensure face coordinates are within person ROI bounds
                                    face_x1 = max(x1, min(face_x1, x2))
                                    face_y1 = max(y1, min(face_y1, y2))
                                    face_x2 = max(x1, min(face_x2, x2))
                                    face_y2 = max(y1, min(face_y2, y2))
                                    
                                    # Compare with known faces if available
                                    if known_face_encodings:
                                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                                        
                                        if any(matches):
                                            # Find the best match
                                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                            best_match_index = np.argmin(face_distances)
                                            if matches[best_match_index]:
                                                person_name = known_face_names[best_match_index]
                                                print(f"✅ Known Person Detected: {person_name}")
                                                
                                                # Alert with cooldown for known person
                                                if current_time - last_alert_time > alert_cooldown:
                                                    if not pygame.mixer.music.get_busy():
                                                        pygame.mixer.music.play()
                                                    log_event("KNOWN_PERSON", f"Detected: {person_name} with objects: {', '.join(detected_objects) if detected_objects else 'None'}")
                                                    last_alert_time = current_time
                                        else:
                                            person_name = "UNKNOWN"
                                            print("⚠️ Unknown Person Detected!")
                                            log_event("UNKNOWN_PERSON", f"With objects: {', '.join(detected_objects) if detected_objects else 'None'}")
                                    
                                    # Draw face bounding box
                                    face_color = (0, 255, 0) if person_name != "UNKNOWN" else (0, 0, 255)
                                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), face_color, 2)
                                    
                                    # Draw face label with name
                                    face_label = f"Face: {person_name}"
                                    label_size = cv2.getTextSize(face_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    cv2.rectangle(frame, (face_x1, face_y1 - label_size[1] - 10), 
                                                (face_x1 + label_size[0], face_y1), face_color, -1)
                                    cv2.putText(frame, face_label, (face_x1, face_y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Alternative: Use MediaPipe for face detection if face_recognition didn't find faces
                        if not face_detected:
                            face_boxes = detect_faces_mediapipe(person_roi)
                            if face_boxes:
                                face_detected = True
                                for face_box in face_boxes:
                                    fx, fy, fw, fh = face_box
                                    # Adjust face coordinates to original frame coordinates
                                    face_x1 = x1 + fx
                                    face_y1 = y1 + fy
                                    face_x2 = face_x1 + fw
                                    face_y2 = face_y1 + fh
                                    
                                    # Draw face bounding box in red for unknown (MediaPipe detection)
                                    cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)
                                    cv2.putText(frame, "Face: UNKNOWN", (face_x1, face_y1 - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                    
                    # Draw person bounding box only if no face was detected
                    # if not face_detected:
                    #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    #     cv2.putText(frame, "Person - No Face", (x1, y1 - 10), 
                    #               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        
        # Display detected objects summary (optional)
        if detected_objects:
            objects_text = f"Objects: {', '.join(set(detected_objects))}"
            cv2.putText(frame, objects_text, (20, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
        
        # Calculate and display FPS (optional)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Security Monitoring', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"An error occurred: {e}")
    log_event("SYSTEM_ERROR", str(e))
finally:
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()  # Close MediaPipe resources
    pygame.mixer.quit()
    log_event("SYSTEM_SHUTDOWN")
    print("Security monitoring stopped.")
