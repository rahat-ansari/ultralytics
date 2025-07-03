import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from ultralytics import YOLO
import smtplib
import pygame
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime

# Initialize YOLO model for person detection
model = YOLO("yolo11n.pt")

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load known faces from a directory (you need to create this)
def load_known_faces(known_faces_dir="known_faces"):
    """Load known faces from a directory containing images."""
    known_face_encodings = []
    known_face_names = []
    
    if os.path.exists(known_faces_dir):
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Load image
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    # Use filename (without extension) as name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"Loaded known face: {name}")
    
    return known_face_encodings, known_face_names

# Load known faces
known_face_encodings, known_face_names = load_known_faces()
print(f"Loaded {len(known_face_encodings)} known faces")

# Setup alarm sound
pygame.mixer.init()
try:
    pygame.mixer.music.load("./pols-aagyi-pols.mp3")
    sound_loaded = True
    print("Alarm sound loaded successfully")
except pygame.error:
    print("Warning: Could not load sound file. Continuing without audio alerts.")
    sound_loaded = False

# Start Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Global variables for alarm control
last_alarm_time = 0
alarm_cooldown = 3  # seconds between alarms
detection_history = {}  # Track detection history for each person

def send_email_alert(alert_type="THREAT", person_id="Unknown"):
    """Function to send email alert when threat is detected."""
    try:
        msg = MIMEMultipart()
        msg['From'] = "rahatansari.tpu@gmail.com"
        msg['To'] = "security_team@gmail.com"
        msg['Subject'] = f"Security Alert: {alert_type}"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
        Security Alert Details:
        - Alert Type: {alert_type}
        - Person: {person_id}
        - Time: {timestamp}
        - Location: Main Security Camera
        
        Please review the situation immediately.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("rahatansari.tpu@gmail.com", "nuzb vkot eauw qihl")
            server.sendmail("rahatansari.tpu@gmail.com", "security_team@gmail.com", msg.as_string())
        print(f"Email alert sent: {alert_type} - {person_id}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def play_alarm():
    """Play alarm sound with cooldown."""
    global last_alarm_time
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    if sound_loaded and (current_time - last_alarm_time) > alarm_cooldown:
        pygame.mixer.music.play()
        last_alarm_time = current_time
        return True
    return False

def detect_suspicious_activity(keypoints):
    """Detect various suspicious activities based on pose landmarks."""
    suspicious_activities = []
    
    try:
        # Get key landmarks
        left_wrist = keypoints[mp_pose.PoseLandmark.LEFT_WRIST]
        left_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_wrist = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW]
        nose = keypoints[mp_pose.PoseLandmark.NOSE]
        
        # Check visibility threshold
        visibility_threshold = 0.5
        
        # 1. Hands raised (surrender position or threatening gesture)
        if (left_wrist.visibility > visibility_threshold and left_shoulder.visibility > visibility_threshold):
            if left_wrist.y < left_shoulder.y - 0.1:  # Left hand significantly above shoulder
                suspicious_activities.append("Left hand raised")
        
        if (right_wrist.visibility > visibility_threshold and right_shoulder.visibility > visibility_threshold):
            if right_wrist.y < right_shoulder.y - 0.1:  # Right hand significantly above shoulder
                suspicious_activities.append("Right hand raised")
        
        # 2. Both hands raised (highly suspicious)
        if len([act for act in suspicious_activities if "hand raised" in act]) >= 2:
            suspicious_activities.append("Both hands raised - HIGH ALERT")
        
        # 3. Hands near face (covering face, suspicious behavior)
        if (left_wrist.visibility > visibility_threshold and nose.visibility > visibility_threshold):
            if abs(left_wrist.x - nose.x) < 0.15 and abs(left_wrist.y - nose.y) < 0.15:
                suspicious_activities.append("Hand near face")
        
        if (right_wrist.visibility > visibility_threshold and nose.visibility > visibility_threshold):
            if abs(right_wrist.x - nose.x) < 0.15 and abs(right_wrist.y - nose.y) < 0.15:
                suspicious_activities.append("Hand near face")
        
        # 4. Aggressive posture (arms spread wide)
        if (left_elbow.visibility > visibility_threshold and right_elbow.visibility > visibility_threshold and
            left_shoulder.visibility > visibility_threshold and right_shoulder.visibility > visibility_threshold):
            
            shoulder_width = abs(left_shoulder.x - right_shoulder.x)
            elbow_span = abs(left_elbow.x - right_elbow.x)
            
            if elbow_span > shoulder_width * 1.5:  # Elbows spread wider than shoulders
                suspicious_activities.append("Aggressive posture")
    
    except (IndexError, AttributeError) as e:
        print(f"Pose analysis error: {e}")
    
    return suspicious_activities

def identify_person(face_encoding):
    """Identify if a person is known or unknown."""
    if not known_face_encodings:
        return "Unknown", False
    
    # Compare with known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
            return known_face_names[best_match_index], True
    
    return "Unknown", False

print("🔒 Security System Started")
print("📋 System Rules:")
print("   ✅ Known faces = Safe (no alerts)")
print("   ⚠️  Unknown persons + suspicious activity = ALARM")
print("   🔍 Monitoring for: raised hands, face covering, aggressive postures")
print("   ⌨️  Press 'q' to quit")
print("-" * 50)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    # YOLO Detection for person detection
    results = model(frame, classes=[0], verbose=False)  # Only detect persons (class 0)
    
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            if conf < 0.5:  # Skip low confidence detections
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            person_id = f"person_{i}"
            is_known_person = False
            person_name = "Unknown"
            
            # Face Recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Look for faces in the person's bounding box
            face_locations = face_recognition.face_locations(rgb_frame[y1:y2, x1:x2])
            
            if face_locations:
                # Adjust face locations to full frame coordinates
                adjusted_face_locations = []
                for (top, right, bottom, left) in face_locations:
                    adjusted_face_locations.append((top + y1, right + x1, bottom + y1, left + x1))
                
                face_encodings = face_recognition.face_encodings(rgb_frame, adjusted_face_locations)
                
                if face_encodings:
                    person_name, is_known_person = identify_person(face_encodings[0])
                    person_id = person_name if is_known_person else f"Unknown_{i}"
            
            # Draw bounding box with different colors for known/unknown
            box_color = (0, 255, 0) if is_known_person else (0, 165, 255)  # Green for known, Orange for unknown
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Pose Estimation for Suspicious Behavior
            person_roi = frame[y1:y2, x1:x2]
            suspicious_activities = []
            
            if person_roi.size > 0:
                results_pose = pose.process(rgb_frame)
                
                if results_pose.pose_landmarks:
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
                    
                    # Detect suspicious activities
                    suspicious_activities = detect_suspicious_activity(results_pose.pose_landmarks.landmark)
            
            # Decision Logic: ALARM only for unknown persons with suspicious activity
            should_alarm = False
            alert_message = ""
            
            if not is_known_person and suspicious_activities:
                should_alarm = True
                alert_message = f"UNKNOWN + SUSPICIOUS!"
                activity_text = ", ".join(suspicious_activities[:2])  # Show first 2 activities
                
                # Display alert
                cv2.putText(frame, alert_message, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, activity_text, (x1, y1 - 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Trigger alarm
                if play_alarm():
                    print(f"🚨 ALARM: Unknown person detected with suspicious activity!")
                    print(f"   Activities: {', '.join(suspicious_activities)}")
                    # Uncomment to send email alerts
                    # send_email_alert("UNKNOWN PERSON WITH SUSPICIOUS ACTIVITY", person_id)
            
            elif is_known_person:
                # Known person - display as safe
                status_text = f"✅ SAFE: {person_name}"
                cv2.putText(frame, status_text, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if suspicious_activities:
                    # Show activities but don't alarm
                    activity_text = "Activities: " + ", ".join(suspicious_activities[:1])
                    cv2.putText(frame, activity_text, (x1, y1 - 35), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            elif not is_known_person and not suspicious_activities:
                # Unknown person but no suspicious activity
                cv2.putText(frame, "Unknown Person", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(frame, "Monitoring...", (x1, y1 - 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    
    # Display system status
    status_text = f"Security System Active | Frame: {frame_count}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display legend
    cv2.putText(frame, "Green=Known Safe | Orange=Unknown | Red=THREAT", 
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow("🔒 Security System - Smart Threat Detection", frame)
