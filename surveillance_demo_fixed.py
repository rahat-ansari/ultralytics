# =========================================
# Computer Vision Surveillance Demo
# =========================================
# Compatible with Jupyter Notebook / Google Colab
# Requirements: ultralytics, streamlit, opencv-python, deepface, smtplib

import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
from deepface import DeepFace
import os

# Load YOLO model (pre-trained COCO dataset)
yolo_model = YOLO("yolo11l.pt")

# Define authorized persons directory
AUTHORIZED_DIR = "family_members"  # or authorized_persons

# Create the directory if it doesn't exist to avoid errors
os.makedirs(AUTHORIZED_DIR, exist_ok=True)

# Function to check if face belongs to authorized person
def is_authorized_face(frame_crop):
    """
    Checks if a face in the cropped frame is authorized by comparing against a database.
    """
    if frame_crop.size == 0:
        return False
    try:
        # The `DeepFace.find` is deprecated. The new approach is to find a face and verify it.
        # We can iterate through known faces and verify.
        for person_image in os.listdir(AUTHORIZED_DIR):
            if person_image.lower().endswith(('.png', '.jpg', '.jpeg')):
                known_face_path = os.path.join(AUTHORIZED_DIR, person_image)
                try:
                    # The verify function handles detection and comparison.
                    result = DeepFace.verify(img1_path=frame_crop,
                                             img2_path=known_face_path,
                                             enforce_detection=False)
                    if result.get("verified", False):
                        return True  # Found a match
                except Exception:
                    # This can happen if no face is found in the crop, which is expected.
                    continue
    except Exception as e:
        print(f"Face recognition error: {e}")
    return False


# Distance estimation (very simplified demo)
def estimate_distance(bbox, frame_width):
    x1, y1, x2, y2 = bbox
    object_width_px = x2 - x1
    focal_length = 500  # placeholder, needs calibration
    real_width_cm = 50  # average cattle/person shoulder width
    distance = (real_width_cm * focal_length) / object_width_px
    return distance


# Send email alert
def send_alert_email(message):
    sender = "your_email@gmail.com"
    password = "your_app_password"  # For Gmail, generate app password
    recipient = "recipient_email@gmail.com"

    msg = MIMEText(message)
    msg["Subject"] = "🚨 Security Alert - Unauthorized Activity Detected"
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
            print("✅ Alert email sent.")
    except Exception as e:
        print("❌ Email failed:", e)


# Threat detection algorithm
def detect_threat(frame):
    results = yolo_model(frame)  # 0=person, 1=bicycle (as proxy for cattle)
    for r in results:
        for box in r.boxes.xyxy:  # bounding boxes
            x1, y1, x2, y2 = map(int, box)
            distance = estimate_distance((x1, y1, x2, y2), frame.shape[1])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Check if authorized
            crop = frame[y1:y2, x1:x2]
            if not is_authorized_face(crop):
                cv2.putText(frame, f"⚠ Unauthorized {distance:.1f}cm", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                if distance < 200:  # threshold: 2 meters
                    # send_alert_email("Unauthorized person detected near cattle!")
                    print("Unauthorized person detected near cattle!")

    return frame


# Main video surveillance loop (for webcam or video file)
def run_surveillance(video_source):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = detect_threat(frame)

        cv2.imshow("Surveillance Feed", processed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the demo
def new_func(run_surveillance):
    run_surveillance("./media_files/animal_surveillance/goru-churi.mp4")

if __name__ == "__main__":
    # run_surveillance(0)  # webc am
    # run_surveillance("./media_files/ice skatting object traking/computer_vision_object_and_detection_tracking_ice_skatting_object_traking_video_20250819_173636_11.mp4")  # video file
    new_func(run_surveillance)  # video file
