# extended_security_alarm.py

import cv2
import numpy as np
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

import pygame

from ultralytics import YOLO, solutions
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class ExtendedSecurityAlarm(solutions.SecurityAlarm):
    """
    Extended Security Alarm System with facial recognition, cattle monitoring, and enhanced features.
    """

    def __init__(self, **kwargs):
        """
        Initialize extended security alarm system.
        """
        # Extract extended parameters before calling parent constructor
        self.family_members_dir = Path(kwargs.pop("family_members_dir", "data/family_members"))
        self.facial_recognition_threshold = kwargs.pop("facial_recognition_threshold", 0.8)
        self.cattle_monitoring = kwargs.pop("cattle_monitoring", True)
        self.alert_cooldown = kwargs.pop("alert_cooldown", 300)
        self.min_cattle_count = kwargs.pop("min_cattle_count", 1)
        
        # Call parent constructor
        super().__init__(**kwargs)
        
        # Extended state
        self.known_faces = {}
        self.face_encodings = {}
        self.authorized_track_ids = set()
        self.cattle_count_history = []
        self.last_alert_time = {}
        self.unauthorized_detections = []
        
        # Initialize pygame for audio alerts
        try:
            pygame.mixer.init()
            print("✅ Pygame mixer initialized for audio alerts")
        except Exception as e:
            print(f"❌ Pygame mixer initialization failed: {e}")
        
        # Initialize extended features
        self._setup_extended_features()

    def _setup_extended_features(self):
        """Setup extended features like face recognition"""
        self.load_known_faces()
        self.setup_logging()

    def setup_logging(self):
        """Setup extended logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('extended_security.log'),
                logging.StreamHandler()
            ]
        )

    def load_known_faces(self):
        """Load known faces from family_members directory"""
        LOGGER.info("Loading known faces...")
        
        if not self.family_members_dir.exists():
            LOGGER.warning(f"Family members directory {self.family_members_dir} not found")
            # Create directory if it doesn't exist
            self.family_members_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"Created directory: {self.family_members_dir}")
            return
        
        face_count = 0
        for person_dir in self.family_members_dir.iterdir():
            if person_dir.is_dir():
                person_name = person_dir.name
                self.known_faces[person_name] = []
                self.face_encodings[person_name] = []
                
                for img_path in person_dir.glob("*.*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        try:
                            image = cv2.imread(str(img_path))
                            if image is not None:
                                encoding = self.extract_face_encoding(image)
                                if encoding is not None:
                                    self.known_faces[person_name].append(img_path)
                                    self.face_encodings[person_name].append(encoding)
                                    face_count += 1
                                    LOGGER.info(f"Loaded face for {person_name}: {img_path.name}")
                        except Exception as e:
                            LOGGER.error(f"Error loading {img_path}: {e}")
        
        LOGGER.info(f"Total faces loaded: {face_count}")

    def extract_face_encoding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face encoding from image."""
        try:
            # Convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encoding = self._simple_face_encoding(rgb_image)
            return encoding
        except Exception as e:
            LOGGER.error(f"Error extracting face encoding: {e}")
            return None

    def _simple_face_encoding(self, image: np.ndarray) -> np.ndarray:
        """Simple face encoding method."""
        # Resize to standard size
        resized = cv2.resize(image, (128, 128))
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        # Flatten and normalize
        encoding = equalized.flatten() / 255.0
        return encoding

    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Recognize face from detected face image."""
        try:
            encoding = self.extract_face_encoding(face_image)
            if encoding is None:
                return "Unknown", 0.0
            
            best_match = None
            best_similarity = 0.0
            
            for person_name, encodings in self.face_encodings.items():
                for known_encoding in encodings:
                    similarity = self._calculate_similarity(encoding, known_encoding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = person_name
            
            if best_similarity >= self.facial_recognition_threshold:
                return best_match, best_similarity
            else:
                return "Unknown", best_similarity
                
        except Exception as e:
            LOGGER.error(f"Error in face recognition: {e}")
            return "Unknown", 0.0

    def _calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate cosine similarity between two encodings."""
        dot_product = np.dot(encoding1, encoding2)
        norm1 = np.linalg.norm(encoding1)
        norm2 = np.linalg.norm(encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def should_send_alert(self, alert_type: str) -> bool:
        """Check if alert should be sent based on cooldown period."""
        current_time = datetime.now()
        
        if alert_type in self.last_alert_time:
            time_diff = (current_time - self.last_alert_time[alert_type]).total_seconds()
            if time_diff < self.alert_cooldown:
                return False
        
        self.last_alert_time[alert_type] = current_time
        return True

    def play_alert_sound(self):
        """Play alert sound using pygame."""
        try:
            if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                # A simple beep sound can be pre-loaded or generated once
                # For simplicity, let's assume a file "alarm.mp3" exists.
                # If you want to generate a beep, do it once in __init__
                if Path("alarm.mp3").exists():
                    pygame.mixer.music.load("alarm.mp3")
                    pygame.mixer.music.play()
                    LOGGER.info("Playing alert sound.")
                else:
                    LOGGER.warning("Alarm sound file 'alarm.mp3' not found.")
            
        except Exception as e:
            LOGGER.error(f"Error playing alert sound: {e}")

    def send_extended_email(self, im0: np.ndarray, alert_type: str, details: Dict):
        """Send extended email alert with detailed information."""
        if not self.server:
            LOGGER.warning("Email server not authenticated")
            return
        
        try:
            # Play alert sound
            self.play_alert_sound()
            
            # Encode image
            img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()
            
            # Create message
            message = MIMEMultipart()
            message["From"] = self.from_email
            message["To"] = self.to_email
            
            # Customize subject based on alert type
            if alert_type == "unauthorized_person":
                message["Subject"] = "SECURITY ALERT: Unauthorized Person Detected"
                body = self._create_unauthorized_person_body(details)
            elif alert_type == "cattle_theft":
                message["Subject"] = "CRITICAL ALERT: Possible Cattle Theft"
                body = self._create_cattle_theft_body(details)
            elif alert_type == "suspicious_activity":
                message["Subject"] = "ALERT: Suspicious Activity Detected"
                body = self._create_suspicious_activity_body(details)
            else:
                message["Subject"] = f"Security Alert: {alert_type}"
                body = f"Security alert triggered: {details}"
            
            message.attach(MIMEText(body, 'plain'))
            
            # Attach image
            image_attachment = MIMEImage(img_bytes, name="security_alert.jpg")
            message.attach(image_attachment)
            
            # Send email
            self.server.send_message(message)
            LOGGER.info(f"{alert_type} email sent successfully!")
            
        except Exception as e:
            LOGGER.error(f"Failed to send {alert_type} email: {e}")

    def _create_unauthorized_person_body(self, details: Dict) -> str:
        """Create email body for unauthorized person alert."""
        return f"""
SECURITY ALERT - Unauthorized Person Detected

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: Surveillance Area
Confidence: {details.get('confidence', 0):.2f}
Position: {details.get('position', 'Unknown')}
Track ID: {details.get('track_id', 'N/A')}

Immediate attention required!

This is an automated security alert from your surveillance system.
"""

    def _create_cattle_theft_body(self, details: Dict) -> str:
        """Create email body for cattle theft alert."""
        return f"""
CRITICAL ALERT - Possible Cattle Theft Detected

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: Cattle Shelter
Current Cattle Count: {details.get('current_count', 0)}
Expected Minimum: {details.get('min_count', 0)}
Unauthorized Persons Nearby: {details.get('unauthorized_nearby', 0)}

IMMEDIATE ACTION REQUIRED!

This is a critical alert from your cattle monitoring system.
"""

    def _create_suspicious_activity_body(self, details: Dict) -> str:
        """Create email body for suspicious activity alert."""
        return f"""
SUSPICIOUS ACTIVITY ALERT

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Activity Type: {details.get('activity_type', 'Unknown')}
Location: {details.get('location', 'Surveillance Area')}
Details: {details.get('details', 'No additional details')}

Investigation recommended.

This is an automated alert from your surveillance system.
"""

    def extract_face_region(self, im0: np.ndarray, box: List) -> Optional[np.ndarray]:
        """Extract face region from bounding box."""
        try:
            x1, y1, x2, y2 = map(int, box[:4])
            
            # Expand face region slightly
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(im0.shape[1], x2 + padding)
            y2 = min(im0.shape[0], y2 + padding)
            
            face_region = im0[y1:y2, x1:x2]
            
            if face_region.size > 0:
                return face_region
            return None
        except Exception as e:
            LOGGER.error(f"Error extracting face region: {e}")
            return None

    def analyze_cattle_situation(self, boxes: List, clss: List, track_ids: List) -> Dict:
        """Analyze cattle situation for theft detection."""
        # Use actual class names from the model
        cattle_classes = ['cow', 'sheep', 'horse', 'bird']  # Common animal classes in YOLO
        current_cattle_count = 0
        unauthorized_near_cattle = 0
        
        # Count cattle
        for box, cls, track_id in zip(boxes, clss, track_ids):
            class_name = self.names[int(cls)] if hasattr(self, 'names') else str(cls)
            if class_name.lower() in cattle_classes:
                current_cattle_count += 1
        
        # Check for unauthorized persons near cattle
        for i, (box, cls, track_id) in enumerate(zip(boxes, clss, track_ids)):
            class_name = self.names[int(cls)] if hasattr(self, 'names') else str(cls)
            if class_name == 'person' and track_id not in self.authorized_track_ids:
                # Check if person is near cattle (simplified)
                person_center = self._get_bbox_center(box)
                for j, (cattle_box, cattle_cls) in enumerate(zip(boxes, clss)):
                    cattle_class_name = self.names[int(cattle_cls)] if hasattr(self, 'names') else str(cattle_cls)
                    if cattle_class_name.lower() in cattle_classes:
                        cattle_center = self._get_bbox_center(cattle_box)
                        distance = self._calculate_distance(person_center, cattle_center)
                        if distance < 200:  # Threshold distance
                            unauthorized_near_cattle += 1
                            break
        
        return {
            'current_count': current_cattle_count,
            'unauthorized_nearby': unauthorized_near_cattle,
            'theft_suspected': (current_cattle_count < self.min_cattle_count) or 
                              (unauthorized_near_cattle > 0)
        }

    def _get_bbox_center(self, box: List) -> Tuple[float, float]:
        """Calculate bounding box center."""
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calculate_distance(self, point1: Tuple, point2: Tuple) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def process(self, im0):
        """
        Extended process method with facial recognition and cattle monitoring.
        """
        # Call parent process to extract tracks and basic detection
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        
        unauthorized_count = 0
        cattle_analysis = {'current_count': 0, 'unauthorized_nearby': 0, 'theft_suspected': False}
        
        # Process each detection
        if hasattr(self, 'boxes') and self.boxes is not None:
            for i, (box, cls, track_id) in enumerate(zip(self.boxes, self.clss, self.track_ids)):
                class_name = self.names[int(cls)]
                
                # Person detection with facial recognition
                if class_name == 'person':
                    face_region = self.extract_face_region(im0, box)
                    
                    if face_region is not None:
                        identity, confidence = self.recognize_face(face_region)
                        
                        if identity == "Unknown":
                            unauthorized_count += 1
                            color = (0, 0, 255)  # Red for unauthorized
                            label = f"Unauthorized {confidence:.2f}"
                            
                            # Check if we should send alert
                            if self.should_send_alert("unauthorized_person"):
                                alert_details = {
                                    'confidence': confidence,
                                    'position': f"{box[0]:.1f}, {box[1]:.1f}",
                                    'track_id': int(track_id) if track_id is not None else i
                                }
                                self.send_extended_email(im0, "unauthorized_person", alert_details)
                        else:
                            if track_id is not None:
                                self.authorized_track_ids.add(int(track_id))
                            color = (0, 255, 0)  # Green for authorized
                            label = f"{identity} {confidence:.2f}"
                    else:
                        color = (0, 165, 255)  # Orange for unknown
                        label = f"Person {int(track_id) if track_id is not None else i}"
                    
                    annotator.box_label(box, label=label, color=color)
                
                # Cattle monitoring
                elif class_name.lower() in ['cow', 'sheep', 'horse', 'bird'] and self.cattle_monitoring:
                    color = (255, 0, 0)  # Blue for cattle
                    label = f"{class_name} {int(track_id) if track_id is not None else i}"
                    annotator.box_label(box, label=label, color=color)
                
                else:
                    # Default annotation for other objects
                    annotator.box_label(box, label=self.names[int(cls)], color=colors(int(cls), True))
            
            # Analyze cattle situation
            if self.cattle_monitoring and len(self.boxes) > 0:
                cattle_analysis = self.analyze_cattle_situation(self.boxes, self.clss, self.track_ids)
                
                # Check for cattle theft
                if cattle_analysis['theft_suspected'] and self.should_send_alert("cattle_theft"):
                    self.send_extended_email(im0, "cattle_theft", cattle_analysis)
        
        # Original security alarm logic (total detections)
        total_det = len(self.clss) if hasattr(self, 'clss') and self.clss is not None else 0
        if total_det >= self.records and not self.email_sent:
            self.send_email(im0, total_det)
            self.email_sent = True
        
        # Add information text to frame using cv2 directly
        info_text = [
            f"Unauthorized: {unauthorized_count}",
            f"Cattle: {cattle_analysis['current_count']}",
            f"Total: {total_det}"
        ]
        
        plot_im = annotator.result()
        
        # Add text using OpenCV
        for i, text in enumerate(info_text):
            y_position = 30 + i * 25
            # Add background for better text visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(plot_im, (10, y_position - text_size[1] - 5), 
                         (10 + text_size[0] + 10, y_position + 5), (0, 0, 0), -1)
            # Add text
            cv2.putText(plot_im, text, (15, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        self.display_output(plot_im)
        
        # Return extended results
        return SolutionResults(
            plot_im=plot_im,
            im0=im0,
            total_tracks=len(self.track_ids) if hasattr(self, 'track_ids') else 0,
            email_sent=self.email_sent,
            unauthorized_count=unauthorized_count,
            cattle_count=cattle_analysis['current_count'],
            theft_suspected=cattle_analysis['theft_suspected']
        )


# ========== 🎥 MAIN LOOP ==========

if __name__ == "__main__":
    # Open video
    cap = cv2.VideoCapture("media_files/animal_surveillance/goru-churi.mp4")
    # cap = cv2.VideoCapture(0)  # Uncomment for webcam
    
    if not cap.isOpened():
        print("❌ Error: Cannot read video source.")
        exit(1)

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Video writer setup
    video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Security Alarm setup: USE THE EXTENDED CLASS
    security_alarm = ExtendedSecurityAlarm(
        show=True,  # Show annotated video
        model="yolo11n.pt",  # Use YOLOv11 nano model (faster for testing)
        records=3,  # Number of detections to trigger event
        # classes=[0, 16],  # 0=person, 16=dog (adjust based on your needs)
        family_members_dir="family_members",  # Directory for authorized faces
        cattle_monitoring=True,
        alert_cooldown=60  # 1 minute cooldown for testing
    )

    # Optional: Email setup (comment out if not using email)
    try:
        from_email = "deveansari@gmail.com"
        password = "ddgl yjef dlaw tuzg"  # App password
        to_email = "rahatansari.tpu@gmail.com"
        security_alarm.authenticate(from_email, password, to_email)
        print("✅ Email authentication successful")
    except Exception as e:
        print(f"❌ Email authentication failed: {e}")
        print("⚠️  Continuing without email alerts")

    print("🎥 Starting video processing... Press 'q' to quit")

    # Process video
    while cap.isOpened():
        success, im0 = cap.read()

        if not success:
            print("✅ Video processing completed.")
            break

        # Process frame with extended security alarm
        results = security_alarm(im0)

        # Write processed frame
        video_writer.write(results.plot_im)

        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⏹️ Processing stopped by user")
            break

    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Clean up pygame mixer
    try:
        pygame.mixer.quit()
    except:
        pass
        
    print("🎉 Processing completed successfully!")