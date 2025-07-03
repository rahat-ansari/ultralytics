"""
Smart Security System with YOLO11, Face Recognition, and Pose Detection
Author: Your Name
Version: 1.0
"""

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
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityConfig:
    """Configuration class for security system settings."""
    known_faces_dir: str = "images"
    alarm_sound_path: str = "./pols-aagyi-pols.mp3"
    alarm_cooldown: int = 3
    confidence_threshold: float = 0.5
    face_recognition_tolerance: float = 0.6
    pose_visibility_threshold: float = 0.5
    camera_index: int = 0
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""
    recipient_email: str = ""

class SecuritySystem:
    """Main security system class."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.known_face_encodings = []
        self.known_face_names = []
        self.last_alarm_time = 0
        self.detection_history = {}
        self.frame_count = 0
        self.sound_loaded = False
        
        # Initialize components
        self._initialize_models()
        self._load_known_faces()
        self._setup_audio()
        self._setup_camera()
        
    def _initialize_models(self):
        """Initialize YOLO and MediaPipe models."""
        try:
            self.model = YOLO("yolo11n.pt")
            logger.info("YOLO model loaded successfully")
            
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.config.confidence_threshold
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe pose model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _load_known_faces(self):
        """Load known faces from directory."""
        known_faces_path = Path(self.config.known_faces_dir)
        
        if not known_faces_path.exists():
            logger.warning(f"Known faces directory '{known_faces_path}' does not exist")
            return
        
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for image_path in known_faces_path.glob('*'):
            if image_path.suffix.lower() in supported_formats:
                try:
                    image = face_recognition.load_image_file(str(image_path))
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        name = image_path.stem
                        self.known_face_names.append(name)
                        logger.info(f"Loaded known face: {name}")
                    else:
                        logger.warning(f"No face found in {image_path}")
                        
                except Exception as e:
                    logger.error(f"Error loading face from {image_path}: {e}")
        
        logger.info(f"Loaded {len(self.known_face_encodings)} known faces")
    
    def _setup_audio(self):
        """Setup audio system for alarms."""
        try:
            pygame.mixer.init()
            if Path(self.config.alarm_sound_path).exists():
                pygame.mixer.music.load(self.config.alarm_sound_path)
                self.sound_loaded = True
                logger.info("Alarm sound loaded successfully")
            else:
                logger.warning(f"Alarm sound file not found: {self.config.alarm_sound_path}")
        except pygame.error as e:
            logger.warning(f"Could not load sound file: {e}")
    
    def _setup_camera(self):
        """Setup camera capture."""
        self.cap = cv2.VideoCapture(self.config.camera_index)
        if not self.cap.isOpened():
            logger.error("Could not open camera")
            raise RuntimeError("Camera initialization failed")
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Camera initialized successfully")
    
    def send_email_alert(self, alert_type: str = "THREAT", person_id: str = "Unknown") -> bool:
        """Send email alert with proper error handling."""
        if not self.config.email_enabled:
            logger.info("Email alerts disabled")
            return False
        
        if not all([self.config.sender_email, self.config.sender_password, self.config.recipient_email]):
            logger.error("Email configuration incomplete")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.sender_email
            msg['To'] = self.config.recipient_email
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
            
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.sendmail(self.config.sender_email, self.config.recipient_email, msg.as_string())
            
            logger.info(f"Email alert sent: {alert_type} - {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def play_alarm(self) -> bool:
        """Play alarm sound with cooldown."""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        if self.sound_loaded and (current_time - self.last_alarm_time) > self.config.alarm_cooldown:
            try:
                pygame.mixer.music.play()
                self.last_alarm_time = current_time
                logger.info("Alarm sound played")
                return True
            except pygame.error as e:
                logger.error(f"Failed to play alarm: {e}")
        
        return False
