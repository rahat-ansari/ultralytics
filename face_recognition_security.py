<<<<<<< HEAD
# security_surveillance_refactored.py
import os
import time
import cv2
import numpy as np
import threading
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Callable
from enum import Enum
import json

# Third-party imports with better error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    YOLO_AVAILABLE = False
    logging.warning(f"YOLO not available: {e}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logging.warning(f"MediaPipe not available: {e}")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning(f"face_recognition not available: {e}")

try:
    import RPi.GPIO as GPIO
    HAS_RPI = True
except (ImportError, RuntimeError):
    HAS_RPI = False
    logging.warning("RPi.GPIO not available. GPIO functionality disabled.")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("Pygame not available. Alarm sound functionality disabled.")


@dataclass
class Config:
    """Configuration settings for the security system."""
    MODEL_PATH: str = "yolo11m.pt"
    KNOWN_FACES_DIR: str = "family_members"
    ALARM_FILE: str = "alarm.mp3"
    LOG_DIR: str = "security_logs"
    OUTPUT_DIR: str = "output_videos"
    VIDEO_SOURCE: Any = "./media_files/WIN_20251103_14_11_20_Pro.mp4"  # Can be camera index (0, 1) or video file path
    
    # Performance and intervals
    FACE_RECOGNITION_INTERVAL: int = 5  # frames
    ALERT_COOLDOWN: int = 10  # seconds
    SAVE_CLIP_SECONDS: int = 10
    CLIP_FPS: int = 15

    # Confidence thresholds
    YOLO_CONFIDENCE: float = 0.5
    FACE_DETECTION_CONF: float = 0.5
    RECOGNITION_DISTANCE_THRESHOLD: float = 0.5

    # UI and Display
    WINDOW_NAME: str = "Security Monitoring"
    SECURE_ZONE_REL: Optional[Tuple[float, float, float, float]] = (0.0, 0.0, 1.0, 1.0) # Full frame

    OBJECTS_OF_INTEREST: List[str] = field(default_factory=lambda: [
        "person", "backpack", "suitcase", "knife"
    ])
    
    # Hardware flags
    USE_GPIO: bool = False
=======
#!/usr/bin/env python3
"""
Advanced Security Monitoring System with Face Recognition
Author: Security Team
Version: 2.2
Description: Real-time security monitoring using YOLO object detection,
             MediaPipe face detection, and face recognition.
"""

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from ultralytics import YOLO
import pygame
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import json

# Constants
DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_LOG_DIR = "security_logs"
FACE_RECOGNITION_TOLERANCE = 0.6
RESIZE_FACTOR = 0.25
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
>>>>>>> 302d2d26dbbece0806ee2341c3a90065f0fa928f


@dataclass
class DetectionResult:
<<<<<<< HEAD
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    person_id: Optional[str] = None
    in_secure_zone: bool = False

class SecuritySurveillance:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._setup_directories()
        self.logger = self._setup_logger()
        self._validate_dependencies()
        
        # Initialize attributes before use
        self.known_face_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []
        
        self._init_models()
        self._load_known_faces()
        self._setup_alarm()
        
        # State management
        self.frame_count = 0
        self._last_t = time.time()
        self.ring_buffer = deque(maxlen=int(self.cfg.SAVE_CLIP_SECONDS * self.cfg.CLIP_FPS))
        self.last_alert_time = 0.0
        self.person_alert_times: Dict[str, float] = {}
        self.detection_history: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"votes": deque(maxlen=10), "last_seen": 0})
        
        # Performance tracking
        self._fps_times = deque(maxlen=20)
        
        self._setup_gpio()

    def _setup_logger(self):
        """Sets up the logger for the system."""
        logger = logging.getLogger("SecuritySurveillance")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_file = os.path.join(self.cfg.LOG_DIR, f"security_{datetime.now().strftime('%Y%m%d')}.log")
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(sh)
        return logger

    def _init_models(self):
        """Initializes YOLO and face detection models."""
        self.logger.info("Loading models...")
        self.yolo = YOLO(self.cfg.MODEL_PATH)
        if MEDIAPIPE_AVAILABLE:
            mp_face_detection = mp.solutions.face_detection
            self.mp_detector = mp_face_detection.FaceDetection(min_detection_confidence=self.cfg.FACE_DETECTION_CONF)
        else:
            self.mp_detector = None
            self.logger.warning("MediaPipe detector not initialized.")
        self.logger.info("Models loaded.")

    def _load_known_faces(self):
        """Loads known faces from the specified directory."""
        if not FACE_RECOGNITION_AVAILABLE:
            self.logger.warning("Face recognition library not available. Skipping loading known faces.")
            return
        
        self.logger.info(f"Loading known faces from '{self.cfg.KNOWN_FACES_DIR}'...")
        if not os.path.isdir(self.cfg.KNOWN_FACES_DIR):
            self.logger.warning("Known faces directory not found.")
            return
            
        for name in os.listdir(self.cfg.KNOWN_FACES_DIR):
            person_dir = os.path.join(self.cfg.KNOWN_FACES_DIR, name)
            if not os.path.isdir(person_dir): continue
            for filename in os.listdir(person_dir):
                try:
                    image_path = os.path.join(person_dir, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_names.append(name)
                        self.logger.info(f"Loaded face for '{name}' from {filename}")
                        break # Load one image per person for simplicity
                except Exception as e:
                    self.logger.error(f"Failed to load face from {filename}: {e}")

    def _setup_alarm(self):
        """Initializes the alarm sound system."""
        if PYGAME_AVAILABLE and os.path.exists(self.cfg.ALARM_FILE):
            pygame.mixer.init()
            pygame.mixer.music.load(self.cfg.ALARM_FILE)
            self.logger.info("Alarm system ready.")
        else:
            self.logger.warning("Alarm sound file not found or pygame not installed. Alarm sound disabled.")

    def _setup_gpio(self):
        """Sets up GPIO pins if running on a Raspberry Pi."""
        if self.cfg.USE_GPIO and HAS_RPI:
            # Add your GPIO setup logic here
            self.logger.info("GPIO setup complete.")
        pass

    def _get_person_id(self, bbox: Tuple[int, int, int, int]) -> str:
        """Generates a simple ID based on the bounding box center."""
        # This is a placeholder for a more robust tracking algorithm
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        return f"person_{cx}_{cy}"

    def _is_in_secure_zone(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> bool:
        """Checks if the center of a bounding box is within the secure zone."""
        if not self.cfg.SECURE_ZONE_REL:
            return True
        h, w, _ = frame_shape
        rx1, ry1, rx2, ry2 = self.cfg.SECURE_ZONE_REL
        sx1, sy1, sx2, sy2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
        
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        
        return sx1 <= cx <= sx2 and sy1 <= cy <= sy2

    def _match_face(self, encoding: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Matches a face encoding against known faces."""
        if not self.known_face_encodings:
            return None, None
        
        distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        
        if distances[best_match_index] < self.cfg.RECOGNITION_DISTANCE_THRESHOLD:
            return self.known_names[best_match_index], distances[best_match_index]
        
        return None, None

    def _update_history(self, person_id: str, vote: str, distance: Optional[float]):
        """Updates the detection history for a person."""
        self.detection_history[person_id]["votes"].append(vote)
        self.detection_history[person_id]["last_seen"] = time.time()
        self.detection_history[person_id]["last_distance"] = distance

    def _confirm_recognition(self, person_id: str, name: str, distance: Optional[float]) -> bool:
        """Confirms recognition based on voting history."""
        # Placeholder for more complex confirmation logic
        history = self.detection_history.get(person_id, {})
        votes = history.get("votes", [])
        return votes and votes[-1] == name

    def _should_alert(self, person_id: str, is_unknown: bool) -> bool:
        """Determines if an alert should be triggered."""
        now = time.time()
        if (now - self.last_alert_time) < self.cfg.ALERT_COOLDOWN:
            return False
        
        last_person_alert = self.person_alert_times.get(person_id, 0)
        if (now - last_person_alert) < self.cfg.ALERT_COOLDOWN:
            return False
            
        return is_unknown

    def _handle_alert_actions(self, person: DetectionResult, frame: np.ndarray):
        """Handles actions to take when an alert is triggered."""
        self.logger.warning(f"ALERT: Unknown person detected at {person.bbox}!")
        now = time.time()
        self.last_alert_time = now
        if person.person_id:
            self.person_alert_times[person.person_id] = now
        
        # Play sound
        if PYGAME_AVAILABLE and pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
            
        # Save clip (simplified)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = os.path.join(self.cfg.OUTPUT_DIR, f"alert_{timestamp}.mp4")
        self.logger.info(f"Saving alert clip to {clip_path}")
        # In a real app, you would use a VideoWriter to save self.ring_buffer

    def _setup_directories(self):
        """Create necessary directories"""
        directories = [self.cfg.LOG_DIR, self.cfg.OUTPUT_DIR, self.cfg.KNOWN_FACES_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _validate_dependencies(self):
        """Validate that required dependencies are available"""
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLO is required but not available. Install with: pip install ultralytics")
        if not MEDIAPIPE_AVAILABLE:
            self.logger.warning("MediaPipe is not available. Face detection will be skipped.")
        if not FACE_RECOGNITION_AVAILABLE:
            self.logger.warning("face_recognition is not available. Face matching will be skipped.")

    def _process_yolo_detections(self, results) -> List[DetectionResult]:
        """Process YOLO results into standardized detection format"""
        detections = []
        for res in results:
            if res.boxes is None:
                continue
                
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = res.names[cls]
                
                detection = DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    class_name=name,
                    confidence=conf
                )
                
                if name == "person":
                    detection.person_id = self._get_person_id((x1, y1, x2, y2))
                    
                detections.append(detection)
                
        return detections

    def _categorize_detections(self, detections: List[DetectionResult]) -> Tuple[List[DetectionResult], List[str]]:
        """Categorize detections into persons and objects of interest"""
        persons = [d for d in detections if d.class_name == "person"]
        objects = [
            d.class_name for d in detections 
            if d.class_name in self.cfg.OBJECTS_OF_INTEREST and d.class_name != "person"
        ]
        return persons, objects

    def _process_person_detection(self, person: DetectionResult, frame: np.ndarray, 
                                annotated: np.ndarray, process_faces: bool) -> None:
        """Process a single person detection"""
        x1, y1, x2, y2 = person.bbox
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        # Draw basic person box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        
        # Check secure zone
        in_zone = self._is_in_secure_zone(person.bbox, frame.shape)
        person.in_secure_zone = in_zone
        
        if not in_zone:
            cv2.putText(annotated, "Outside Zone", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return

        # Face detection and recognition
        if process_faces and self.mp_detector and FACE_RECOGNITION_AVAILABLE:
            self._process_face_recognition(roi, person, frame, annotated, (x1, y1))

    def _process_face_recognition(self, roi: np.ndarray, person: DetectionResult, 
                                frame: np.ndarray, annotated: np.ndarray, offset: Tuple[int, int]) -> None:
        """Handle face detection and recognition for a person"""
        faces = self._detect_faces_mediapipe(roi)
        x1_offset, y1_offset = offset
        
        if not faces:
            return

        # Process the first detected face for simplicity
        fx, fy, fw, fh = faces[0]
        # Draw face box
        cv2.rectangle(annotated, 
                        (x1_offset + fx, y1_offset + fy),
                        (x1_offset + fx + fw, y1_offset + fy + fh),
                        (255, 0, 0), 1)
        
        # Extract face crop for recognition
        face_crop = self._extract_face_crop(roi, fx, fy, fw, fh)
        if face_crop is not None and person.person_id:
            self._perform_face_recognition(face_crop, person.person_id, person, annotated, offset)

    def _detect_faces_mediapipe(self, roi: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe"""
        faces = []
        try:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mp_detector.process(rgb)
            
            if res and res.detections:
                h, w = roi.shape[:2]
                for det in res.detections:
                    r = det.location_data.relative_bounding_box
                    fx = int(r.xmin * w)
                    fy = int(r.ymin * h)
                    fw = int(r.width * w)
                    fh = int(r.height * h)
                    faces.append((fx, fy, fw, fh))
                    
        except Exception as e:
            self.logger.error(f"MediaPipe face detection failed: {e}")
            
        return faces

    def _extract_face_crop(self, roi: np.ndarray, fx: int, fy: int, fw: int, fh: int) -> Optional[np.ndarray]:
        """Extract face crop from ROI with bounds checking"""
        fx0, fy0 = max(0, fx), max(0, fy)
        fx1, fy1 = min(roi.shape[1], fx + fw), min(roi.shape[0], fy + fh)
        
        if fx1 > fx0 and fy1 > fy0:
            return roi[fy0:fy1, fx0:fx1]
        return None

    def _perform_face_recognition(self, face_crop: np.ndarray, person_id: str, 
                                person: DetectionResult, annotated: np.ndarray, offset: Tuple[int, int]) -> None:
        """Perform face recognition on a face crop"""
        try:
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)
            
            if encodings:
                name, dist = self._match_face(encodings[0])
                vote_name = name if name is not None else "UNKNOWN"
                self._update_history(person_id, vote_name, dist)
                
                # Handle recognition confirmation and alerting
                self._handle_recognition_result(person_id, name, dist, person, annotated, offset)
                
        except Exception as e:
            self.logger.error(f"Face recognition failed: {e}")

    def _handle_recognition_result(self, person_id: str, name: Optional[str], dist: Optional[float],
                                 person: DetectionResult, annotated: np.ndarray, offset: Tuple[int, int]) -> None:
        """Handle the result of face recognition"""
        x1, y1, _, _ = person.bbox
        
        # Check if confirmed as known person
        is_confirmed_known = False
        recognized_name = None
        
        if self.known_names:  # Only check if we have known faces
            for known_name in self.known_names:
                if self._confirm_recognition(person_id, known_name, self.detection_history.get(person_id, {}).get("last_distance")):
                    is_confirmed_known = True
                    recognized_name = known_name
                    break

        # Check if confirmed as unknown
        is_confirmed_unknown = (not is_confirmed_known and 
                              self._confirm_recognition(person_id, "UNKNOWN", 
                                                      self.detection_history.get(person_id, {}).get("last_distance")))

        # Update display and handle alerts
        if is_confirmed_known:
            cv2.putText(annotated, f"{recognized_name} ✓", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif is_confirmed_unknown:
            if self._should_alert(person_id, is_unknown=True):
                self._handle_alert_actions(person, self.ring_buffer[-1] if self.ring_buffer else None)
            cv2.putText(annotated, "UNKNOWN - ALERT", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _calculate_fps(self) -> int:
        """Calculate current FPS using moving average"""
        now = time.time()
        frame_time = now - self._last_t
        self._last_t = now
        
        self._fps_times.append(frame_time)
        avg_frame_time = sum(self._fps_times) / len(self._fps_times)
        
        return int(1.0 / max(1e-6, avg_frame_time))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main frame processing pipeline"""
        self.frame_count += 1
        self.ring_buffer.append(frame.copy())
        
        annotated = frame.copy()
        
        try:
            # Object detection
            results = self.yolo(frame, conf=self.cfg.YOLO_CONFIDENCE, verbose=False)
            detections = self._process_yolo_detections(results)
            persons, objects = self._categorize_detections(detections)
            
            # Log objects of interest
            if objects:
                self.logger.debug("Objects of interest detected: %s", objects)
            
            # Process persons
            process_faces = (self.frame_count % self.cfg.FACE_RECOGNITION_INTERVAL) == 0
            for person in persons:
                self._process_person_detection(person, frame, annotated, process_faces)
                
        except Exception as e:
            self.logger.exception("Frame processing failed")
        
        # Draw UI elements
        self._draw_secure_zone(annotated, frame.shape)
        fps = self._calculate_fps()
        cv2.putText(annotated, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated

    def _draw_secure_zone(self, annotated: np.ndarray, frame_shape: Tuple[int, int, int]) -> None:
        """Draw secure zone rectangle if configured"""
        if not self.cfg.SECURE_ZONE_REL:
            return
            
        h, w, _ = frame_shape
        rx1, ry1, rx2, ry2 = self.cfg.SECURE_ZONE_REL
        sx1, sy1, sx2, sy2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
        
        cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1)
        cv2.putText(annotated, "Secure Zone", (sx1 + 5, sy1 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(self.cfg.VIDEO_SOURCE)
        if not cap.isOpened():
            self.logger.error("Cannot open video source: %s", self.cfg.VIDEO_SOURCE)
            return
            
        self.logger.info("Starting monitoring. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Frame read failed or video ended")
                    break
                    
                annotated = self.process_frame(frame)
                cv2.imshow(self.cfg.WINDOW_NAME, annotated)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self._cleanup(cap)

    def _cleanup(self, cap: cv2.VideoCapture) -> None:
        """Clean up resources"""
        self.logger.info("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        
        if self.cfg.USE_GPIO and HAS_RPI:
            GPIO.cleanup()
            
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
            
        self.logger.info("Shutdown complete")

if __name__ == "__main__":
    # Create a default configuration
    config = Config()
    
    # === USER: Adjust configuration here if needed ===
    # config.VIDEO_SOURCE = "media_files/animal_surveillance/goru-churi.mp4"
    # config.MODEL_PATH = "yolo11m.pt" 
    # ===============================================

    try:
        system = SecuritySurveillance(config)
        system.run()
    except Exception as e:
        logging.exception("An unhandled error occurred.")
=======
    """Data class for detection results."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str
    class_id: int


@dataclass
class FaceResult:
    """Data class for face recognition results."""
    bbox: Tuple[int, int, int, int]
    name: str
    confidence: float
    is_known: bool


class ConfigManager:
    """Configuration management class."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """Initialize ConfigManager."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        default_config = {
            "video_source": 0,  # 0 for webcam, path for video file
            "yolo_model": "yolo11m.pt",
            "known_faces_dir": "images",
            "alarm_file": "pols-aagyi-pols.mp3",
            "log_dir": DEFAULT_LOG_DIR,
            "face_recognition_interval": 5,
            "alert_cooldown": 10,
            "detection_confidence": 0.5,
            "iou_threshold": 0.5,
            "face_detection_confidence": 0.5,
            "objects_of_interest": [
                "person", "bicycle", "car", "motorcycle", "bus", "truck",
                "backpack", "umbrella", "handbag", "tie", "suitcase",
                "cell phone", "laptop", "book", "scissors", "knife"
            ],
            "display_settings": {
                "show_fps": True,
                "show_objects": True,
                "font_scale": 0.7,
                "thickness": 2
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logging.warning(f"Failed to load config: {e}. Using defaults.")
        else:
            self._save_config(default_config)

        return default_config

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except IOError as e:
            logging.error(f"Failed to save config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)


class Logger:
    """Enhanced logging system."""

    def __init__(self, log_dir: str = DEFAULT_LOG_DIR):
        """Initialize Logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.log_dir / f"security_{datetime.now().strftime('%Y-%m-%d')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def log_event(self, event_type: str, details: str = "") -> None:
        """Log security events."""
        self.logger.info(f"{event_type}: {details}")

    def log_error(self, error: str) -> None:
        """Log errors."""
        self.logger.error(error)

    def log_warning(self, warning: str) -> None:
        """Log warnings."""
        self.logger.warning(warning)


class FaceManager:
    """Face recognition management class."""

    def __init__(self, known_faces_dir: str):
        """Initialize FaceManager."""
        self.known_faces_dir = Path(known_faces_dir)
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.face_recognition_lock = threading.Lock()
        self._load_known_faces()

    def _load_known_faces(self) -> None:
        """Load known faces from directory."""
        if not self.known_faces_dir.exists():
            logging.warning(f"Known faces directory {self.known_faces_dir} does not exist")
            return

        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}

        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name

            for image_file in person_dir.iterdir():
                if image_file.suffix.lower() not in supported_formats:
                    continue

                try:
                    image = face_recognition.load_image_file(str(image_file))
                    face_encodings = face_recognition.face_encodings(image)

                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(person_name)
                        logging.info(f"Loaded face: {person_name} from {image_file.name}")
                    else:
                        logging.warning(f"No face found in {image_file}")

                except Exception as e:
                    logging.error(f"Error loading {image_file}: {e}")

        if self.known_face_encodings:
            logging.info(f"Successfully loaded {len(self.known_face_encodings)} face encodings "
                         f"for {len(set(self.known_face_names))} people")
        else:
            logging.warning("No face encodings loaded. Face recognition will not work.")

    def recognize_face(self, face_encoding: np.ndarray) -> Tuple[str, float]:
        """Recognize a face encoding."""
        if not self.known_face_encodings:
            return "Unknown", 0.0

        with self.face_recognition_lock:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE
            )

            if any(matches):
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    confidence = 1.0 - face_distances[best_match_index]
                    return self.known_face_names[best_match_index], confidence

        return "Unknown", 0.0


class MediaPipeFaceDetector:
    """MediaPipe face detection wrapper."""

    def __init__(self, min_detection_confidence: float = 0.5):
        """Initialize MediaPipeFaceDetector."""
        self.min_detection_confidence = min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection

    @contextmanager
    def get_detector(self):
        """Context manager for face detection."""
        detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.min_detection_confidence
        )
        try:
            yield detector
        finally:
            detector.close()

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame and return bounding boxes."""
        face_boxes = []

        with self.get_detector() as detector:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb_frame)

            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(w - x, int(bbox.width * w))
                    height = min(h - y, int(bbox.height * h))

                    if width > 0 and height > 0:
                        face_boxes.append((x, y, width, height))

        return face_boxes


class SecurityMonitor:
    """Main security monitoring class."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """Initialize SecurityMonitor."""
        self.config = ConfigManager(config_path)
        self.logger = Logger(self.config.get("log_dir"))
        self.face_manager = FaceManager(self.config.get("known_faces_dir"))
        self.face_detector = MediaPipeFaceDetector(
            self.config.get("face_detection_confidence")
        )

        # Initialize models
        self.yolo_model = self._load_yolo_model()

        # Initialize pygame for audio
        self._setup_audio()

        # Performance tracking
        self.frame_count = 0
        self.last_alert_time = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()

        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.executor = ThreadPoolExecutor(max_workers=2)

        self.logger.log_event("SYSTEM_INIT", "Security monitor initialized")

    def _load_yolo_model(self) -> YOLO:
        """Load YOLO model."""
        try:
            model_path = self.config.get("yolo_model")
            model = YOLO(model_path)
            self.logger.log_event("MODEL_LOADED", f"YOLO model loaded: {model_path}")
            return model
        except Exception as e:
            self.logger.log_error(f"Failed to load YOLO model: {e}")
            raise

    def _setup_audio(self) -> None:
        """Setup audio system."""
        try:
            pygame.mixer.init()
            alarm_file = self.config.get("alarm_file")

            if Path(alarm_file).exists():
                pygame.mixer.music.load(alarm_file)
                self.logger.log_event("AUDIO_INIT", f"Alarm loaded: {alarm_file}")
            else:
                self.logger.log_warning(f"Alarm file not found: {alarm_file}")
        except Exception as e:
            self.logger.log_error(f"Failed to initialize audio: {e}")

    def _detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect objects using YOLO."""
        try:
            results = self.yolo_model(
                frame,
                conf=self.config.get("detection_confidence"),
                iou=self.config.get("iou_threshold"),
                verbose=False
            )

            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()

                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = result.names[int(cls)]

                        detections.append(DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=float(conf),
                            class_name=class_name,
                            class_id=int(cls)
                        ))

            return detections

        except Exception as e:
            self.logger.log_error(f"Object detection failed: {e}")
            return []

    def _process_face_recognition(self, person_roi: np.ndarray) -> List[FaceResult]:
        """Process face recognition on person ROI."""
        if person_roi.size == 0:
            return []

        try:
            # Resize for faster processing
            small_frame = cv2.resize(person_roi, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Get face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)

            if not face_locations:
                return []

            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_results = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back to original size
                top = int(top / RESIZE_FACTOR)
                right = int(right / RESIZE_FACTOR)
                bottom = int(bottom / RESIZE_FACTOR)
                left = int(left / RESIZE_FACTOR)

                # Recognize face
                name, confidence = self.face_manager.recognize_face(face_encoding)

                face_results.append(FaceResult(
                    bbox=(left, top, right, bottom),
                    name=name,
                    confidence=confidence,
                    is_known=(name != "Unknown")
                ))

            return face_results

        except Exception as e:
            self.logger.log_error(f"Face recognition failed: {e}")
            return []

    def _draw_detections(self, frame: np.ndarray, detections: List[DetectionResult],
                         face_results: List[FaceResult]) -> Tuple[np.ndarray, List[str]]:
        """Draw all detections on frame."""
        result_frame = frame.copy()
        objects_of_interest = self.config.get("objects_of_interest")
        display_settings = self.config.get("display_settings")
        font_scale = display_settings["font_scale"]
        thickness = display_settings["thickness"]

        detected_objects = []

        # Draw object detections
        for detection in detections:
            if detection.class_name in objects_of_interest:
                x1, y1, x2, y2 = detection.bbox

                if detection.class_name != "person":
                    detected_objects.append(detection.class_name)
                    # Draw object bounding box in yellow
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), YELLOW, thickness) # Fix: Changed to YELLOW for non-person objects

                    # Draw object label
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    label_size = cv2
>>>>>>> 302d2d26dbbece0806ee2341c3a90065f0fa928f
