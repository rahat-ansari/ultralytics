"""Security system configuration dataclass."""

from dataclasses import dataclass

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
