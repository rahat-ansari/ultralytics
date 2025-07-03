"""Configuration management for security system."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any

# Import SecurityConfig from the separate config module
from security_config import SecurityConfig

# Setup logger
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manage configuration settings."""
    
    def __init__(self, config_file: str = "security_config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            "known_faces_dir": "images",
            "alarm_sound_path": "./pols-aagyi-pols.mp3",
            "alarm_cooldown": 3,
            "confidence_threshold": 0.5,
            "face_recognition_tolerance": 0.6,
            "pose_visibility_threshold": 0.5,
            "camera_index": 0,
            "email_enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "",
            "sender_password": "",
            "recipient_email": ""
        }
    
    def load_config(self) -> SecurityConfig:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                config_data = self.default_config
        else:
            config_data = self.default_config
            self.save_config(config_data)
        
        return SecurityConfig(**config_data)
    
    def save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
        # return SecurityConfig(**self.default_config)    