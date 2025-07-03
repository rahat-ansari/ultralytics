"""Environment variable loader."""

import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables."""
    load_dotenv()
    
    return {
        'sender_email': os.getenv('SENDER_EMAIL', ''),
        'sender_password': os.getenv('SENDER_PASSWORD', ''),
        'recipient_email': os.getenv('RECIPIENT_EMAIL', ''),
        'camera_index': int(os.getenv('CAMERA_INDEX', '0')),
        'known_faces_dir': os.getenv('KNOWN_FACES_DIR', './images'),
        'alarm_sound_path': os.getenv('ALARM_SOUND_PATH', './pols-aagyi-pols.mp3')
    }