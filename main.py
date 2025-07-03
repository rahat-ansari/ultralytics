"""Main application entry point."""

import cv2
import signal
import sys
from contextlib import contextmanager

from security_system import SecuritySystem, SecurityConfig
from config import ConfigManager
from env_loader import load_environment

class SecurityApp:
    """Main security application."""
    
    def __init__(self):
        self.system = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    @contextmanager
    def _managed_resources(self):
        """Context manager for resource cleanup."""
        try:
            yield
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        if self.system and hasattr(self.system, 'cap'):
            self.system.cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        logger.info("Resources cleaned up")
    
    def run(self):
        """Main application loop."""
        try:
            # Load configuration
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Load environment variables
            env_vars = load_environment()
            
            # Update config with environment variables
            for key, value in env_vars.items():
                if hasattr(config, key) and value:
                    setattr(config, key, value)
            
            # Initialize security system
            self.system = SecuritySystem(config)
            
            with self._managed_resources():
                self._main_loop()
                
        except Exception as e:
            logger.error(f"Application error: {e}")
            sys.exit(1)
    
    def _main_loop(self):
        """Main processing loop."""
        self.running = True
        logger.info("🔒 Security System Started")
        logger.info("📋 System Rules:")
        logger.info("   ✅ Known faces = Safe (no alerts)")
        logger.info("   ⚠️  Unknown persons + suspicious activity = ALARM")
        logger.info("   🔍 Monitoring for: raised hands, face covering, aggressive postures")
        logger.info("   ⌨️  Press 'q' to quit")
        logger.info("-" * 50)
        
        while self.running:
            ret, frame = self.system.cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break
            
            try:
                processed_frame = self.system.process_frame(frame)
                cv2.imshow("🔒 Security System - Smart Threat Detection", processed_frame)
                
                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit command received")
                    break
                    
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue

if __name__ == "__main__":
    app = SecurityApp()
    app.run()