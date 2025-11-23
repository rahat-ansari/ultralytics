# security_surveillance_app.py

import cv2
import argparse
from pathlib import Path
from extended_security_alarm import ExtendedSecurityAlarm


class SecuritySurveillanceApp:
    """
    Comprehensive Security Surveillance Application
    
    Supports multiple input sources:
    - Webcam
    - Video files
    - Image files
    - RTSP streams
    """
    
    def __init__(self, config):
        self.config = config
        self.security_alarm = None
        self.setup_security_alarm()
    
    def setup_security_alarm(self):
        """Initialize the extended security alarm system"""
        alarm_config = {
            "show": True,
            "model": self.config.model,
            "records": self.config.records,
            "family_members_dir": self.config.family_members_dir,
            "facial_recognition_threshold": self.config.face_threshold,
            "cattle_monitoring": self.config.cattle_monitoring,
            "alert_cooldown": self.config.alert_cooldown,
            "min_cattle_count": self.config.min_cattle_count,
        }
        
        self.security_alarm = ExtendedSecurityAlarm(**alarm_config)
        
        # Authenticate email if provided
        if all([self.config.from_email, self.config.email_password, self.config.to_email]):
            self.security_alarm.authenticate(
                self.config.from_email,
                self.config.email_password,
                self.config.to_email
            )
            print("Email authentication successful")
        else:
            print("Email credentials not provided - running without email alerts")
    
    def process_webcam(self, camera_id=0):
        """Process webcam feed for real-time surveillance"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam {camera_id}")
            return
        
        print("Starting webcam surveillance... Press 'q' to quit")
        
        while True:
            success, im0 = cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            # Process frame
            results = self.security_alarm(im0)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self, video_path):
        """Process video file with security surveillance"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer
        output_path = Path(video_path).stem + "_processed.mp4"
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )
        
        print(f"Processing video: {video_path}")
        frame_count = 0
        
        while True:
            success, im0 = cap.read()
            if not success:
                print("Video processing completed")
                break
            
            # Process frame
            results = self.security_alarm(im0)
            
            # Write processed frame
            video_writer.write(results.plot_im)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        video_writer.release()
        print(f"Output saved: {output_path}")
    
    def process_image(self, image_path):
        """Process single image for security analysis"""
        im0 = cv2.imread(str(image_path))
        if im0 is None:
            print(f"Error: Could not load image {image_path}")
            return
        
        # Process image
        results = self.security_alarm(im0)
        
        # Save result
        output_path = Path(image_path).stem + "_analyzed.jpg"
        cv2.imwrite(output_path, results.plot_im)
        print(f"Analysis complete. Output: {output_path}")
        
        # Print results
        if hasattr(results, 'unauthorized_count'):
            print(f"Unauthorized persons detected: {results.unauthorized_count}")
        if hasattr(results, 'cattle_count'):
            print(f"Cattle detected: {results.cattle_count}")
    
    def process_rtsp(self, rtsp_url):
        """Process RTSP stream"""
        print(f"Connecting to RTSP stream: {rtsp_url}")
        self.process_webcam(rtsp_url)  # OpenCV can handle RTSP URLs


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Extended Security Surveillance System')
    
    # Input source
    parser.add_argument('--source', type=str, required=True,
                       help='Input source: webcam, video/path, image/path, rtsp/url')
    
    # Model settings
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='YOLO model path (yolo11n.pt, yolo11s.pt, etc.)')
    parser.add_argument('--records', type=int, default=5,
                       help='Number of detections to trigger basic alert')
    
    # Face recognition
    parser.add_argument('--family-members-dir', type=str, default='data/family_members',
                       help='Directory containing authorized person images')
    parser.add_argument('--face-threshold', type=float, default=0.8,
                       help='Face recognition confidence threshold')
    
    # Cattle monitoring
    parser.add_argument('--cattle-monitoring', action='store_true', default=True,
                       help='Enable cattle monitoring')
    parser.add_argument('--min-cattle-count', type=int, default=1,
                       help='Minimum cattle count for theft detection')
    
    # Alert settings
    parser.add_argument('--alert-cooldown', type=int, default=300,
                       help='Cooldown between alerts in seconds')
    
    # Email settings
    parser.add_argument('--from-email', type=str,
                       help='Sender email address')
    parser.add_argument('--email-password', type=str,
                       help='Email app password')
    parser.add_argument('--to-email', type=str,
                       help='Receiver email address')
    
    args = parser.parse_args()
    
    # Create application
    app = SecuritySurveillanceApp(args)
    
    # Process based on source type
    source = args.source.lower()
    
    try:
        if source == 'webcam':
            app.process_webcam()
        elif source.startswith('video/'):
            video_path = source[6:]
            app.process_video(video_path)
        elif source.startswith('image/'):
            image_path = source[6:]
            app.process_image(image_path)
        elif source.startswith('rtsp/'):
            rtsp_url = source[5:]
            app.process_rtsp(rtsp_url)
        else:
            print("Invalid source type. Use: webcam, video/path, image/path, or rtsp/url")
    
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()