import os
import cv2
import numpy as np
import face_recognition
from ultralytics import solutions
from ultralytics.utils import LOGGER
import pygame
# Corrected imports for SolutionAnnotator and SolutionResults
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors

# --- 1. Pygame and Known Faces Setup (Global) ---

# Initialize alarm sound
pygame.mixer.init()
alarm_file = "pols-aagyi-pols.mp3"
if os.path.exists(alarm_file):
    pygame.mixer.music.load(alarm_file)
else:
    print(f"⚠️ Warning: Alarm file '{alarm_file}' not found — please check the path.")

# Define the directory for known faces
KNOWN_FACES_DIR = "family_members" 

# --- 2. Extended Class Definitions ---

class FaceRecognitionAlarm(solutions.SecurityAlarm):
    def __init__(self, face_data_path, **kwargs):
        super().__init__(**kwargs)
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_data_path = face_data_path
        self.sound_played = False
        self._load_known_faces()
        self.show = True

    def _load_known_faces(self):
        """Loads face encodings from the specified directory."""
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.exists(self.face_data_path):
             LOGGER.warning(f"Face data path '{self.face_data_path}' not found.")
             return

        for root, dirs, files in os.walk(self.face_data_path):
            for image_name in files:
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, image_name)
                    person_name = os.path.basename(root)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            self.known_face_encodings.append(face_encodings[0])
                            self.known_face_names.append(person_name)
                            LOGGER.info(f"Loaded face: {person_name} from {image_name}")
                        else:
                            LOGGER.warning(f"No face found in image: {image_path}")
                    except Exception as e:
                        LOGGER.error(f"Error loading {image_path}: {e}")

        if self.known_face_encodings:
            LOGGER.info(f"Loaded {len(self.known_face_encodings)} faces for {len(set(self.known_face_names))} people")
        else:
            LOGGER.warning("No known faces loaded. All persons will be unknown.")

    def process(self, im0):
        """
        Overrides process to check for UNKNOWN persons and trigger alarm.
        self.extract_tracks() is called by the parent's __call__ method.
        """
        # First, let the parent class extract tracks
        self.extract_tracks(im0)
        
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        unknown_person_count = 0
        person_cls_id = 0

        small_frame = cv2.resize(im0, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for i, box in enumerate(self.boxes):
            cls = self.clss[i]
            label = self.names[cls]
            color = colors(cls, True) # FIX: Use imported 'colors'

            if cls == person_cls_id:
                x1, y1, x2, y2 = map(int, box)
                found_face = False
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    top, right, bottom, left = top*4, right*4, bottom*4, left*4
                    face_center_x, face_center_y = (left + right) // 2, (top + bottom) // 2

                    if x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2:
                        found_face = True
                        name = "Unknown"
                        is_match = False
                        if self.known_face_encodings:
                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.55)
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            if len(face_distances) > 0:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    name = self.known_face_names[best_match_index]
                                    is_match = True
                        
                        if is_match:
                            label = f"{name} (Known)"
                            color = (0, 255, 0) # Green
                        else:
                            label = f"Unknown (ALARM!)"
                            color = (0, 0, 255) # Red
                            unknown_person_count += 1
                        break
                
                if not found_face:
                    label = f"{self.names[cls]} (No Face)"

            annotator.box_label(box, label=label, color=color)

        # Alarm Trigger Logic
        total_det = unknown_person_count
        
        if total_det >= self.records:
            if not self.email_sent:
                # self.send_email(im0, total_det)
                self.email_sent = True
            if not self.sound_played:
                if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                    LOGGER.info("🚨 Playing security alarm!")
                    pygame.mixer.music.play()
                    self.sound_played = True
        
        # Only reset the alarm if the condition is no longer met AND the sound has finished playing.
        # A simpler approach is to use a cooldown before resetting.
        if total_det < self.records:
            if self.sound_played and pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                self.email_sent = False
                self.sound_played = False
                LOGGER.info("🟢 Alarm system reset as threat is gone and sound finished.")

        
        plot_im = annotator.result()
        # if self.show:
        #     self.display_output(plot_im) 

        return SolutionResults(
            plot_im=plot_im, 
            im0=im0,
            total_tracks=len(self.track_ids),
            email_sent=self.email_sent,
            sound_played=self.sound_played,
            unknown_persons=unknown_person_count
        )

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    cap = cv2.VideoCapture("media_files/animal_surveillance/goru-churi.mp4")
    assert cap.isOpened(), "❌ Error: Cannot read video file."

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter("security_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    securityalarm = FaceRecognitionAlarm(
        show=True, 
        model="yolo11m.pt", # FIX: Changed to a model that likely exists
        records=3,
        classes=[0, 19], # Only detect 'person'
        face_data_path=KNOWN_FACES_DIR
    )

    # Optional: Email setup
    # from_email = "deveansari@gmail.com"
    # password = "ddgl yjef dlaw tuzg" 
    # to_email = "rahatansari.tpu@gmail.com"
    # securityalarm.authenticate(from_email, password, to_email)

    print("\n--- Starting Video Processing. Press 'q' to terminate. ---")
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("✅ Video processing completed.")
            break

        results = securityalarm(im0)
        video_writer.write(results.plot_im)
        # cv2.imshow("Security Alarm", results.plot_im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    if pygame.mixer.get_init():
        pygame.mixer.quit()
