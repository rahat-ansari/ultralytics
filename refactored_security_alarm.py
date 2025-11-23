import os
import cv2
from matplotlib import colors
import numpy as np
import face_recognition
import pygame
from ultralytics import solutions
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults

# ========== 🔊 SOUND SETUP ==========
pygame.mixer.init()
ALARM_FILE = "pols-aagyi-pols.mp3"
if os.path.exists(ALARM_FILE):
    pygame.mixer.music.load(ALARM_FILE)
else:
    print(f"[WARNING] Alarm file '{ALARM_FILE}' not found.")


# ========== 🧠 KNOWN FACE ENCODING LOADER ==========
KNOWN_FACE_DIR = "family_members"
known_face_encodings, known_face_names = [], []

if os.path.exists(KNOWN_FACE_DIR):
    for name in os.listdir(KNOWN_FACE_DIR):
        person_dir = os.path.join(KNOWN_FACE_DIR, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            try:
                img = face_recognition.load_image_file(path)
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_face_encodings.append(enc[0])
                    known_face_names.append(name)
                    print(f"[INFO] Loaded face for {name} from {filename}")
            except Exception as e:
                print(f"[ERROR] Failed loading {path}: {e}")
else:
    print("[WARNING] No known_faces directory found.")


# ========== 👁️ FACE-RECOGNITION ALARM (REVISED & OPTIMIZED) ==========
class FaceRecognitionAlarm(solutions.SecurityAlarm):
    def __init__(self, *args, known_face_encodings=None, known_face_names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.known_face_encodings = known_face_encodings or []
        self.known_face_names = known_face_names or []
        self.sound_played = False
        # Best practice: Set face recognition tolerance during initialization
        self.face_tolerance = 0.55

    def play_sound(self):
        """Plays the alarm sound if it's not already playing."""
        if not self.sound_played:
            if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
                self.sound_played = True
                LOGGER.info("🚨 Alarm Triggered: Unknown person count reached threshold.")

    def reset_sound(self):
        """Stops the alarm sound and resets the state."""
        if self.sound_played:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            self.sound_played = False
            LOGGER.info("🟢 Alarm Reset: Area clear.")

    def __call__(self, im0):
        """
        Processes a single frame for person detection and face recognition.
        This implementation follows best practices for accuracy and performance.
        """
        # 1. Get person detections from the base class
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)
        unknown_person_count = 0

        # 2. Optimize by finding all faces in the frame at once (on a smaller version)
        # This is much faster than processing crops for each person.
        h, w, _ = im0.shape
        small_frame = cv2.resize(im0, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # 3. Iterate through detected PERSONS from YOLO
        for box, conf, cls in zip(self.boxes, self.confs, self.clss):
            if int(cls) != 0:  # Skip if not a person
                continue

            name = "Unknown"
            is_known = False
            
            # 4. Associate faces with person boxes
            # Check if any detected face is inside this person's bounding box
            person_box_left, person_box_top, person_box_right, person_box_bottom = map(int, box)
            
            for (face_top, face_right, face_bottom, face_left), face_encoding in zip(face_locations, face_encodings):
                # Scale face locations back to original image size
                face_top *= 4
                face_right *= 4
                face_bottom *= 4
                face_left *= 4

                # Check if the center of the face is inside the person's box
                face_center_x = (face_left + face_right) // 2
                face_center_y = (face_top + face_bottom) // 2

                if (person_box_left <= face_center_x <= person_box_right and
                    person_box_top <= face_center_y <= person_box_bottom):
                    
                    # 5. Use robust face matching for the associated face
                    if self.known_face_encodings:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if face_distances[best_match_index] < self.face_tolerance:
                            name = self.known_face_names[best_match_index]
                            is_known = True
                    
                    # Once a face is matched to this person, stop checking other faces
                    break 
            
            # 6. Update counter and draw labels
            if not is_known:
                unknown_person_count += 1
                color = (0, 0, 255) # Red for Unknown
                label = f"Unknown ({conf:.2f})"
            else:
                color = (0, 255, 0) # Green for Known
                label = f"{name} ({conf:.2f})"
            
            annotator.box_label(box, label, color=color)
        # else:
        #     annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        # 7. Trigger alarm based on the COUNT of unknown people and the 'records' threshold
        if unknown_person_count >= self.records:
            self.play_sound()
        else:
            self.reset_sound()

        plot_im = annotator.result()
        self.display_output(plot_im)
        
        # Display track count on the frame
        total_tracks = len(getattr(self, "track_ids", []))
        cv2.putText(plot_im, f"Tracks: {total_tracks}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # return SolutionResults(im0=im0, plot_im=plot_im)
        # return SolutionResults(plot_im=plot_im, im0=im0, total_tracks=len(getattr(self, "track_ids", [])), email_sent=self.email_sent, sound_played=self.sound_played)
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), email_sent=self.email_sent)

# ========== 🎥 MAIN LOOP ==========
if __name__ == "__main__":
    cap = cv2.VideoCapture("media_files/animal_surveillance/goru-churi.mp4")
    # cap = cv2.VideoCapture("./media_files/WIN_20251103_14_11_20_Pro.mp4")
    assert cap.isOpened(), "Error: video not found or cannot be opened."

    w, h, fps = (int(cap.get(x)) for x in
                 (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter("security_output.avi",
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    face_alarm = FaceRecognitionAlarm(
        show=True,
        model="yolo11m.pt",
        records=1,
        # classes=[0, 19],  # person
        known_face_encodings=known_face_encodings, 
        known_face_names=known_face_names,
        conf=0.5,
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video finished or empty frame.")
            break

        results = face_alarm(frame)
        # writer.write(results.plot_im)
        # cv2.imshow("Face Recognition Security Alarm", results.plot_im)
        # pygame.event.pump()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    pygame.quit()
    print("[INFO] Surveillance session ended.")
