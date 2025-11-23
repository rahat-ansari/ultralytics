import os
import cv2
import numpy as np
import face_recognition
from ultralytics import solutions
from ultralytics.utils import LOGGER
import pygame
# Import internal components from the solutions module
from ultralytics.solutions.solutions import SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

# --- 1. Pygame and Known Faces Setup (Global) ---

# Initialize alarm sound safely
try:
    pygame.mixer.init()
except Exception as e:
    LOGGER.warning(f"Pygame mixer init failed: {e}")

alarm_file = "alarm.mp3"
if os.path.exists(alarm_file):
    try:
        pygame.mixer.music.load(alarm_file)
    except Exception as e:
        LOGGER.warning(f"Failed to load alarm file '{alarm_file}': {e}")
else:
    LOGGER.warning(f"Alarm file '{alarm_file}' not found — please check the path.")

# Define the directory for known faces
KNOWN_FACES_DIR = "family_members"

# --- 2. FaceRecognitionAlarm Class (Consolidated logic) ---
class FaceRecognitionAlarm(solutions.SecurityAlarm):
    """
    A security alarm that uses face recognition to trigger alerts only for unknown persons.
    """

    def __init__(self, face_data_path, **kwargs):
        # Initialize base class (may set attributes like records, names, etc.)
        super().__init__(**kwargs)

        # Ensure expected attributes exist with sensible defaults if base class did not provide them
        self.records = getattr(self, "records", kwargs.get("records", 1))
        self.email_sent = getattr(self, "email_sent", False)
        self.sound_played = getattr(self, "sound_played", False)
        # detection outputs: some base classes use 'clss' while others use 'cls'
        self.boxes = getattr(self, "boxes", [])
        # Keep both possible names, prefer existing attribute on instance or kwargs
        self.clss = getattr(self, "clss", getattr(self, "cls", kwargs.get("classes", [])))
        self.names = getattr(self, "names", {0: "person"})
        self.line_width = getattr(self, "line_width", 2)
        self.track_ids = getattr(self, "track_ids", [])

        # Face recognition state
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_data_path = face_data_path
        self.face_tolerance = 0.55

        # Load known faces once during initialization
        self._load_known_faces()

    def _load_known_faces(self):
        """Loads face encodings from the specified directory and updates instance attributes."""
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.exists(self.face_data_path):
             LOGGER.warning(f"Face data path '{self.face_data_path}' not found. Cannot load known faces.")
             return

        # Use os.walk to search through subdirectories
        for root, dirs, files in os.walk(self.face_data_path):
            for image_name in files:
                if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, image_name)
                    # The person name is the parent directory name
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
            LOGGER.info(f"Successfully loaded {len(self.known_face_encodings)} face encodings for {len(set(self.known_face_names))} people")
        else:
            LOGGER.warning("No known faces were loaded. All detected persons will be considered unknown.")

    def _resolve_detection_lists(self, results):
        """
        Robustly extract boxes, classes, names mapping from results or instance attributes.
        Returns boxes_list (list of 4-tuples), classes_list (list), names_map (dict).
        """
        # Boxes
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            boxes = getattr(self, "boxes", [])
        # If boxes is a list of objects with .xyxy attribute (common in ultralytics),
        # try to extract numeric lists
        boxes_list = []
        try:
            # If boxes is a list of 4-tuples already
            if isinstance(boxes, (list, tuple)) and boxes and isinstance(boxes[0], (list, tuple, np.ndarray)):
                for b in boxes:
                    # ensure exactly 4 values
                    if len(b) >= 4:
                        boxes_list.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            else:
                # Attempt to handle objects with .xyxy or .xyxy[0]
                for b in boxes:
                    # Many ultralytics boxes provide .xyxy or .xyxy[0]
                    xy = None
                    if hasattr(b, "xyxy"):
                        try:
                            arr = b.xyxy
                            # arr may be a tensor; convert to list
                            xy = list(arr[0]) if len(arr) and hasattr(arr[0], "__len__") else list(arr)
                        except Exception:
                            pass
                    if xy is None and hasattr(b, "xyxy"):
                        try:
                            xy = list(b.xyxy)
                        except Exception:
                            pass
                    if xy is None and hasattr(b, "data"):
                        try:
                            # fallback: some box objects have .data or similar
                            xy = list(b.data[:4])
                        except Exception:
                            pass
                    if xy:
                        boxes_list.append([int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])])
        except Exception:
            # As ultimate fallback, try to treat boxes as list-like raw values
            try:
                for b in boxes:
                    if len(b) >= 4:
                        boxes_list.append([int(b[0]), int(b[1]), int(b[2]), int(b[3])])
            except Exception:
                boxes_list = []

        # Classes (cls/clss)
        classes = getattr(results, "clss", None)
        if classes is None:
            classes = getattr(results, "cls", None)
        if classes is None:
            classes = getattr(self, "clss", getattr(self, "cls", []))

        classes_list = []
        try:
            if isinstance(classes, (list, tuple, np.ndarray)):
                classes_list = [int(c) for c in classes]
            else:
                # Maybe each box has .cls attribute
                for b in boxes:
                    if hasattr(b, "cls"):
                        classes_list.append(int(b.cls))
        except Exception:
            classes_list = []

        # Names mapping
        names_map = getattr(results, "names", None)
        if names_map is None:
            names_map = getattr(self, "names", {0: "person"})

        return boxes_list, classes_list, names_map

    def __call__(self, im0, results=None, annotator=None):
        """
        Process a single frame (im0), annotate with known/unknown labels, and control alarm.
        - results: optional detection results object from the detector; if None, try to call the base class
                   to get detections (if supported).
        """
        if im0 is None:
            LOGGER.error("No input image provided to FaceRecognitionAlarm.")
            return SolutionResults(plot_im=None, im0=None, total_tracks=0, email_sent=self.email_sent, sound_played=self.sound_played)

        # If results were not provided, attempt to call the base class detector (if available)
        if results is None:
            try:
                # Some solution classes are callable and return results when invoked with an image
                results = super().__call__(im0)
            except Exception:
                # If that fails, attempt to access an attribute 'predict' or 'model' - best-effort only
                try:
                    results = self.model(im0)  # may not exist; ignore if not available
                except Exception:
                    results = None

        # Prepare annotator
        if annotator is None:
            try:
                annotator = SolutionAnnotator(im0, line_width=self.line_width)
            except Exception:
                # Fallback: create a simple stub annotator with minimal API used below
                class _StubAnnotator:
                    def __init__(self, img):
                        self.img = img
                    def box_label(self, box, label=None, color=(255, 0, 0)):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(self.img, (x1, y1), (x2, y2), color, 2)
                        if label:
                            cv2.putText(self.img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    def result(self):
                        return self.img
                annotator = _StubAnnotator(im0)

        # Resolve detection lists (robust to multiple formats)
        boxes_list, classes_list, names_map = self._resolve_detection_lists(results)

        unknown_person_count = 0
        # COCO 'person' class id default to 0; user can supply classes mapping in names_map/cls list
        person_cls_id = 0

        # Resize frame for faster face recognition and convert to RGB
        try:
            small_frame = cv2.resize(im0, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            LOGGER.error(f"Error preparing frame for face recognition: {e}")
            rgb_small_frame = None

        face_locations = []
        face_encodings = []
        if rgb_small_frame is not None:
            try:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            except Exception as e:
                LOGGER.error(f"Face recognition error: {e}")
                face_locations = []
                face_encodings = []

        # Process each detection box
        for i, box in enumerate(boxes_list):
            # Safely get class id for this detection
            cls_id = None
            try:
                cls_id = classes_list[i] if i < len(classes_list) else None
            except Exception:
                cls_id = None

            label = names_map.get(cls_id, str(cls_id)) if isinstance(names_map, dict) else str(cls_id)
            color = colors(cls_id, True) if isinstance(names_map, dict) else (255, 255, 255)

            # Only proceed if the detection is a person (or cls_id is None and user wants to treat all as person)
            if cls_id == person_cls_id or cls_id is None:
                x1, y1, x2, y2 = map(int, box)
                found_face = False

                # Check each detected face against the YOLO person box
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale face locations back to original image size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Check if the center of the face is inside the YOLO person box
                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2

                    if x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2:
                        found_face = True
                        if self.known_face_encodings:
                            try:
                                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.55)
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                if len(face_distances) > 0:
                                    best_match_index = int(np.argmin(face_distances))
                                    if matches[best_match_index]:
                                        name = self.known_face_names[best_match_index]
                                        label = f"{name} (Known)"
                                        color = (0, 255, 0)  # Green
                                    else:
                                        label = "Unknown (ALARM!)"
                                        color = (0, 0, 255)  # Red
                                        unknown_person_count += 1
                                else:
                                    label = "Unknown (ALARM!)"
                                    color = (0, 0, 255)
                                    unknown_person_count += 1
                            except Exception as e:
                                LOGGER.error(f"Face comparison error: {e}")
                                label = "Unknown (ALARM!)"
                                color = (0, 0, 255)
                                unknown_person_count += 1
                        else:
                            label = "Unknown (ALARM!)"
                            color = (0, 0, 255)  # Red
                            unknown_person_count += 1
                        break  # Face associated with this person box handled

                if not found_face:
                    # No face inside the detected person box
                    label = f"{names_map.get(cls_id, 'person')} (No Face)"

            # Annotate the box with the determined label and color
            try:
                annotator.box_label(box, label=label, color=color)
            except Exception:
                # fallback drawing
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(im0, (x1, y1), (x2, y2), color, 2)
                if label:
                    cv2.putText(im0, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- Alarm Control ---
        # Trigger alarm based on the count of 'Unknown' persons
        if unknown_person_count >= self.records:
            if not self.email_sent:
                # self.send_email(im0, unknown_person_count) # Uncomment if email is setup
                self.email_sent = True
                LOGGER.info(f"📧 Email alert condition met for {unknown_person_count} unknown person(s).")
            if not self.sound_played:
                if pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                    LOGGER.info("🚨 Playing security alarm!")
                    try:
                        pygame.mixer.music.play()
                        self.sound_played = True
                    except Exception as e:
                        LOGGER.error(f"Failed to play alarm sound: {e}")
        else:
            # Reset logic: only reset flags if the sound has finished playing
            # A better approach is to let the sound finish.
            if self.sound_played and pygame.mixer.get_init() and not pygame.mixer.music.get_busy():
                self.email_sent = False
                self.sound_played = False
                LOGGER.info("🟢 Alarm reset.")


        # --- End Alarm Control ---

        plot_im = annotator.result()
        # If base class has display_output method, call it to maintain compatibility
        try:
            self.display_output(plot_im)
        except Exception:
            pass

        # Return the standard SolutionResults object (best-effort fields)
        try:
            total_tracks = len(self.track_ids) if hasattr(self, "track_ids") else 0
        except Exception:
            total_tracks = 0

        return SolutionResults(
            plot_im=plot_im,
            im0=im0,
            total_tracks=total_tracks,
            email_sent=self.email_sent,
            sound_played=self.sound_played,
        )

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Open video
    cap = cv2.VideoCapture("media_files/animal_surveillance/goru-churi.mp4")
    assert cap.isOpened(), "❌ Error: Cannot read video file."

    # Video writer setup
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("security_output.avi", fourcc, fps, (w, h))

    # Initialize the FaceRecognitionAlarm
    # Alarm triggers if 3 or more UNKNOWN persons are detected.
    security_alarm = FaceRecognitionAlarm(
        show=True,
        model="yolo11m.pt",  # Using a standard model
        records=3,
        # classes=[0],  # Only detect 'person' (ID 0) for face recognition
        face_data_path=KNOWN_FACES_DIR
    )

    print("--- Starting Video Processing. Press 'q' to terminate. ---")
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("✅ Video processing completed.")
            break

        # Run detection and alarm logic
        try:
            results = security_alarm(im0)
        except Exception as e:
            LOGGER.error(f"Processing frame failed: {e}")
            break

        # Write the processed frame to the output video
        if results and getattr(results, "plot_im", None) is not None:
            video_writer.write(results.plot_im)

        # Check for 'q' key press to terminate
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Termination key 'q' pressed. Stopping...")
            break

        # Allow pygame to process events to keep the sound responsive (optional)
        try:
            pygame.event.pump()
        except Exception:
            pass

    # Cleanup
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    try:
        pygame.mixer.quit()
    except Exception:
        pass
    print("--- Resources released. ---")