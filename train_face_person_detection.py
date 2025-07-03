from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
# You'll need to create a dataset.yaml file with face and person classes
results = model.train(
    data="face_person_dataset.yaml",  # path to your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    name="face_person_detector"
)