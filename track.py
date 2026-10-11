from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.track(source="test_video.mp4", tracker = "bytetrack.yaml", save= True, stream= True,)

for result in results:
    pass