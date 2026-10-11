from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(source="https://ultralytics.com/images/bus.jpg", save = True,)