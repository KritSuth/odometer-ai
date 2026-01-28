from ultralytics import YOLO

model = YOLO("yolov11s.pt")

train_results = model.train(
    data="data/data.yaml",
    epochs=150,
    imgsz=640,
    batch=32,
    device=0,
    workers=8
)
