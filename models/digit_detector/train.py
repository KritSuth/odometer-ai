from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")

    model.train(
        data="models/digit_detector/data_cropped/data.yaml",
        epochs=200,
        imgsz=640,
        batch=32,
        device=0,
        workers=8,      # ใช้หลาย worker ได้แล้ว
        patience=30
    )

if __name__ == "__main__":
    main()