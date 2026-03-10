from ultralytics import YOLO

def main():
    model = YOLO("../../runs/detect/train5/weights/best.pt")

    model.train(
        data="data_cropped/data.yaml",
        epochs=30,
        imgsz=768,
        batch=16,
        device=0,
        workers=8,      # ใช้หลาย worker ได้แล้ว
        patience=10,
        cache=True,
        # lr0 = 0.0005
    )

if __name__ == "__main__":
    main()