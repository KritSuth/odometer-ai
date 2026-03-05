from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train2/weights/best.pt")
    
    model.train(
        data="data/data.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        device="0",
        workers=8,
        lr0=0.0005,
        patience=10
    )

if __name__ == "__main__":
    main()
