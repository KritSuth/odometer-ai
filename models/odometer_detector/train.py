from ultralytics import YOLO

def main():
    model = YOLO("yolo11s.pt")
    
    model.train(
        data="data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",
        workers=8
    )

if __name__ == "__main__":
    main()
