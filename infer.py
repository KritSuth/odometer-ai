from ultralytics import YOLO
import cv2
from collections import defaultdict

# model แรก: detect odometer (1 class)
model_odo = YOLO("odometer_last.pt")
# model ที่สอง: detect digit (0-9, "-", X)
model_digit = YOLO("digit_last.pt")

CLASS_TO_CHAR = {
    "-": ".",   # จุดทศนิยม
    "X": None   # ignore
}

img = cv2.imread("input.jpg")
if img is None:
    raise ValueError("❌ โหลดรูปไม่สำเร็จ")

res_odo = model_odo(img, conf=0.4)[0]

if len(res_odo.boxes) == 0:
    raise RuntimeError("❌ ไม่เจอ odometer")

box = res_odo.boxes[0]
x1, y1, x2, y2 = map(int, box.xyxy[0])

odo_crop = img[y1:y2, x1:x2]

res_digit = model_digit(odo_crop, conf=0.25)[0]

digits = []

for b in res_digit.boxes:
    cls_id = int(b.cls[0])
    cls_name = model_digit.names[cls_id]

    # map "-" -> "." , "X" -> ignore
    char = CLASS_TO_CHAR.get(cls_name, cls_name)
    if char is None:
        continue

    x, y, w, h = map(float, b.xywh[0])

    digits.append({
        "char": char,
        "x": x,
        "y": y,
        "conf": float(b.conf[0])
    })

lines = defaultdict(list)

for d in digits:
    key = round(d["y"] / 20)  # cluster ตามแนวแกน Y
    lines[key].append(d)

# เลือกแถวที่มี digit เยอะที่สุด
main_line = max(lines.values(), key=len)

main_line.sort(key=lambda d: d["x"])

mileage = "".join(d["char"] for d in main_line)

print("📏 mileage =", mileage)


