# from string import digits
# from matplotlib import lines
# from ultralytics import YOLO
# import cv2
# from collections import defaultdict

# # ================= LOAD MODELS =================
# model_odo = YOLO("models/odometer_detector/runs/detect/train2/weights/last.pt")
# model_digit = YOLO("runs/detect/train5/weights/last.pt")

# # class mapping (ตาม dataset)
# CLASS_TO_CHAR = {
#     "-": ".",   # decimal
#     "X": None   # ignore / noise
# }

# # ============================================================
# # MAIN PIPELINE
# # ============================================================
# def recognize_odometer_two_stage(img):
#     """
#     img : numpy array (BGR)
#     return : dict (backend-friendly)
#     """

#     # --------------------------------------------------------
#     # 1) DETECT ODOMETER
#     # --------------------------------------------------------
#     res_odo = model_odo(img, conf=0.4, verbose=False)[0]

#     if len(res_odo.boxes) == 0:
#         return {"success": False, "message": "Odometer not found"}

#     # เลือก box ที่มั่นใจที่สุด
#     odo_box = max(res_odo.boxes, key=lambda b: float(b.conf[0]))
#     x1, y1, x2, y2 = map(int, odo_box.xyxy[0])

#     odo_crop = img[y1:y2, x1:x2]
#     if odo_crop.size == 0:
#         return {"success": False, "message": "Invalid odometer crop"}

#     # --------------------------------------------------------
#     # 2) DETECT DIGITS
#     # --------------------------------------------------------
#     res_digit = model_digit(odo_crop, conf=0.25, verbose=False)[0]

#     digits = []
#     for b in res_digit.boxes:
#         cls_id = int(b.cls[0])
#         cls_name = model_digit.names[cls_id]
#         conf = float(b.conf[0])

#         char = CLASS_TO_CHAR.get(cls_name, cls_name)
#         if char is None:
#             continue

#         x, y, w, h = map(float, b.xywh[0])

#         digits.append({
#             "digit": char,
#             "x": x,
#             "y": y,
#             "conf": conf,
#             "bbox": list(map(int, b.xyxy[0]))
#         })

#     if not digits:
#         return {"success": False, "message": "No digits detected"}

#     # --------------------------------------------------------
#     # 3) CLUSTER ตามแนว Y → เลือกบรรทัดหลัก
#     # --------------------------------------------------------

# # ---------- 3. cluster Y ----------
#    # --------------------------------------------------------
# # 3) CLUSTER Y → เลือกเลขไมล์จริง
# # --------------------------------------------------------
#     h = odo_crop.shape[0]
#     center_y = h / 2

#     lines = defaultdict(list)
#     for d in digits:
#         key = round(d["y"] / (h * 0.12))
#         lines[key].append(d)

# # ---------- FILTER LINE ที่เป็นไปได้ ----------
#     candidates = []
#     for line in lines.values():
#         if len(line) < 4:
#             continue  # เลขไมล์ต้องยาวพอ
 
#         avg_y = sum(d["y"] for d in line) / len(line)

#     # เลขไมล์จริงต้องอยู่ "เหนือ" center
#         if avg_y >= center_y:
#             continue

#         candidates.append(line)

#     if not candidates:
#         return {"success": False, "message": "No valid digit line found"}

# # ---------- เลือก line ที่ใกล้ center มากที่สุด ----------
#     def distance_to_center(line):
#         avg_y = sum(d["y"] for d in line) / len(line)
#         return abs(avg_y - center_y)

#     main_line = min(candidates, key=distance_to_center)

# # ---------- sort ซ้าย → ขวา ----------
#     main_line.sort(key=lambda d: d["x"])


#     # --------------------------------------------------------
#     # 5) BUILD RESULT
#     # --------------------------------------------------------
#     value = "".join(d["digit"] for d in main_line)
#     confidence = sum(d["conf"] for d in main_line) / len(main_line)

#     return {
#         "success": True,
#         "value": value,
#         "digit_count": len(value),
#         "confidence": round(confidence, 4),
#         "digits": [
#             {
#                 "digit": d["digit"],
#                 "x": round(d["x"], 2),
#                 "y": round(d["y"], 2),
#                 "conf": round(d["conf"], 4)
#             }
#             for d in main_line
#         ]
#     }

# # ============================================================
# # DEBUG UTILITIES
# # ============================================================
# def draw_boxes(img, boxes, names, color):
#     for b in boxes:
#         x1, y1, x2, y2 = map(int, b.xyxy[0])
#         cls_id = int(b.cls[0])
#         cls_name = names[cls_id]
#         conf = float(b.conf[0])

#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(
#             img,
#             f"{cls_name} {conf:.2f}",
#             (x1, max(y1 - 5, 15)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.45,
#             color,
#             1,
#             cv2.LINE_AA
#         )

# def debug_recognize_image(image_path, show=True, save_path=None):
#     img = cv2.imread(image_path)
#     if img is None:
#         print("❌ Cannot read image")
#         return

#     print("\n=== DETECT ODOMETER ===")
#     res_odo = model_odo(img, conf=0.4, verbose=False)[0]

#     if len(res_odo.boxes) == 0:
#         print("❌ Odometer not found")
#         return

#     odo_box = max(res_odo.boxes, key=lambda b: float(b.conf[0]))
#     x1, y1, x2, y2 = map(int, odo_box.xyxy[0])
#     odo_crop = img[y1:y2, x1:x2].copy()

#     print(f"Odometer conf: {float(odo_box.conf[0]):.3f}")
#     draw_boxes(img, [odo_box], model_odo.names, (255, 0, 0))

#     print("\n=== DETECT DIGITS (RAW) ===")
#     res_digit = model_digit(odo_crop, conf=0.25, verbose=False)[0]

#     raw_digits = []
#     for b in res_digit.boxes:
#         cls_id = int(b.cls[0])
#         cls_name = model_digit.names[cls_id]
#         conf = float(b.conf[0])
#         x, y, w, h = map(float, b.xywh[0])

#         char = CLASS_TO_CHAR.get(cls_name, cls_name)
#         print(f"digit={cls_name} -> {char} | conf={conf:.3f} | x={x:.1f}, y={y:.1f}")

#         if char is not None:
#             raw_digits.append({
#                 "digit": char,
#                 "x": x,
#                 "y": y,
#                 "conf": conf,
#                 "bbox": list(map(int, b.xyxy[0]))
#             })

#     draw_boxes(odo_crop, res_digit.boxes, model_digit.names, (0, 255, 0))

#     # -------- FINAL PIPELINE RESULT --------
#     print("\n=== FINAL RESULT ===")
#     result = recognize_odometer_two_stage(img)
#     print(result)

#     if show:
#         cv2.imshow("Odometer", img)
#         cv2.imshow("Digits", odo_crop)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     if save_path:
#         cv2.imwrite(save_path, img)

# # ============================================================
# # RUN DEBUG
# # ============================================================
# if __name__ == "__main__":
#     debug_recognize_image(
#         "test/images/20260213090205_1-925390703.jpg",
#         show=True,
#         save_path="output_debug.jpg"
#     )

from ultralytics import YOLO
import cv2
from collections import defaultdict
import torch
import os

DEBUG_DRAW = True

# ================= DEVICE CHECK =================
def get_device():
    env_device = os.getenv("DEVICE", "cpu").lower()

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if env_device in ["cuda", "gpu"] and torch.cuda.is_available():
        print("✅ Using GPU (CUDA)")
        return "cuda"
    else:
        print("⚠️ Using CPU")
        return "cpu"

DEVICE = get_device()

# ================= LOAD MODELS =================
model_odo = YOLO("models/odometer_detector/runs/detect/train2/weights/last.pt").to(DEVICE)
model_digit = YOLO("runs/detect/train5/weights/last.pt").to(DEVICE)

# class mapping (ตาม dataset)
CLASS_TO_CHAR = {
    "-": ".",   # decimal
    "X": None   # ignore / noise
}

def remove_close_duplicates(digits, x_threshold=2.0):
    """
    Remove digits that are too close in x position.
    Keep the one with higher confidence.
    """
    if not digits:
        return digits

    # sort by x
    digits = sorted(digits, key=lambda d: d["x"])

    filtered = [digits[0]]

    for current in digits[1:]:
        prev = filtered[-1]

        if abs(current["x"] - prev["x"]) <= x_threshold:
            # ถ้าใกล้กัน → เลือก conf สูงกว่า
            if current["conf"] > prev["conf"]:
                filtered[-1] = current
        else:
            filtered.append(current)

    return filtered


def recognize_odometer_two_stage(img):
    res_odo = model_odo(img, conf=0.4, device=DEVICE,verbose=False)[0] #ตรงนี้ยังสามารถลด conf ลงได้อีก

    if len(res_odo.boxes) == 0:
        return {"success": False, "message": "Odometer not found"}

    odo_box = max(res_odo.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, odo_box.xyxy[0])

    odo_crop = img[y1:y2, x1:x2]
    if odo_crop.size == 0:
        return {"success": False, "message": "Invalid odometer crop"}

    res_digit = model_digit(odo_crop, conf=0.25, device=DEVICE, verbose=False)[0]

    digits = []
    for b in res_digit.boxes:
        cls_id = int(b.cls[0])
        cls_name = model_digit.names[cls_id]
        conf = float(b.conf[0])

        char = CLASS_TO_CHAR.get(cls_name, cls_name)
        if char is None:
            continue

        x, y, w, h = map(float, b.xywh[0])
        digits.append({
            "digit": char,
            "x": x,
            "y": y,
            "conf": conf,
            "bbox": list(map(int, b.xyxy[0]))
        })

    if not digits:
        return {"success": False, "message": "No digits detected"}

    # ---------- LINE GROUPING ----------
    h = odo_crop.shape[0]
    center_y = h / 2

    lines = defaultdict(list)
    for d in digits:
        key = round(d["y"] / max(h * 0.1, 10))
        lines[key].append(d)

    candidates = []
    for line in lines.values():
        if len(line) >= 3:
            candidates.append(line)

    if not candidates:
        return {"success": False, "message": "No valid digit line found"}

    def score(line):
        avg_conf = sum(d["conf"] for d in line) / len(line)
        avg_y = sum(d["y"] for d in line) / len(line)
        center_penalty = abs(avg_y - center_y) / center_y
        return avg_conf - center_penalty

    main_line = max(candidates, key=score)
    main_line.sort(key=lambda d: d["x"])

    if len(main_line) > 1:
        spacings = [
            main_line[i+1]["x"] - main_line[i]["x"]
            for i in range(len(main_line)-1)
        ]
        avg_spacing = sum(spacings) / len(spacings)
        x_threshold = avg_spacing * 0.3   # ปรับได้ 0.25 - 0.4
    else:
        x_threshold = 2.0

    main_line = remove_close_duplicates(main_line, x_threshold)

    value = ""
    dot_used = False
    for d in main_line:
        if d["digit"] == ".":
            if dot_used:
                continue
            dot_used = True
        value += d["digit"]

    confidence = sum(d["conf"] for d in main_line) / len(main_line)

    # ================= DEBUG DRAW =================
    if DEBUG_DRAW:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for d in main_line:

            dx1, dy1, dx2, dy2 = d["bbox"]

            cv2.rectangle(
                img,
                (x1 + dx1, y1 + dy1),
                (x1 + dx2, y1 + dy2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                f'{d["digit"]} {d["conf"]:.2f}',
                (x1 + dx1, y1 + dy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

    return {
        "success": True,
        "value": value,
        "digit_count": len(value),
        "confidence": round(confidence, 4),
        "digits": main_line
    }

# ============================================================
# DEBUG UTILITIES
# ============================================================

def draw_boxes(img, boxes, names, color):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cls_id = int(b.cls[0])
        cls_name = names[cls_id]
        conf = float(b.conf[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{cls_name} {conf:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

def merge_two_stage_results(results, max_digits=6, x_threshold_ratio=0.5):

    # valid = [r for r in results if r.get("success")]

    #fix error from line above
    valid = [
        r for r in results
        if r.get("success") and "digits" in r and r["digits"]
    ]

    if not valid:
        return {"success": False, "message": "No valid results"}

    # -----------------------------
    # STEP 1: เลือก reference
    # -----------------------------

    reference = max(
        valid,
        key=lambda r: (len(r["digits"]), r["confidence"])
    )

    ref_digits = sorted(reference["digits"], key=lambda d: d["x"])

    ref_digits = ref_digits[:max_digits]

    ref_positions = [d["x"] for d in ref_digits]

    max_length = len(ref_positions)

    # fix crash error from above line
    if max_length == 0:
        return {"success": False, "message": "Reference has no digits"}

    # average spacing
    if len(ref_positions) > 1:
        spacings = [
            ref_positions[i+1] - ref_positions[i]
            for i in range(len(ref_positions)-1)
        ]
        avg_spacing = sum(spacings) / len(spacings)
    else:
        avg_spacing = 50

    threshold = avg_spacing * x_threshold_ratio

    # -----------------------------
    # STEP 2: build position map
    # -----------------------------

    position_map = {i: [] for i in range(max_length)}

    for r in valid:

        for d in r["digits"]:

            x = d["x"]

            # find closest reference position
            closest_idx = None
            closest_dist = 1e9

            for i, ref_x in enumerate(ref_positions):

                dist = abs(x - ref_x)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_idx = i

            # accept only if within threshold
            # if closest_dist <= threshold:
            #     position_map[closest_idx].append(d)

            # fix crash error from line above
            if (
                closest_idx is not None
                and closest_idx in position_map
                and closest_dist <= threshold
            ):
                position_map[closest_idx].append(d)

    # -----------------------------
    # STEP 3: select best per position
    # -----------------------------

    final_digits = []
    positions = []
    final_conf = []

    for i in range(max_length):

        candidates = position_map[i]

        if not candidates:
            continue

        best = max(candidates, key=lambda d: d["conf"])

        final_digits.append(best["digit"])
        final_conf.append(best["conf"])

        positions.append({
            "position": i,
            "digit": best["digit"],
            "conf": round(best["conf"], 4)
        })

    if not final_digits:
        return {"success": False, "message": "Merge failed"}

    value = "".join(final_digits)

    confidence = sum(final_conf) / len(final_conf)

    return {
        "success": True,
        "value": value,
        "digit_count": len(value),
        "confidence": round(confidence, 4),
        "positions": positions
    }


def debug_recognize_image(image_path, show=True, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read image")
        return

    print("\n=== DETECT ODOMETER ===")
    res_odo = model_odo(img, conf=0.4, device=DEVICE, verbose=False)[0]

    if len(res_odo.boxes) == 0:
        print("Odometer not found")
        return

    odo_box = max(res_odo.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, odo_box.xyxy[0])
    odo_crop = img[y1:y2, x1:x2].copy()

    print(f"Odometer conf: {float(odo_box.conf[0]):.3f}")
    draw_boxes(img, [odo_box], model_odo.names, (255, 0, 0))

    print("\n=== DETECT DIGITS (RAW) ===")
    res_digit = model_digit(odo_crop, conf=0.25, device=DEVICE, verbose=False)[0]

    raw_digits = []
    for b in res_digit.boxes:
        cls_id = int(b.cls[0])
        cls_name = model_digit.names[cls_id]
        conf = float(b.conf[0])
        x, y, w, h = map(float, b.xywh[0])

        char = CLASS_TO_CHAR.get(cls_name, cls_name)
        print(f"digit={cls_name} -> {char} | conf={conf:.3f} | x={x:.1f}, y={y:.1f}")

        if char is not None:
            raw_digits.append({
                "digit": char,
                "x": x,
                "y": y,
                "conf": conf,
                "bbox": list(map(int, b.xyxy[0]))
            })

    draw_boxes(odo_crop, res_digit.boxes, model_digit.names, (0, 255, 0))

    # -------- FINAL PIPELINE RESULT --------
    print("\n=== FINAL RESULT ===")
    result = recognize_odometer_two_stage(img)
    print(result)

    if show:
        cv2.imshow("Odometer", img)
        cv2.imshow("Digits", odo_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, img)

# ============================================================
# RUN DEBUG
# ============================================================

if __name__ == "__main__":
    debug_recognize_image(
        "test/images/20260213081627_1-039463.jpg",
        show=True,
        save_path="output_debug.jpg"
    )




