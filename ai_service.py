from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import concurrent.futures
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = "model/yolo_single.pt"
CNN_MODEL_PATH = "model/cnn_mobilenetv2_jambu.h5"

IMG_SIZE    = 224
CLASS_NAMES = ["A", "B", "C"]

# Threshold — diturunkan sedikit agar tidak terlalu strict
MIN_CONFIDENCE = 0.25
MIN_BOX_AREA   = 0.01
# MIN_CONFIDENCE = 0.40
# MIN_BOX_AREA   = 0.02

print("🚀 Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model  = load_model(CNN_MODEL_PATH)

# =========================
# INFERENCE LOCK
# YOLO tidak thread-safe — tanpa lock, request bersamaan
# saling ganggu dan hasilnya kosong.
# Lock ini membuat request antri, bukan ditolak.
# =========================
inference_lock = asyncio.Lock()


# =========================
# PREPROCESS
# =========================
def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def extract_jambu_only(crop):
    try:
        mask = np.zeros(crop.shape[:2], np.uint8)

        bgModel = np.zeros((1, 65), np.float64)
        fgModel = np.zeros((1, 65), np.float64)

        h, w = crop.shape[:2]
        rect = (5, 5, w - 10, h - 10)

        cv2.grabCut(crop, mask, rect, bgModel, fgModel, 3, cv2.GC_INIT_WITH_RECT)

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
        mask = np.clip(mask, 0, 1)

        result = crop.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= mask

        result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)
        return result

    except:
        return crop


def make_circle_crop(img):
    h, w = img.shape[:2]

    mask   = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.48)

    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


# =========================
# CORE PROCESS
# Fungsi sync — dijalankan di thread pool via run_in_executor
# =========================
def process_image_np(img):
    h, w     = img.shape[:2]
    img_area = h * w
    # results  = yolo_model(img)
    results = yolo_model(img, imgsz=640, conf=0.25)

    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            box_area = (x2 - x1) * (y2 - y1)
            if box_area / img_area < MIN_BOX_AREA:
                continue

            pad_x = int(0.05 * (x2 - x1))
            pad_y = int(0.05 * (y2 - y1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = extract_jambu_only(crop)
            crop = normalize_lighting(crop)
            crop = make_circle_crop(crop)

            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop_array   = np.expand_dims(crop_resized, axis=0)
            crop_array   = preprocess_input(crop_array)

            pred       = cnn_model.predict(crop_array, verbose=0)[0]
            class_idx  = np.argmax(pred)
            confidence = float(pred[class_idx])
            label      = CLASS_NAMES[class_idx]

            if confidence < MIN_CONFIDENCE:
                print(f"  ⚠ Deteksi diabaikan: confidence {confidence:.2f} < {MIN_CONFIDENCE}")
                continue

            detections.append({"label": label, "confidence": confidence})

    if len(detections) == 0:
        return None

    labels      = [d["label"] for d in detections]
    confidences = [d["confidence"] for d in detections]

    count_A = labels.count("A")
    count_B = labels.count("B")
    count_C = labels.count("C")

    if count_A > count_B and count_A > count_C:
        final_grade = "A"
    elif count_B > count_A and count_B > count_C:
        final_grade = "B"
    elif count_C > count_A and count_C > count_B:
        final_grade = "C"
    else:
        score = 0
        for l, c in zip(labels, confidences):
            if l == "A":
                score += 3 * c
            elif l == "B":
                score += 2 * c
            else:
                score += 1 * c

        avg = score / sum(confidences)

        if avg >= 2.3:
            final_grade = "A"
        elif avg >= 1.7:
            final_grade = "B"
        else:
            final_grade = "C"

    return {
        "final_grade":    final_grade,
        "confidence_avg": float(sum(confidences) / len(confidences)),
        "total_detected": len(detections),
    }


# =========================
# ENDPOINT /grade
# =========================
@app.post("/grade")
async def grade(images: list[UploadFile] = File(...)):
    # Baca semua file DULU sebelum masuk lock (I/O tidak butuh lock)
    image_data = []
    for idx, file in enumerate(images):
        contents = await file.read()
        np_arr   = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        side     = "front" if idx == 0 else "back"
        image_data.append({"side": side, "img": img})

    per_image     = []
    valid_results = []

    # Semua inference dalam 1 lock per request
    # Request lain akan menunggu di sini sampai giliran
    async with inference_lock:
        print(f"\n🔒 Lock — memproses {len(image_data)} foto")

        for idx, item in enumerate(image_data):
            side = item["side"]
            img  = item["img"]

            print(f"📸 Processing image {idx} ({side})")

            if img is None:
                per_image.append({
                    "side":  side,
                    "error": "cannot_decode_image",
                    "valid": False,
                })
                continue

            # run_in_executor agar inference tidak block event loop
            # sehingga FastAPI tetap bisa terima koneksi baru saat menunggu
            res = await asyncio.get_event_loop().run_in_executor(
                executor,
                process_image_np,
                img
            )
            # loop = asyncio.get_event_loop()
            # res  = await loop.run_in_executor(None, process_image_np, img)

            if res is None:
                print(f"  ⚠ Image {idx}: no jambu detected")
                per_image.append({
                    "side":  side,
                    "error": "no_jambu_detected",
                    "valid": False,
                })
                continue

            result = {
                "side":           side,
                "grade":          res["final_grade"],
                "confidence":     res["confidence_avg"],
                "total_detected": res["total_detected"],
                "valid":          True,
            }
            per_image.append(result)
            valid_results.append(result)

        print(f"🔓 Lock selesai — {len(valid_results)}/{len(image_data)} valid")

    if len(valid_results) == 0:
        print("❌ No valid jambu detected in any image")
        return {
            "success":    False,
            "error_code": "no_jambu_detected",
            "message":    "Jambu tidak terdeteksi. Pastikan foto jelas dan berisi buah jambu.",
            "ai_result":  per_image,
        }

    score_map   = {"A": 3, "B": 2, "C": 1}
    score       = sum(score_map[p["grade"]] for p in valid_results) / len(valid_results)
    final_grade = "A" if score >= 2.5 else "B" if score >= 1.5 else "C"

    confidence_avg  = float(sum(p["confidence"] for p in valid_results) / len(valid_results))
    consistency     = "HIGH" if len(set(p["grade"] for p in valid_results)) == 1 else "LOW"
    defect_detected = any(p["grade"] == "C" for p in valid_results)

    print(f"✅ Result: grade={final_grade}, confidence={confidence_avg:.2f}")

    return {
        "success":         True,
        "grade":           final_grade,
        "confidence_avg":  confidence_avg,
        "total_detected":  sum(p["total_detected"] for p in valid_results),
        "ai_result":       per_image,
        "consistency":     consistency,
        "defect_detected": defect_detected,
    }


@app.post("/predict")
async def predict(images: list[UploadFile] = File(...)):
    return await grade(images)