from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================
# YOLO_MODEL_PATH = r"runs/detect/train2/weights/best.pt"
YOLO_MODEL_PATH = r"jambu-detection.v4-last.yolov8\runs\detect\train\weights\best.pt" 
CNN_MODEL_PATH = "model/final_mobilenetv2_jambu.h5"

IMG_SIZE = 224
CLASS_NAMES = ["A", "B", "C"]

print("🚀 Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
cnn_model = load_model(CNN_MODEL_PATH)

# =========================
# PREPROCESS
# =========================
def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))
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

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.48)

    cv2.circle(mask, center, radius, 255, -1)

    return cv2.bitwise_and(img, img, mask=mask)


# =========================
# CORE PROCESS
# =========================
def process_image_np(img):
    h, w = img.shape[:2]
    results = yolo_model(img)

    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

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
            crop_array = np.expand_dims(crop_resized, axis=0)
            crop_array = preprocess_input(crop_array)

            pred = cnn_model.predict(crop_array, verbose=0)[0]

            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            label = CLASS_NAMES[class_idx]

            detections.append({
                "label": label,
                "confidence": confidence
            })

    if len(detections) == 0:
        return None

    labels = [d["label"] for d in detections]
    confidences = [d["confidence"] for d in detections]

    count_A = labels.count("A")
    count_B = labels.count("B")
    count_C = labels.count("C")

    # ===== FINAL GRADE (PER IMAGE) =====
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

    confidence_avg = float(sum(confidences) / len(confidences))

    return {
        "final_grade": final_grade,
        "confidence_avg": confidence_avg,
        "total_detected": len(detections)
    }


# =========================
# API ENDPOINT
# =========================
@app.post("/grade")
async def grade(images: list[UploadFile] = File(...)):
    per_image = []
    valid_results = []

    for idx, file in enumerate(images):
        print(f"\n📸 Processing image {idx}")

        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        side = "front" if idx == 0 else "back"

        if img is None:
            per_image.append({
                "side": side,
                "grade": None,
                "confidence": 0.0,
                "total_detected": 0,
                "error": "Cannot decode image"
            })
            continue

        res = process_image_np(img)

        if res is None:
            per_image.append({
                "side": side,
                "grade": None,
                "confidence": 0.0,
                "total_detected": 0,
                "error": "No object detected"
            })
            continue

        result = {
            "side": side,
            "grade": res["final_grade"],
            "confidence": res["confidence_avg"],
            "total_detected": res["total_detected"]
        }

        per_image.append(result)
        valid_results.append(result)
        

    if len(valid_results) == 0:
        return {
            "error": "No object detected in any image",
            "ai_result": per_image
        }

    if res is None:
        print("❌ No detection")
    else:
        print("✅ Detection result:", res)
    # ===== FINAL PER BUAH =====
    score_map = {"A": 3, "B": 2, "C": 1}
    score = sum(score_map[p["grade"]] for p in valid_results) / len(valid_results)

    if score >= 2.5:
        final_grade = "A"
    elif score >= 1.5:
        final_grade = "B"
    else:
        final_grade = "C"

    confidence_avg = float(sum(p["confidence"] for p in valid_results) / len(valid_results))

    consistency = "HIGH" if len(set(p["grade"] for p in valid_results)) == 1 else "LOW"
    defect_detected = any(p["grade"] == "C" for p in valid_results)

    return {
        "grade": final_grade,
        "confidence_avg": confidence_avg,
        "total_detected": sum(p["total_detected"] for p in valid_results),

        "ai_result": per_image,

        "consistency": consistency,
        "defect_detected": defect_detected
    }



@app.post("/predict")
async def predict(images: list[UploadFile] = File(...)):
    return await grade(images)
# from fastapi import FastAPI, UploadFile, File
# import shutil
# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# app = FastAPI()

# # =========================
# # CONFIG
# # =========================
# YOLO_MODEL_PATH = "model/yolo_single.pt"
# CNN_MODEL_PATH = "model/cnn_mobilenetv2_jambu.h5"

# IMG_SIZE = 224
# CLASS_NAMES = ["A", "B", "C"]

# TEMP_DIR = "temp_upload"
# os.makedirs(TEMP_DIR, exist_ok=True)

# # =========================
# # LOAD MODEL (sekali saja)
# # =========================
# print("🚀 Loading AI models...")
# yolo_model = YOLO(YOLO_MODEL_PATH)
# cnn_model = load_model(CNN_MODEL_PATH)

# # =========================
# # PREPROCESSING
# # =========================
# def normalize_lighting(img):
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     l = clahe.apply(l)

#     lab = cv2.merge((l,a,b))
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# def extract_jambu_only(crop):
#     try:
#         mask = np.zeros(crop.shape[:2], np.uint8)

#         bgModel = np.zeros((1, 65), np.float64)
#         fgModel = np.zeros((1, 65), np.float64)

#         h, w = crop.shape[:2]
#         rect = (5, 5, w - 10, h - 10)

#         cv2.grabCut(crop, mask, rect, bgModel, fgModel, 3, cv2.GC_INIT_WITH_RECT)

#         mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#         mask = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
#         mask = np.clip(mask, 0, 1)

#         result = crop.astype(np.float32)

#         for c in range(3):
#             result[:, :, c] = result[:, :, c] * mask

#         result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)

#         return result

#     except:
#         return crop


# def make_circle_crop(img):
#     h, w = img.shape[:2]

#     mask = np.zeros((h, w), dtype=np.uint8)
#     center = (w // 2, h // 2)
#     radius = int(min(w, h) * 0.48)

#     cv2.circle(mask, center, radius, 255, -1)

#     return cv2.bitwise_and(img, img, mask=mask)

# # =========================
# # PROCESS 1 IMAGE
# # =========================
# def process_image(path):
#     img = cv2.imread(path)

#     if img is None:
#         return None, 0

#     results = yolo_model(img)

#     best_box = None
#     best_conf = 0

#     # ambil 1 jambu terbaik (karena 1 gambar = 1 buah)
#     for r in results:
#         boxes = r.boxes
#         if boxes is None:
#             continue

#         for i in range(len(boxes)):
#             conf = float(boxes.conf[i])
#             if conf > best_conf:
#                 best_conf = conf
#                 best_box = boxes.xyxy[i].cpu().numpy()

#     if best_box is None:
#         return None, 0

#     x1, y1, x2, y2 = best_box.astype(int)

#     crop = img[y1:y2, x1:x2]

#     # preprocessing
#     crop = extract_jambu_only(crop)
#     crop = normalize_lighting(crop)
#     crop = make_circle_crop(crop)

#     # CNN
#     crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
#     arr = np.expand_dims(crop, axis=0)
#     arr = preprocess_input(arr)

#     pred = cnn_model.predict(arr, verbose=0)[0]

#     idx = np.argmax(pred)
#     confidence = float(pred[idx])
#     label = CLASS_NAMES[idx]

#     return label, confidence

# # =========================
# # COMBINE 2 GAMBAR (FINAL)
# # =========================
# def combine(g1, g2):
#     score_map = {"A": 3, "B": 2, "C": 1}

#     avg = (score_map[g1] + score_map[g2]) / 2

#     if avg >= 2.5:
#         return "A", avg
#     elif avg >= 1.5:
#         return "B", avg
#     else:
#         return "C", avg

# # =========================
# # API ENDPOINT
# # =========================
# @app.post("/grading")
# async def grading(
#     image1: UploadFile = File(...),
#     image2: UploadFile = File(...)
# ):
#     try:
#         path1 = os.path.join(TEMP_DIR, image1.filename)
#         path2 = os.path.join(TEMP_DIR, image2.filename)

#         # simpan file sementara
#         with open(path1, "wb") as f:
#             shutil.copyfileobj(image1.file, f)

#         with open(path2, "wb") as f:
#             shutil.copyfileobj(image2.file, f)

#         # proses AI
#         g1, c1 = process_image(path1)
#         g2, c2 = process_image(path2)

#         if g1 is None or g2 is None:
#             return {
#                 "status": "error",
#                 "message": "deteksi gagal"
#             }

#         final, score = combine(g1, g2)

#         return {
#             "status": "success",
#             "data": {
#                 "front": {
#                     "grade": g1,
#                     "confidence": round(c1, 3)
#                 },
#                 "back": {
#                     "grade": g2,
#                     "confidence": round(c2, 3)
#                 },
#                 "final": {
#                     "grade": final,
#                     "score": round(score, 2),
#                     "method": "average_scoring"
#                 }
#             }
#         }

#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e)
#         }