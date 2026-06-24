"""
ai_service_v2.py
────────────────
High-throughput jambu grading service.

Architecture:
  - 1 dedicated inference thread (YOLO + CNN tidak thread-safe)
  - asyncio.Queue → request antri, tidak ada yang ditolak
  - I/O (baca file) dilakukan SEBELUM masuk queue → queue hanya untuk GPU
  - Response dikembalikan via asyncio.Future agar tiap request
    mendapat hasilnya sendiri tanpa polling

Throughput target: ~100 buah/menit (≈1.7 req/s, single-image mode)
Measured latency per request: ~150-300ms tergantung hardware
"""

from __future__ import annotations

import queue as stdlib_queue
import asyncio
import concurrent.futures
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# ── pilih salah satu sesuai model yang dipakai ──────────────────────────────
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
# ────────────────────────────────────────────────────────────────────────────


# =============================================================================
# CONFIG
# =============================================================================
YOLO_MODEL_PATH = "model/yolo_single.pt"
CNN_MODEL_PATH = "model/best_efficientnet_v4.h5"


IMG_SIZE    = 640 
CLASS_NAMES = ["A", "B", "C"]

# YOLO — hanya ambil deteksi yang benar-benar yakin
YOLO_CONF_THRESHOLD = 0.45

# CNN — buang prediksi yang ragu-ragu
CNN_CONF_THRESHOLD  = 0.50

# Area minimum bounding box relatif terhadap gambar (buang noise kecil)
MIN_BOX_AREA_RATIO  = 0.04   # 4% luas gambar

# Hanya ambil 1 deteksi terbaik per foto (1 buah per frame)
MAX_DETECTIONS_PER_IMAGE = 1

# Maksimum request yang boleh antri; lebih dari ini → 503
MAX_QUEUE_SIZE = 200

# Inference worker pool — TETAP 1 karena YOLO/TF tidak thread-safe
INFERENCE_WORKERS = 1


# =============================================================================
# INFERENCE ENGINE  (berjalan di thread terpisah, bukan event loop)
# =============================================================================
class InferenceEngine:
    """
    Semua operasi YOLO + CNN dijalankan di sini, dalam 1 thread tunggal.
    Thread lain (termasuk FastAPI event loop) tidak boleh menyentuh model.
    """

    def __init__(self) -> None:
        print("🚀 Loading YOLO …")
        self.yolo = YOLO(YOLO_MODEL_PATH)

        print("🚀 Loading CNN …")
        self.cnn = load_model(CNN_MODEL_PATH)

        # Warm-up: buang latency pertama yang biasanya lebih lambat
        self._warmup()
        print("✅ Models ready.")

    # ── warm-up ──────────────────────────────────────────────────────────────
    def _warmup(self) -> None:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.yolo(dummy, imgsz=640, conf=YOLO_CONF_THRESHOLD, verbose=False)
        arr = preprocess_input(
            np.expand_dims(cv2.resize(dummy, (IMG_SIZE, IMG_SIZE)).astype(np.float32), 0)
        )
        self.cnn.predict(arr, verbose=0)

    # ── pre-processing helpers ────────────────────────────────────────────────
    @staticmethod
    def _normalize_lighting(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _grabcut_foreground(crop: np.ndarray) -> np.ndarray:
        try:
            h, w = crop.shape[:2]
            mask     = np.zeros((h, w), np.uint8)
            bgm      = np.zeros((1, 65), np.float64)
            fgm      = np.zeros((1, 65), np.float64)
            rect     = (5, 5, w - 10, h - 10)
            cv2.grabCut(crop, mask, rect, bgm, fgm, 3, cv2.GC_INIT_WITH_RECT)
            mask     = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
            kernel   = np.ones((5, 5), np.uint8)
            mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            soft     = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
            result   = crop.astype(np.float32)
            for c in range(3):
                result[:, :, c] *= soft
            return cv2.convertScaleAbs(result, alpha=1.1, beta=5)
        except Exception:
            return crop

    @staticmethod
    def _circle_crop(img: np.ndarray) -> np.ndarray:
        h, w   = img.shape[:2]
        mask   = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = int(min(w, h) * 0.48)
        cv2.circle(mask, center, radius, 255, -1)
        return cv2.bitwise_and(img, img, mask=mask)

    # ── core inference ────────────────────────────────────────────────────────
    def infer(self, img: np.ndarray) -> Optional[dict]:
        """
        Jalankan YOLO → crop → CNN untuk 1 gambar.
        Return dict hasil atau None kalau tidak ada jambu valid.
        """
        h, w     = img.shape[:2]
        img_area = h * w

        results = self.yolo(img, imgsz=640, conf=YOLO_CONF_THRESHOLD, verbose=False)

        candidates: list[tuple[float, np.ndarray]] = []   # (yolo_conf, box)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, yolo_conf in zip(boxes, confs):
                x1, y1, x2, y2 = box.astype(int)
                box_area = (x2 - x1) * (y2 - y1)

                # Buang deteksi yang terlalu kecil (bukan buah utama)
                if box_area / img_area < MIN_BOX_AREA_RATIO:
                    continue

                candidates.append((float(yolo_conf), box))

        if not candidates:
            return None

        # Urutkan descending confidence, ambil N terbaik
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:MAX_DETECTIONS_PER_IMAGE]

        detections: list[dict] = []

        for yolo_conf, box in candidates:
            x1, y1, x2, y2 = box.astype(int)

            # Padding kecil agar tepi buah tidak terpotong
            pad_x = int(0.04 * (x2 - x1))
            pad_y = int(0.04 * (y2 - y1))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Pre-process: grabcut → CLAHE → circle mask
            crop = self._grabcut_foreground(crop)
            crop = self._normalize_lighting(crop)
            crop = self._circle_crop(crop)

            crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            arr          = preprocess_input(
                np.expand_dims(crop_resized.astype(np.float32), 0)
            )

            pred       = self.cnn.predict(arr, verbose=0)[0]
            class_idx  = int(np.argmax(pred))
            cnn_conf   = float(pred[class_idx])
            label      = CLASS_NAMES[class_idx]

            if cnn_conf < CNN_CONF_THRESHOLD:
                # CNN ragu-ragu → tolak deteksi ini
                continue

            detections.append({
                "label":      label,
                "cnn_conf":   cnn_conf,
                "yolo_conf":  yolo_conf,
            })

        if not detections:
            return None

        # ── agregasi multi-deteksi (kalau MAX_DETECTIONS_PER_IMAGE > 1) ──
        labels = [d["label"]    for d in detections]
        confs  = [d["cnn_conf"] for d in detections]

        count = {c: labels.count(c) for c in CLASS_NAMES}
        majority = max(count, key=lambda c: (count[c], confs[labels.index(c)] if c in labels else 0))

        # Kalau count semua sama → pakai weighted score
        if len(set(count.values())) == 1:
            score_map = {"A": 3, "B": 2, "C": 1}
            weighted  = sum(score_map[l] * c for l, c in zip(labels, confs))
            avg       = weighted / sum(confs)
            majority  = "A" if avg >= 2.5 else "B" if avg >= 1.5 else "C"

        return {
            "grade":      majority,
            "confidence": float(sum(confs) / len(confs)),
            "n_detected": len(detections),
        }


# =============================================================================
# QUEUE WORKER
# Satu thread tunggal menguras antrian dan mengerjakan inference.
# =============================================================================
class InferenceQueue:
    """
    asyncio.Queue-based dispatcher.

    Item antrian: (images_np, loop, future)
      - images_np : list of (side_str, np.ndarray)
      - loop       : event loop asyncio milik request
      - future     : asyncio.Future untuk mengembalikan hasil
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine   = engine
        self._queue = stdlib_queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # Masukkan job ke queue (dipanggil dari async context)
    async def submit(self, images_np: list) -> dict:
        loop   = asyncio.get_event_loop()
        future = loop.create_future()

        try:
            self._queue.put_nowait((images_np, loop, future))
        except stdlib_queue.Full:
            raise HTTPException(
                status_code=503,
                detail="Server sedang sibuk, coba lagi dalam beberapa detik."
            )

        return await future   # tunggu sampai worker selesai

    # Worker — berjalan di thread terpisah, bukan event loop
    def _worker(self) -> None:
        # Ambil event loop dari thread utama tidak bisa; kita
        # pakai loop milik future (dikirim bersama job) untuk
        # set_result secara thread-safe.
        while True:
            # Blocking get — thread tidur kalau queue kosong
            item = self._queue.get()
            if item is None:
                break

            images_np, loop, future = item
            t0 = time.perf_counter()

            try:
                result = self._process(images_np)
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                elapsed = time.perf_counter() - t0
                result["_latency_ms"] = round(elapsed * 1000, 1)
                loop.call_soon_threadsafe(future.set_result, result)
                

    # ── proses 1 request (1–2 gambar) ────────────────────────────────────────
    def _process(self, images_np: list) -> dict:
        per_image     = []
        valid_results = []

        for side, img in images_np:
            if img is None:
                per_image.append({"side": side, "error": "decode_failed", "valid": False})
                continue

            res = self.engine.infer(img)

            if res is None:
                per_image.append({"side": side, "error": "no_jambu_detected", "valid": False})
                continue

            entry = {
                "side":       side,
                "grade":      res["grade"],
                "confidence": res["confidence"],
                "n_detected": res["n_detected"],
                "valid":      True,
            }
            per_image.append(entry)
            valid_results.append(entry)

        # ── tidak ada deteksi valid sama sekali ──────────────────────────────
        if not valid_results:
            return {
                "success":    False,
                "error_code": "no_jambu_detected",
                "message":    "Jambu tidak terdeteksi. Pastikan foto jelas dan berisi 1 buah jambu.",
                "ai_result":  per_image,
            }

        # ── agregasi lintas foto (front + back) ──────────────────────────────
        score_map   = {"A": 3, "B": 2, "C": 1}
        score       = sum(score_map[p["grade"]] for p in valid_results) / len(valid_results)
        final_grade = "A" if score >= 2.5 else "B" if score >= 1.5 else "C"

        conf_avg        = float(sum(p["confidence"] for p in valid_results) / len(valid_results))
        consistency     = "HIGH" if len({p["grade"] for p in valid_results}) == 1 else "LOW"
        defect_detected = any(p["grade"] == "C" for p in valid_results)

        return {
            "success":         True,
            "grade":           final_grade,
            "confidence_avg":  round(conf_avg, 4),
            "total_detected":  sum(p["n_detected"] for p in valid_results),
            "ai_result":       per_image,
            "consistency":     consistency,
            "defect_detected": defect_detected,
        }


# =============================================================================
# APP LIFECYCLE
# =============================================================================
engine_ref: Optional[InferenceEngine]  = None
queue_ref:  Optional[InferenceQueue]   = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine_ref, queue_ref
    engine_ref = InferenceEngine()
    queue_ref  = InferenceQueue(engine_ref)
    yield
    # cleanup (opsional)
    engine_ref = None
    queue_ref  = None


app = FastAPI(title="Jambu Grading API v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/grade")
async def grade(images: list[UploadFile] = File(...)):
    """
    Terima 1–2 foto jambu (front, back), kembalikan grade A/B/C.

    Request antri otomatis — tidak akan ditolak kecuali queue penuh (503).
    """
    if not images:
        raise HTTPException(status_code=400, detail="Tidak ada gambar yang dikirim.")

    if len(images) > 2:
        raise HTTPException(status_code=400, detail="Maksimal 2 gambar per request.")

    # Baca file I/O SEBELUM masuk queue — tidak perlu GPU
    images_np: list[tuple[str, Optional[np.ndarray]]] = []
    for idx, file in enumerate(images):
        contents = await file.read()
        side     = "front" if idx == 0 else "back"
        np_arr   = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        images_np.append((side, img))

    # Kirim ke queue, tunggu hasil
    result = await queue_ref.submit(images_np)
    return result


@app.post("/predict")
async def predict(images: list[UploadFile] = File(...)):
    """Alias untuk /grade (backward compat)."""
    return await grade(images)


@app.get("/health")
async def health():
    """Cek apakah service jalan dan berapa request yang antri."""
    return {
        "status":    "ok",
        "queue_size": queue_ref._queue.qsize() if queue_ref else -1,
    }