import argparse
import json
import os
import threading
import time
import urllib.request
import cv2
import mediapipe as mp
import torch

from src.model_arch import load_trained_model
from src.feature_extractor import AR_MODES, extract_landmarks
from src.utils import SequenceBuffer, PredictionSmoother, draw_hand_connections_fast, draw_hud

HOLISTIC_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/1/holistic_landmarker.task"
)
HOLISTIC_TASK_PATH = os.path.join("models", "holistic_landmarker.task")


def ensure_holistic_task_model(model_path: str = HOLISTIC_TASK_PATH) -> str:
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[INFO] Downloading MediaPipe Holistic task bundle to {model_path}...")
        urllib.request.urlretrieve(HOLISTIC_TASK_URL, model_path)
        print("[INFO] Download complete.")
    return model_path


class ThreadedCamera:
    """Non-blocking background frame reader to keep webcam capture responsive."""

    def __init__(self, camera_index: int):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(camera_index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            fallback_index = 1 if camera_index == 0 else 0
            print(f"[WARNING] Camera index {camera_index} inaccessible. Switching to index {fallback_index}...")
            self.cap = cv2.VideoCapture(fallback_index, backend)
        if not self.cap.isOpened():
            raise RuntimeError("No accessible webcam found. Close other camera apps and check your connection.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Webcam configured: {actual_w}x{actual_h}")

        self._lock = threading.Lock()
        self._ret, self._frame = self.cap.read()
        self._stopped = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        with self._lock:
            if not self._ret or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SignFlow - Real-Time ASL Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold (default: 0.55)")
    parser.add_argument("--infer-every", type=int, default=3, help="Run BiGRU every N frames (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.path.join("models", "model.pt")
    label_map_path = os.path.join("models", "label_map.json")

    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"Label map not found at {label_map_path}. Ensure label_map.json is in models/.")

    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    idx_to_label = {int(v): k for k, v in label_map.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"[INFO] Initializing runtime on device: {device}")

    model = load_trained_model(model_path, num_classes=len(label_map), device=device)
    model_num_classes = model.fc[1].out_features
    if model_num_classes > len(idx_to_label) and "idle" not in label_map:
        idx_to_label[len(idx_to_label)] = "idle"
    print(f"[INFO] Model loaded successfully ({model_num_classes} classes).")

    buffer = SequenceBuffer(sequence_length=32, feature_dim=98)
    smoother = PredictionSmoother(confidence_threshold=args.threshold, idle_label="idle", vote_window=4)

    cam = ThreadedCamera(args.camera)

    # Default to 3:4 (GISLR normalized), toggleable via 'a' (3:4 -> 9:16 -> Raw)
    ar_idx = 0
    prev_time = time.perf_counter()
    fps_ema = 0.0
    last_ts_ms = -1
    frame_count = 0

    current_label = "Waiting for gesture..."
    current_conf = 0.0
    top_str = ""

    use_tasks_api = not hasattr(mp, "solutions")
    if use_tasks_api:
        task_model_path = ensure_holistic_task_model()
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=task_model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
            min_hand_landmarks_confidence=0.35,
            output_face_blendshapes=False,
            output_pose_segmentation_masks=False,
        )
        holistic_ctx = vision.HolisticLandmarker.create_from_options(options)
    else:
        mp_holistic = mp.solutions.holistic
        holistic_ctx = mp_holistic.Holistic(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
            smooth_landmarks=True,
        )

    print("[INFO] Webcam feed operational. Keep head & shoulders in frame! Controls: [a] AR | [c] Clear | [q] Quit")

    try:
        with holistic_ctx as holistic:
            while True:
                ret, frame = cam.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                frame_count += 1
                # Full 640x480 RGB preserves reliable hand/pose detection even in dim indoor light
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if use_tasks_api:
                    ts_ms = int(time.perf_counter() * 1000)
                    if ts_ms <= last_ts_ms:
                        ts_ms = last_ts_ms + 1
                    last_ts_ms = ts_ms
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    results = holistic.detect_for_video(mp_image, ts_ms)
                else:
                    rgb_frame.flags.writeable = False
                    results = holistic.process(rgb_frame)

                # Minimal drawing overhead: render hand connections only
                draw_hand_connections_fast(frame, results)

                # Mirror for natural display (HUD drawn after flip)
                frame = cv2.flip(frame, 1)

                ar_mode = AR_MODES[ar_idx]
                features = extract_landmarks(results, ar_mode=ar_mode, frame_shape=frame.shape[:2])
                buffer.append(features)

                pose_lms = getattr(results.pose_landmarks, "landmark", results.pose_landmarks) if results.pose_landmarks else None
                pose_ok = bool(pose_lms and len(pose_lms) > 16)
                motion_score, is_active_motion, is_high_dynamic = buffer.compute_motion()
                hand_tag = f"{'L' if features[96] > 0.5 else '-'}{'R' if features[97] > 0.5 else '-'}"

                if not pose_ok:
                    status_line = f"[{hand_tag}|NO BODY/SHOULDERS] Show head & shoulders in camera!"
                elif buffer.is_ready():
                    if frame_count % max(1, args.infer_every) == 0:
                        input_tensor = buffer.get_tensor(device)
                        with torch.inference_mode():
                            logits = model(input_tensor)
                            probs_np = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                        current_label, current_conf, top3 = smoother.update(
                            probs_np,
                            idx_to_label,
                            is_active_motion=is_active_motion,
                            is_high_dynamic_motion=is_high_dynamic,
                            top_k=3,
                        )
                        top_str = " ".join(f"{i+1}:{lbl}({p*100:.0f}%)" for i, (lbl, p) in enumerate(top3))
                    status_line = f"[{hand_tag}|AR:{ar_mode}] {top_str} | [A] AR [C] Clear [Q] Quit"
                else:
                    status_line = (
                        f"[{hand_tag}|AR:{ar_mode}] Buffering: {len(buffer.buffer)}/{buffer.sequence_length} "
                        "| [A] AR [C] Clear [Q] Quit"
                    )

                now = time.perf_counter()
                inst_fps = 1.0 / max(now - prev_time, 1e-6)
                fps_ema = inst_fps if fps_ema == 0.0 else (0.2 * inst_fps + 0.8 * fps_ema)
                prev_time = now

                frame = draw_hud(
                    frame,
                    current_label,
                    current_conf,
                    fps=fps_ema,
                    motion_active=is_active_motion,
                    status_line=status_line,
                )

                cv2.imshow("SignFlow - Real-Time ASL Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    buffer.clear()
                    smoother.reset()
                    current_label = "Waiting for gesture..."
                    current_conf = 0.0
                    top_str = ""
                    print("[INFO] 32-frame buffer cleared.")
                elif key == ord("a"):
                    ar_idx = (ar_idx + 1) % len(AR_MODES)
                    buffer.clear()
                    smoother.reset()
                    current_label = "Waiting for gesture..."
                    current_conf = 0.0
                    top_str = ""
                    print(f"[INFO] Switched Aspect Ratio Mode to: {AR_MODES[ar_idx]}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
