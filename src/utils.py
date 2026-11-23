from collections import Counter, deque
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

STATIC_FACIAL_SIGNS = {"dad", "grandma", "grandpa", "mom"}
DYNAMIC_WAVE_SIGNS = {"bye", "hello"}

# Minimal 21-point hand skeleton connections for ultra-fast cv2.line drawing
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class SequenceBuffer:
    """32-frame sliding window buffer with 10-frame wrist/hand motion & variance tracking."""

    def __init__(
        self,
        sequence_length: int = 32,
        feature_dim: int = 98,
        motion_window: int = 10,
        motion_threshold: float = 0.020,
        dynamic_threshold: float = 0.075,
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.motion_window = motion_window
        self.motion_threshold = motion_threshold
        self.dynamic_threshold = dynamic_threshold
        self.buffer = deque(maxlen=sequence_length)

    def append(self, features: np.ndarray) -> None:
        self.buffer.append(features)

    def clear(self) -> None:
        self.buffer.clear()

    def is_ready(self) -> bool:
        return len(self.buffer) == self.sequence_length

    def compute_motion(self) -> Tuple[float, bool, bool]:
        """
        Calculates wrist and hand landmark variance + displacement over the last 10 frames.

        Returns:
            (motion_score, is_active_motion, is_high_dynamic_motion)
        """
        if len(self.buffer) < 3:
            return 0.0, False, False

        recent = np.asarray(list(self.buffer)[-self.motion_window :], dtype=np.float32)
        scores = []
        # Check left hand (0..41, wrist 0..1, pose wrist 92..93) and right hand (42..83, wrist 42..43, pose wrist 94..95)
        for sl, w_sl, mask_idx in (
            (slice(0, 42), slice(0, 2), 96),
            (slice(42, 84), slice(42, 44), 97),
        ):
            present = recent[:, mask_idx] > 0.5
            if int(present.sum()) >= 3:
                hand_seq = recent[present, sl]
                wrist_seq = recent[present, w_sl]
                hand_std = float(np.mean(np.std(hand_seq, axis=0)))
                wrist_disp = float(np.mean(np.linalg.norm(np.diff(wrist_seq, axis=0), axis=1)))
                scores.append(hand_std + wrist_disp)

        if not scores:
            return 0.0, False, False

        motion_score = max(scores)
        return (
            motion_score,
            motion_score >= self.motion_threshold,
            motion_score >= self.dynamic_threshold,
        )

    def get_tensor(self, device: torch.device) -> torch.Tensor:
        seq_array = np.asarray(self.buffer, dtype=np.float32)
        return torch.from_numpy(seq_array).unsqueeze(0).to(device)


class PredictionSmoother:
    """
    Confidence gating (>= 0.60), 'idle' suppression, dynamic-wave prioritization,
    and 4-step majority voting to eliminate single-frame flickering.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        idle_label: str = "idle",
        vote_window: int = 4,
    ):
        self.confidence_threshold = confidence_threshold
        self.idle_label = idle_label
        self.vote_window = vote_window
        self._vote_history: deque = deque(maxlen=vote_window)
        self.last_valid_label = "Waiting for gesture..."
        self.last_valid_confidence = 0.0

    def update(
        self,
        probabilities: np.ndarray,
        idx_to_label: Dict[int, str],
        is_active_motion: bool = True,
        is_high_dynamic_motion: bool = False,
        top_k: int = 3,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        probs = np.asarray(probabilities, dtype=np.float32).copy()

        # When wrist/hand oscillation is high (waving), boost dynamic signs over static facial signs
        if is_high_dynamic_motion:
            for idx, lbl in idx_to_label.items():
                low = lbl.lower()
                if idx < len(probs):
                    if low in STATIC_FACIAL_SIGNS:
                        probs[idx] *= 0.40
                    elif low in DYNAMIC_WAVE_SIGNS:
                        probs[idx] *= 1.45
            total = float(probs.sum())
            if total > 1e-6:
                probs /= total

        top_indices = probs.argsort()[-top_k:][::-1]
        top_preds = [(idx_to_label.get(int(i), self.idle_label), float(probs[int(i)])) for i in top_indices]

        if not is_active_motion:
            self._vote_history.append(self.idle_label)
            return self.last_valid_label, self.last_valid_confidence, top_preds

        best_label, best_conf = top_preds[0]
        if best_label.lower() == self.idle_label.lower() or best_conf < self.confidence_threshold:
            self._vote_history.append(self.idle_label)
            return self.last_valid_label, self.last_valid_confidence, top_preds

        self._vote_history.append(best_label)
        majority_label, count = Counter(self._vote_history).most_common(1)[0]
        min_votes = max(2, (self.vote_window // 2) + 1)  # 3 of 4 votes
        if majority_label != self.idle_label and count >= min_votes:
            self.last_valid_label = majority_label
            self.last_valid_confidence = best_conf

        return self.last_valid_label, self.last_valid_confidence, top_preds

    def process(self, predicted_label: str, confidence: float) -> Tuple[str, float]:
        if predicted_label != self.idle_label and confidence >= self.confidence_threshold:
            self._vote_history.append(predicted_label)
            majority_label, count = Counter(self._vote_history).most_common(1)[0]
            if majority_label != self.idle_label and count >= max(2, (self.vote_window // 2) + 1):
                self.last_valid_label = majority_label
                self.last_valid_confidence = confidence
        else:
            self._vote_history.append(self.idle_label)
        return self.last_valid_label, self.last_valid_confidence

    def reset(self) -> None:
        self._vote_history.clear()
        self.last_valid_label = "Waiting for gesture..."
        self.last_valid_confidence = 0.0


def draw_hand_connections_fast(frame: np.ndarray, results) -> None:
    """Minimal, zero-allocation hand-only skeleton renderer for maximum FPS."""
    h, w = frame.shape[:2]
    for container, color in (
        (results.left_hand_landmarks, (255, 190, 70)),
        (results.right_hand_landmarks, (0, 235, 170)),
    ):
        lms = getattr(container, "landmark", container) if container else None
        if not lms or len(lms) != 21:
            continue
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], (230, 230, 230), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, color, -1)


def draw_spatial_guides(frame: np.ndarray, results) -> Tuple[np.ndarray, str]:
    """Lightweight helper retained for compatibility."""
    draw_hand_connections_fast(frame, results)
    return frame, "OK"


def draw_hud(
    frame: np.ndarray,
    text: str,
    confidence: float,
    fps: Optional[float] = None,
    motion_active: bool = False,
    status_line: Optional[str] = None,
) -> np.ndarray:
    """Fast ROI-only HUD banner (avoids copying the full frame)."""
    h, w = frame.shape[:2]
    banner_h = min(84, h)
    roi = frame[0:banner_h, 0:w]
    dark = np.full_like(roi, (18, 22, 28))
    cv2.addWeighted(dark, 0.82, roi, 0.18, 0, roi)
    cv2.line(frame, (0, banner_h - 1), (w, banner_h - 1), (38, 198, 218), 2)

    if confidence > 0.0:
        display_text = f"Sign: {text.upper()} ({confidence * 100:.1f}%)"
        text_color = (0, 255, 180)
    else:
        display_text = f"Sign: {text}"
        text_color = (205, 205, 205)

    cv2.putText(frame, display_text, (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.80, text_color, 2, cv2.LINE_AA)

    motion_label = "MOTION: ACTIVE" if motion_active else "MOTION: STATIC"
    motion_color = (0, 235, 150) if motion_active else (140, 155, 170)
    cv2.putText(frame, motion_label, (w - 270, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.54, motion_color, 2, cv2.LINE_AA)

    if fps is not None:
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 98, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (205, 205, 205), 2, cv2.LINE_AA)

    if status_line:
        cv2.putText(frame, status_line, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 195, 210), 1, cv2.LINE_AA)

    return frame
