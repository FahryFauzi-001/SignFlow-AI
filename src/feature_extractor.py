import math
from typing import Optional, Tuple
import numpy as np

AR_MODES = ("3:4", "9:16", "Raw")


def _get_landmarks(landmark_container):
    """Supports both MediaPipe Tasks (list[NormalizedLandmark]) and legacy Solutions (.landmark)."""
    if not landmark_container:
        return None
    return getattr(landmark_container, "landmark", landmark_container)


def _y_multiplier(ar_mode: str, frame_shape: Optional[Tuple[int, int]] = None) -> float:
    """
    Converts landscape webcam MediaPipe y-coordinates into portrait GISLR proportions.
    On a 640x480 (4:3) webcam:
    - '3:4'  -> (480/640) * (3/4)  = 0.5625
    - '9:16' -> (480/640) * (9/16) = 0.4219
    - 'Raw'  -> 1.0
    """
    if ar_mode == "Raw":
        return 1.0
    h, w = frame_shape if (frame_shape is not None and frame_shape[1] > 0) else (480, 640)
    cam_hw = h / float(w)
    if ar_mode == "3:4":
        return cam_hw * (3.0 / 4.0)
    if ar_mode == "9:16":
        return cam_hw * (9.0 / 16.0)
    return 1.0


def extract_landmarks(
    results,
    ar_mode: str = "3:4",
    frame_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Extracts 98 body-centered, scale-invariant features matching the SignFlow BiGRU checkpoint:
    - [0..41]  Left Hand landmarks: 21 points (x_norm, y_norm) = 42 features
    - [42..83] Right Hand landmarks: 21 points (x_norm, y_norm) = 42 features
    - [84..95] Upper-body Pose landmarks [11, 12, 13, 14, 15, 16] (x_norm, y_norm) = 12 features
    - [96..97] Hand presence flags: [has_left_hand, has_right_hand] = 2 features
    Total: 42 + 42 + 12 + 2 = 98 float32 features.
    """
    pose = _get_landmarks(results.pose_landmarks)
    left_hand = _get_landmarks(results.left_hand_landmarks)
    right_hand = _get_landmarks(results.right_hand_landmarks)

    y_mult = _y_multiplier(ar_mode, frame_shape)

    # Mid-shoulder centering & shoulder-width scaling
    if pose and len(pose) > 16:
        l_sh, r_sh = pose[11], pose[12]
        l_y = l_sh.y * y_mult
        r_y = r_sh.y * y_mult
        mid_x = (l_sh.x + r_sh.x) * 0.5
        mid_y = (l_y + r_y) * 0.5
        shoulder_dist = max(0.15, math.hypot(r_sh.x - l_sh.x, r_y - l_y))
    else:
        mid_x = 0.5
        mid_y = 0.5 * y_mult
        shoulder_dist = 0.25

    inv_dist = 1.0 / shoulder_dist
    out = np.zeros(98, dtype=np.float32)

    # 1. Left Hand landmarks (42 values: indices 0..41)
    if left_hand and len(left_hand) == 21:
        out[96] = 1.0
        for i, lm in enumerate(left_hand):
            out[2 * i] = max(-5.0, min(5.0, (lm.x - mid_x) * inv_dist))
            out[2 * i + 1] = max(-5.0, min(5.0, (lm.y * y_mult - mid_y) * inv_dist))

    # 2. Right Hand landmarks (42 values: indices 42..83)
    if right_hand and len(right_hand) == 21:
        out[97] = 1.0
        for i, lm in enumerate(right_hand):
            out[42 + 2 * i] = max(-5.0, min(5.0, (lm.x - mid_x) * inv_dist))
            out[42 + 2 * i + 1] = max(-5.0, min(5.0, (lm.y * y_mult - mid_y) * inv_dist))

    # 3. Upper-body Pose landmarks [11..16] (12 values: indices 84..95)
    if pose and len(pose) > 16:
        for i, idx in enumerate((11, 12, 13, 14, 15, 16)):
            lm = pose[idx]
            out[84 + 2 * i] = max(-5.0, min(5.0, (lm.x - mid_x) * inv_dist))
            out[84 + 2 * i + 1] = max(-5.0, min(5.0, (lm.y * y_mult - mid_y) * inv_dist))

    return out
