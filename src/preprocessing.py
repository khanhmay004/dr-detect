"""
https://kaggle.com/competitions/diabetic-retinopathy-detection/discussion/15801
"""

import cv2
import numpy as np
from typing import Tuple
from config import IMAGE_SIZE, CROP_MARGIN, BEN_GRAHAM_SIGMA_SCALE

#Step 1: Circular crop (remove black borders)

def find_retina_circle(image: np.ndarray) -> Tuple[int, int, int]:
    """Detect the retina's bounding circle via thresholding + contour.
    Returns:
        (center_x, center_y, radius)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        h, w = image.shape[:2]
        return (w // 2, h // 2, min(w, h) // 2)

    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return (int(x), int(y), int(radius))


def circular_crop(
    image: np.ndarray,
    target_size: int = IMAGE_SIZE,
    margin: float = CROP_MARGIN,
) -> np.ndarray:
    """Cắt ảnh theo hộp giới hạn của võng mạc + lề, resize"""
    h, w = image.shape[:2]
    cx, cy, r = find_retina_circle(image)

    # Add small margin to avoid clipping the retina edge
    r = int(r * (1 + margin))

    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)

    cropped = image[y1:y2, x1:x2]
    if cropped.shape[0] < 10 or cropped.shape[1] < 10:
        cropped = image

    resized = cv2.resize(
        cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4
    )
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    center = target_size // 2
    cv2.circle(mask, (center, center), center, 255, -1)
    resized[mask == 0] = 0

    return resized


#Step 2: local color normalization (gaussian blur weighted addition)
def local_color_normalization(
    image: np.ndarray,
    sigma_scale: int = BEN_GRAHAM_SIGMA_SCALE,
) -> np.ndarray:
    h, w = image.shape[:2]
    sigma = w / sigma_scale

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    result = cv2.addWeighted(image, 4, blurred, -4, 128)

    return result


# Full pipeline
def ben_graham_preprocess(
    image: np.ndarray,
    target_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """``(target_size, target_size, 3)``"""
    cropped = circular_crop(image, target_size)
    result = local_color_normalization(cropped)
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        result = ben_graham_preprocess(img, IMAGE_SIZE)
        print(f"Input: {img.shape} -> Output: {result.shape}")
