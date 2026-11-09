
import cv2
import numpy as np
from PIL import Image


def enhance_image(image, mode="text_safe"):
    """
    Conservative 2x image upscaling.
    No OCR or text rewriting.
    """

    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL image")

    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if mode == "text_safe":
        # Upscale first using Lanczos interpolation.
        h, w = bgr.shape[:2]
        enlarged = cv2.resize(
            bgr,
            (w * 2, h * 2),
            interpolation=cv2.INTER_LANCZOS4
        )

        # Very mild luminance-only sharpening.
        lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)

        blurred = cv2.GaussianBlur(
            light, (0, 0), sigmaX=0.65
        )

        light = cv2.addWeighted(
            light, 1.12,
            blurred, -0.12, 0
        )

        processed = cv2.cvtColor(
            cv2.merge((light, a, b)),
            cv2.COLOR_LAB2BGR
        )

    elif mode == "photo":
        h, w = bgr.shape[:2]
        processed = cv2.resize(
            bgr,
            (w * 2, h * 2),
            interpolation=cv2.INTER_LANCZOS4
        )

        processed = cv2.detailEnhance(
            processed,
            sigma_s=10,
            sigma_r=0.12
        )

    else:
        raise ValueError("Unsupported enhancement mode")

    result = cv2.cvtColor(
        processed, cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(result)
