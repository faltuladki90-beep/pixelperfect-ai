
import cv2
import numpy as np
from PIL import Image


def enhance_image(image, mode="text_safe"):
    """
    Conservative image enhancement.
    Does not generate or rewrite text.
    """

    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL image")

    original = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

    if mode == "text_safe":
        # Enhance luminance while retaining color information
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.5,
            tileGridSize=(8, 8)
        )
        l = clahe.apply(l)

        processed = cv2.cvtColor(
            cv2.merge((l, a, b)),
            cv2.COLOR_LAB2BGR
        )

        # Mild unsharp masking to improve edge definition
        blurred = cv2.GaussianBlur(
            processed, (0, 0), 0.7
        )
        processed = cv2.addWeighted(
            processed, 1.18,
            blurred, -0.18, 0
        )

    elif mode == "photo":
        processed = cv2.detailEnhance(
            bgr, sigma_s=10, sigma_r=0.12
        )

    else:
        raise ValueError("Unsupported enhancement mode")

    result = cv2.cvtColor(
        processed, cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(result)
