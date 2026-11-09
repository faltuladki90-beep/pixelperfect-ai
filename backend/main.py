from io import BytesIO

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from enhancement import enhance_image

app = FastAPI(title="PixelPerfect AI Backend")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "PixelPerfect AI backend is running"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    mode: str = Form("text_safe")
):
    if mode not in ("text_safe", "photo"):
        raise HTTPException(status_code=400, detail="Invalid enhancement mode")

    allowed_types = ("image/png", "image/jpeg", "image/webp", "image/bmp")
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Upload a PNG, JPG, WebP or BMP image")

    try:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Maximum upload size is 20 MB")

        with Image.open(BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")

        if image.width * image.height > MAX_PIXELS:
            raise HTTPException(status_code=413, detail="Image exceeds 40 megapixels")

        result = enhance_image(image, mode=mode)
        output = BytesIO()
        result.save(output, format="PNG")
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="pixelperfect-enhanced.png"'}
        )

    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError
    ) as exc:
        raise HTTPException(status_code=400, detail="Image could not be processed") from exc
    finally:
        await file.close()
