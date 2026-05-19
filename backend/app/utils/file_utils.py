import re
import uuid
from pathlib import Path
from typing import List

from fastapi import UploadFile, HTTPException
from PIL import Image

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    DATASET_DIR,
    MAX_IMAGE_SIZE_BYTES,
)


def sanitize_class_name(class_name: str) -> str:
    cleaned_name = class_name.strip()

    if not cleaned_name:
        raise HTTPException(
            status_code=400,
            detail="Class name cannot be empty."
        )

    cleaned_name = re.sub(r"[^a-zA-Z0-9_-]", "_", cleaned_name)

    if len(cleaned_name) > 50:
        raise HTTPException(
            status_code=400,
            detail="Class name is too long. Maximum 50 characters allowed."
        )

    return cleaned_name


def validate_image_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image type '{extension}'. Allowed types are: jpg, jpeg, png, webp."
        )

    return extension


async def validate_image_size(file: UploadFile) -> bytes:
    contents = await file.read()

    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Image size is too large. Maximum allowed size is 10 MB."
        )

    await file.seek(0)
    return contents


def verify_image_file(file_path: Path) -> None:
    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception:
        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )


async def save_uploaded_images(class_name: str, files: List[UploadFile]) -> dict:
    safe_class_name = sanitize_class_name(class_name)

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one image."
        )

    class_folder = DATASET_DIR / safe_class_name
    class_folder.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for file in files:
        extension = validate_image_extension(file.filename)
        await validate_image_size(file)

        unique_filename = f"{uuid.uuid4()}{extension}"
        destination_path = class_folder / unique_filename

        with open(destination_path, "wb") as buffer:
            buffer.write(await file.read())

        verify_image_file(destination_path)

        saved_files.append(unique_filename)

    return {
        "class_name": safe_class_name,
        "saved_count": len(saved_files),
        "saved_files": saved_files,
        "folder_path": str(class_folder)
    }