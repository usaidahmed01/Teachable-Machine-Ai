from typing import Annotated, List

from fastapi import APIRouter, File, Form, UploadFile

from app.utils.file_utils import save_uploaded_images

router = APIRouter()


@router.get("/")
def root():
    return {
        "message": "Teachable Machine AI Backend is running.",
        "status": "success"
    }


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FastAPI ML Backend"
    }


@router.post("/upload-sample")
async def upload_sample(
    class_name: Annotated[str, Form(...)],
    files: Annotated[List[UploadFile], File(...)]
):
    result = await save_uploaded_images(class_name, files)

    return {
        "message": "Images uploaded successfully.",
        "data": result
    }