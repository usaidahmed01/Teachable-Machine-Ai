from typing import Annotated, List

from fastapi import APIRouter, File, Form, UploadFile

from app.ml_engine import predict_image, train_model
from app.utils.file_utils import (
    delete_class_dataset,
    get_dataset_summary,
    save_uploaded_images,
)

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


@router.post("/train")
def train():
    result = train_model()

    return {
        "message": "Training completed successfully.",
        "data": result
    }


@router.post("/predict")
def predict(
    file: Annotated[UploadFile, File(...)]
):
    result = predict_image(file)

    return {
        "message": "Prediction completed successfully.",
        "data": result
    }


@router.get("/dataset-summary")
def dataset_summary():
    result = get_dataset_summary()

    return {
        "message": "Dataset summary fetched successfully.",
        "data": result
    }


@router.delete("/delete-class")
def delete_class(
    class_name: Annotated[str, Form(...)]
):
    result = delete_class_dataset(class_name)

    return {
        "message": result["message"],
        "data": result
    }