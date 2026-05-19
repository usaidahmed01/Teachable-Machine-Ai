from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    DATASET_DIR,
    IMAGE_SIZE,
    MIN_CLASSES_REQUIRED,
    MIN_IMAGES_PER_CLASS,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)


def get_device() -> torch.device:
    """
    Select GPU if available, otherwise use CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocessing_transform():
    """
    Create the exact image preprocessing pipeline used before feature extraction.
    This same transform must also be used during prediction.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_feature_extractor(device: torch.device):
    """
    Load MobileNetV3 Small pre-trained on ImageNet.
    Remove the final classification layer so the model works as a feature extractor.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)

    return model


def get_class_folders() -> List[Path]:
    """
    Return all valid class folders inside the dataset directory.
    """
    if not DATASET_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Dataset folder does not exist. Please upload images first."
        )

    class_folders = [
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    ]

    if len(class_folders) < MIN_CLASSES_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_CLASSES_REQUIRED} classes are required for training."
        )

    return class_folders


def get_image_paths_from_class_folder(class_folder: Path) -> List[Path]:
    """
    Return valid image paths from one class folder.
    """
    image_paths = [
        image_path for image_path in class_folder.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]

    return image_paths


def validate_dataset(class_folders: List[Path]) -> Dict[str, int]:
    """
    Check that every class has enough images.
    Return class distribution.
    """
    class_distribution = {}

    for class_folder in class_folders:
        image_paths = get_image_paths_from_class_folder(class_folder)
        image_count = len(image_paths)

        if image_count < MIN_IMAGES_PER_CLASS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Class '{class_folder.name}' has only {image_count} image(s). "
                    f"Minimum {MIN_IMAGES_PER_CLASS} images per class are required."
                )
            )

        class_distribution[class_folder.name] = image_count

    return class_distribution


def load_image_as_tensor(image_path: Path, transform) -> torch.Tensor:
    """
    Load one image, convert to RGB, apply preprocessing, and return a tensor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
        return image_tensor

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image found: {image_path.name}"
        )


def extract_features_for_image(
    image_tensor: torch.Tensor,
    feature_extractor,
    device: torch.device
) -> np.ndarray:
    """
    Pass one image tensor through MobileNetV3 and return extracted features.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    return features.cpu().numpy().flatten()


def build_feature_dataset(
    class_folders: List[Path],
    feature_extractor,
    transform,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert all dataset images into feature vectors and labels.
    """
    features = []
    labels = []

    for class_folder in class_folders:
        class_name = class_folder.name
        image_paths = get_image_paths_from_class_folder(class_folder)

        for image_path in image_paths:
            image_tensor = load_image_as_tensor(image_path, transform)
            image_features = extract_features_for_image(
                image_tensor=image_tensor,
                feature_extractor=feature_extractor,
                device=device
            )

            features.append(image_features)
            labels.append(class_name)

    return np.array(features), np.array(labels)


def train_model() -> Dict:
    """
    Main training function.
    It scans the dataset, extracts MobileNetV3 features,
    trains Logistic Regression, saves model.pkl, and returns metrics.
    """
    class_folders = get_class_folders()
    class_distribution = validate_dataset(class_folders)

    device = get_device()
    transform = get_preprocessing_transform()
    feature_extractor = load_feature_extractor(device)

    features, labels = build_feature_dataset(
        class_folders=class_folders,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device
    )

    if len(features) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid images found for training."
        )

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=encoded_labels
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            encoded_labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

    classifier = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "classifier": classifier,
        "label_encoder": label_encoder,
        "class_names": list(label_encoder.classes_),
        "class_distribution": class_distribution,
        "image_size": IMAGE_SIZE,
        "feature_extractor": "mobilenet_v3_small",
        "accuracy": float(accuracy),
        "total_images": int(len(labels)),
    }

    joblib.dump(model_package, MODEL_PATH)

    return {
        "message": "Model trained successfully.",
        "model_path": str(MODEL_PATH),
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy) * 100, 2),
        "classes": list(label_encoder.classes_),
        "class_distribution": class_distribution,
        "total_images": int(len(labels)),
        "device": str(device),
    }