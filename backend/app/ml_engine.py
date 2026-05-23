from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import models, transforms

from app.core.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    MIN_CLASSES_REQUIRED,
    MIN_IMAGES_PER_CLASS,
    RANDOM_STATE,
    TEST_SIZE,
)
from app.utils.session_utils import get_session_dataset_dir, get_session_model_path


MODEL_CACHE: Dict[str, Dict] = {}
FEATURE_EXTRACTOR_CACHE: Dict[str, object] = {}

CONFIDENCE_THRESHOLD = 60.0
RECOMMENDED_IMAGES_PER_CLASS = 5
CLASS_IMBALANCE_RATIO = 3.0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preprocessing_transform():
    """
    Create the exact image preprocessing pipeline used before feature extraction.

    The same transform must be used during both training and prediction.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_feature_extractor(device: torch.device):
    """
    Load MobileNetV3 Small pre-trained on ImageNet and cache it.

    MobileNetV3 is used as a feature extractor. The final classifier layer is
    removed, so the model returns feature vectors instead of ImageNet classes.
    """
    cache_key = str(device)

    if cache_key in FEATURE_EXTRACTOR_CACHE:
        return FEATURE_EXTRACTOR_CACHE[cache_key]

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    model.classifier = torch.nn.Identity()
    model.eval()
    model.to(device)

    FEATURE_EXTRACTOR_CACHE[cache_key] = model

    return model


def get_class_folders(session_id: str) -> List[Path]:
    """
    Return all class folders inside the current session dataset directory.
    """
    dataset_dir = get_session_dataset_dir(session_id)

    if not dataset_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="Dataset folder does not exist. Please upload images first.",
        )

    class_folders = [folder for folder in dataset_dir.iterdir() if folder.is_dir()]

    if len(class_folders) < MIN_CLASSES_REQUIRED:
        raise HTTPException(
            status_code=400,
            detail=f"At least {MIN_CLASSES_REQUIRED} classes are required for training.",
        )

    return class_folders


def get_image_paths_from_class_folder(class_folder: Path) -> List[Path]:
    """
    Return valid image paths from one class folder.
    """
    return [
        image_path
        for image_path in class_folder.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]


def validate_dataset(class_folders: List[Path]) -> Dict[str, int]:
    """
    Hard dataset validation.

    Training stops if any class has fewer than MIN_IMAGES_PER_CLASS images.
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
                ),
            )

        class_distribution[class_folder.name] = image_count

    return class_distribution


def generate_dataset_quality_warnings(class_distribution: Dict[str, int]) -> List[str]:
    """
    Soft dataset quality warnings.

    These warnings do not block training. They explain why the model may perform
    weakly if samples are too few or class distribution is imbalanced.
    """
    warnings = []

    if not class_distribution:
        return warnings

    for class_name, image_count in class_distribution.items():
        if image_count < RECOMMENDED_IMAGES_PER_CLASS:
            warnings.append(
                f"Class '{class_name}' has only {image_count} image(s). "
                f"For better accuracy, add at least {RECOMMENDED_IMAGES_PER_CLASS} images."
            )

    min_images = min(class_distribution.values())
    max_images = max(class_distribution.values())

    if min_images > 0 and max_images / min_images >= CLASS_IMBALANCE_RATIO:
        warnings.append(
            "Class imbalance detected. One class has significantly more images "
            "than another. The model may become biased, although "
            "class_weight='balanced' helps reduce this issue."
        )

    return warnings


def load_image_as_tensor(image_path: Path, transform) -> torch.Tensor:
    """
    Load one image, convert it to RGB, apply preprocessing, and return tensor.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image)

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image found: {image_path.name}",
        )


def extract_features_for_image(
    image_tensor: torch.Tensor,
    feature_extractor,
    device: torch.device,
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
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Convert all dataset images into feature vectors, labels, and references.

    Image references are used later to create sample prediction examples.
    """
    features = []
    labels = []
    image_references = []

    for class_folder in class_folders:
        class_name = class_folder.name
        image_paths = get_image_paths_from_class_folder(class_folder)

        for image_path in image_paths:
            image_tensor = load_image_as_tensor(image_path, transform)

            image_features = extract_features_for_image(
                image_tensor=image_tensor,
                feature_extractor=feature_extractor,
                device=device,
            )

            features.append(image_features)
            labels.append(class_name)
            image_references.append(
                {
                    "file_name": image_path.name,
                    "class_name": class_name,
                    "path": str(image_path),
                }
            )

    return np.array(features), np.array(labels), image_references


def calculate_per_class_accuracy(
    confusion_matrix_values: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """
    Calculate accuracy for each class.

    Per-class accuracy:
    correct predictions for a class / total actual images of that class
    """
    per_class_accuracy = {}

    for index, class_name in enumerate(class_names):
        total_actual = int(confusion_matrix_values[index].sum())
        correct_predictions = int(confusion_matrix_values[index][index])

        if total_actual == 0:
            per_class_accuracy[class_name] = 0.0
        else:
            per_class_accuracy[class_name] = round(
                (correct_predictions / total_actual) * 100,
                2,
            )

    return per_class_accuracy


def build_sample_prediction_examples(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    prediction_probabilities: np.ndarray,
    test_image_references: List[Dict],
    label_encoder: LabelEncoder,
    max_examples: int = 8,
) -> List[Dict]:
    """
    Build a small list of actual-vs-predicted test examples.
    """
    examples = []
    total_examples = min(len(y_test), max_examples)

    for index in range(total_examples):
        actual_encoded = int(y_test[index])
        predicted_encoded = int(y_pred[index])

        actual_class = label_encoder.inverse_transform([actual_encoded])[0]
        predicted_class = label_encoder.inverse_transform([predicted_encoded])[0]

        confidence = round(
            float(np.max(prediction_probabilities[index])) * 100,
            2,
        )

        examples.append(
            {
                "file_name": test_image_references[index]["file_name"],
                "actual_class": actual_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "is_correct": actual_class == predicted_class,
            }
        )

    return examples


def build_evaluation_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    prediction_probabilities: np.ndarray,
    label_encoder: LabelEncoder,
    test_image_references: List[Dict],
) -> Dict:
    """
    Build complete model evaluation metrics.

    Includes confusion matrix, precision, recall, F1-score, per-class accuracy,
    macro/weighted averages, and sample prediction examples.
    """
    class_names = list(label_encoder.classes_)
    labels = list(range(len(class_names)))

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = {}

    for index, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            "precision": round(float(precision[index]) * 100, 2),
            "recall": round(float(recall[index]) * 100, 2),
            "f1_score": round(float(f1[index]) * 100, 2),
            "support": int(support[index]),
        }

    per_class_accuracy = calculate_per_class_accuracy(
        confusion_matrix_values=cm,
        class_names=class_names,
    )

    sample_prediction_examples = build_sample_prediction_examples(
        y_test=y_test,
        y_pred=y_pred,
        prediction_probabilities=prediction_probabilities,
        test_image_references=test_image_references,
        label_encoder=label_encoder,
    )

    return {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names,
        "per_class_metrics": per_class_metrics,
        "per_class_accuracy": per_class_accuracy,
        "macro_average": {
            "precision": round(float(report["macro avg"]["precision"]) * 100, 2),
            "recall": round(float(report["macro avg"]["recall"]) * 100, 2),
            "f1_score": round(float(report["macro avg"]["f1-score"]) * 100, 2),
        },
        "weighted_average": {
            "precision": round(float(report["weighted avg"]["precision"]) * 100, 2),
            "recall": round(float(report["weighted avg"]["recall"]) * 100, 2),
            "f1_score": round(float(report["weighted avg"]["f1-score"]) * 100, 2),
        },
        "sample_prediction_examples": sample_prediction_examples,
    }


def train_model(session_id: str) -> Dict:
    """
    Train the model for the current session.

    Flow:
    dataset validation -> feature extraction -> label encoding -> train/test split
    -> balanced Logistic Regression -> evaluation metrics -> save model package
    -> cache model in memory
    """
    class_folders = get_class_folders(session_id)
    class_distribution = validate_dataset(class_folders)
    dataset_warnings = generate_dataset_quality_warnings(class_distribution)

    device = get_device()
    transform = get_preprocessing_transform()
    feature_extractor = load_feature_extractor(device)

    features, labels, image_references = build_feature_dataset(
        class_folders=class_folders,
        feature_extractor=feature_extractor,
        transform=transform,
        device=device,
    )

    if len(features) == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid images found for training.",
        )

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_refs,
            test_refs,
        ) = train_test_split(
            features,
            encoded_labels,
            image_references,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=encoded_labels,
        )
    except ValueError:
        (
            X_train,
            X_test,
            y_train,
            y_test,
            train_refs,
            test_refs,
        ) = train_test_split(
            features,
            encoded_labels,
            image_references,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    prediction_probabilities = classifier.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    evaluation_metrics = build_evaluation_metrics(
        y_test=y_test,
        y_pred=y_pred,
        prediction_probabilities=prediction_probabilities,
        label_encoder=label_encoder,
        test_image_references=test_refs,
    )

    session_model_path = get_session_model_path(session_id)
    session_model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "classifier": classifier,
        "label_encoder": label_encoder,
        "class_names": list(label_encoder.classes_),
        "class_distribution": class_distribution,
        "dataset_warnings": dataset_warnings,
        "evaluation_metrics": evaluation_metrics,
        "image_size": IMAGE_SIZE,
        "feature_extractor": "mobilenet_v3_small",
        "accuracy": float(accuracy),
        "accuracy_percentage": round(float(accuracy) * 100, 2),
        "total_images": int(len(labels)),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    joblib.dump(model_package, session_model_path)

    MODEL_CACHE[session_id] = model_package

    return {
        "message": "Model trained successfully.",
        "model_path": str(session_model_path),
        "accuracy": round(float(accuracy), 4),
        "accuracy_percentage": round(float(accuracy) * 100, 2),
        "classes": list(label_encoder.classes_),
        "class_distribution": class_distribution,
        "dataset_warnings": dataset_warnings,
        "evaluation_metrics": evaluation_metrics,
        "total_images": int(len(labels)),
        "device": str(device),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


def load_trained_model_package(session_id: str) -> Dict:
    """
    Load trained model package from cache or disk.
    """
    if session_id in MODEL_CACHE:
        return MODEL_CACHE[session_id]

    session_model_path = get_session_model_path(session_id)

    if not session_model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No trained model found for this session. Please train a model first.",
        )

    try:
        model_package = joblib.load(session_model_path)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to load trained model. Please retrain the model.",
        )

    MODEL_CACHE[session_id] = model_package

    return model_package


def clear_model_cache(session_id: str) -> None:
    """
    Clear cached model for a session.

    Use this when the user resets the project.
    """
    if session_id in MODEL_CACHE:
        del MODEL_CACHE[session_id]


def predict_image(session_id: str, image_file) -> Dict:
    """
    Predict a single image using the trained model for this session.

    If confidence is below CONFIDENCE_THRESHOLD, the final display class becomes
    'Uncertain', while best_match still keeps the highest-probability class.
    """
    model_package = load_trained_model_package(session_id)

    classifier = model_package["classifier"]
    label_encoder = model_package["label_encoder"]
    class_names = model_package["class_names"]

    device = get_device()
    transform = get_preprocessing_transform()
    feature_extractor = load_feature_extractor(device)

    try:
        image = Image.open(image_file.file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    image_tensor = transform(image)

    image_features = extract_features_for_image(
        image_tensor=image_tensor,
        feature_extractor=feature_extractor,
        device=device,
    )

    image_features = image_features.reshape(1, -1)

    if not hasattr(classifier, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="The trained classifier does not support probability prediction.",
        )

    probabilities_array = classifier.predict_proba(image_features)[0]

    best_probability_index = int(np.argmax(probabilities_array))
    best_encoded_label = classifier.classes_[best_probability_index]
    best_class = label_encoder.inverse_transform([best_encoded_label])[0]

    confidence = round(float(probabilities_array[best_probability_index]) * 100, 2)

    confidence_threshold = float(
        model_package.get("confidence_threshold", CONFIDENCE_THRESHOLD)
    )

    is_uncertain = confidence < confidence_threshold
    display_class = "Uncertain" if is_uncertain else best_class

    probability_dict = {
        class_name: round(float(probability) * 100, 2)
        for class_name, probability in zip(class_names, probabilities_array)
    }

    return {
        "predicted_class": display_class,
        "best_match": best_class,
        "confidence": confidence,
        "confidence_threshold": confidence_threshold,
        "is_uncertain": is_uncertain,
        "probabilities": probability_dict,
        "model_accuracy": round(float(model_package.get("accuracy", 0)) * 100, 2),
        "trained_classes": class_names,
        "feature_extractor": model_package.get("feature_extractor", "mobilenet_v3_small"),
        "image_size": model_package.get("image_size", IMAGE_SIZE),
    }