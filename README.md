# Teachable Machine AI — Full Stack Image Classification App

A full-stack AI web application inspired by Google Teachable Machine.  
The project allows users to create custom image classes, upload or capture image samples, train a transfer learning model, and test predictions using image upload or webcam input.

The application uses:

- **Streamlit** for the frontend UI
- **FastAPI** for the backend ML API
- **MobileNetV3** for transfer learning feature extraction
- **Logistic Regression** for fast custom classification
- **Docker Compose** for running frontend and backend together

---

## Project Overview

This project moves beyond notebook-based machine learning and implements a real-world full-stack AI workflow.

Users can:

1. Create image classes
2. Upload image samples
3. Capture webcam samples
4. Train a custom image classifier
5. Predict images from file upload
6. Predict images from webcam preview
7. View confidence scores for each class
8. Export the trained model
9. Reset the project session

---

## Features

### Frontend Features

- Teachable Machine inspired UI
- Dynamic class cards
- Editable class names
- Add, disable, enable, and delete classes
- Upload image samples per class
- Capture webcam samples per class
- Training status display
- Preview panel for predictions
- File-based prediction
- Webcam frame prediction
- Confidence bars for model output
- Backend online/offline status
- Reset project button
- Export trained model button

### Backend Features

- FastAPI REST API
- Upload image samples
- Session-based dataset storage
- Temporary per-user dataset folders
- Dataset summary endpoint
- Class deletion endpoint
- Transfer learning training endpoint
- Prediction endpoint
- Model export endpoint
- Reset session endpoint
- Old session cleanup endpoint

### Machine Learning Features

- Pre-trained MobileNetV3 feature extractor
- ImageNet normalization
- 224x224 image preprocessing
- Logistic Regression classifier
- Label encoding
- Train/test split
- Accuracy calculation
- Saved model package using joblib
- Class-wise prediction probabilities

---

## Tech Stack

### Frontend

- Python
- Streamlit
- streamlit-webrtc
- OpenCV
- Pillow
- Requests

### Backend

- Python
- FastAPI
- Uvicorn
- PyTorch
- Torchvision
- Scikit-learn
- Joblib
- Pillow
- Python Multipart

### DevOps

- Docker
- Docker Compose

---

## Project Structure

```txt
teachable-machine-ai/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── utils/
│   │   │   ├── file_utils.py
│   │   │   └── session_utils.py
│   │   ├── main.py
│   │   └── ml_engine.py
│   │
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
│
├── docker-compose.yml
├── .dockerignore
└── README.md