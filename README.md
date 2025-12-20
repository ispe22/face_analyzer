# Face Analyzer: Age & Celebrity Matcher

A computer vision project that analyzes facial images to predict age and identify celebrity lookalikes, featuring a unified web interface and two fine-tuned ResNet models.

**Live Demo:** [https://face-analyzer-render.onrender.com/](https://face-analyzer-render.onrender.com/)

*Note: The Hugging Face Spaces APIs may take a moment to load and provide predictions.*

**Hugging Face Spaces:**
*   [Age Predictor API](https://huggingface.co/spaces/ispe/age_predictor)
*   [Celebrity Matcher API](https://huggingface.co/spaces/ispe/face_similarity)

## Features

-   **Age Prediction:** Estimates a person's age using a regression model.
-   **Celebrity Lookalike:** Matches the face against 8,630 famous identities to find the top 5 most similar celebrities.
-   **Interactive UI:** A static web frontend that uploads images and polls multiple API endpoints asynchronously.

## Architecture

1.  **Frontend:** Static HTML/JS hosted on Render.
2.  **Backend:** Two separate Gradio applications hosted on Hugging Face Spaces acting as API endpoints.
3.  **Models:** Two ResNet-18 models trained with FastAI.

## Tech Stack

*   **Core:** Python, FastAI, PyTorch
*   **Deployment:** Hugging Face Spaces, Docker, Render
*   **Web:** Vanilla JavaScript, HTML5, CSS3

## Model Performance

### 1. Celebrity Lookalike Model
*   **Task:** Classification (Identity).
*   **Dataset:** VGGFace2 (Filtered to 8,630 classes with 100 images each).
*   **Accuracy:** **80.6%** (Performance is based on the validation set, not a separate test set).

### 2. Age Prediction Model
*   **Task:** Regression (Age estimation).
*   **Dataset:** UTKFace (filtered to ~10,000 images for a roughly balanced dataset).
*   **Error (MAE):** **± 6.60 years** (Performance is based on the validation set, not a separate test set).

## Screenshot

<img width="523" height="945" alt="image" src="https://github.com/user-attachments/assets/0e9d740c-5b5e-4ef8-ac43-210dacdb5bad" />
