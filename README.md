# Pose_Inference: A Real-time Pose Classification Framework

Pose_Inference is a framework for building custom pose-based applications, such as exercise analysis systems. It enables users to run real-time pose inference using a pre-trained model and also allows them to train a custom pose inference model on their own dataset. 
Developed with a focus on robust testing, this project is intended to serve as a reliable foundation for more complex tasks based on pose prediction.

## Pose Estimation Demo Video
Click the image below to watch the demo video on YouTube:

<a href="https://youtu.be/5zruC2FnAPU" target="_blank" rel="noopener noreferrer">
  <img src="https://img.youtube.com/vi/5zruC2FnAPU/hqdefault.jpg" width="600" alt="Watch on YouTube — 5zruC2FnAPU" />
</a>

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [Features](#features)
3.  [Technical Stack](#technical-stack)
4.  [Project Structure](#project-structure)
5.  [Getting Started: Running the Application](#getting-started-running-the-application)
6.  [Training a Custom Pose Model](#training-a-custom-pose-model)
7.  [Testing the Project](#testing-the-project)

## Project Overview

This project provides a complete pipeline for real-time and efficient pose estimation and classification. It uses a webcam for real-time video capture, detects human poses using MediaPipe, and classifies them into predefined categories using a trained XGBoost model. The framework is modular, separating the inference pipeline from the training process, and includes a comprehensive test suite to ensure reliability.
It can run smoothly on an Intel Celeron CPU, without a GPU.

## Features

*   **Real-time Pose Classification:** Classify human poses from a live webcam feed.
*   **Custom Model Training:** A full pipeline to train a new classification model on your own image dataset.
*   **Modular Architecture:** Code is cleanly separated into a training module (`train_pose_inference`) and an inference module (`pose_model_pipeline`).
*   **Heavily Tested:** Developed using Test-Driven Development (TDD) to ensure core logic is reliable and robust.

## Technical Stack

*   **Core:** Python 3.12
*   **Computer Vision:** OpenCV, MediaPipe
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Testing:** Pytest, Pytest-Cov

## Project Structure

Understanding the project layout will help you navigate the codebase and customize it for your needs.

```
pose_estimation_rough/
│
├── __main__.py               # Main entry point to run the real-time application.
├── requirements.txt          # Project dependencies.
│
├── assets/
│   └── classification_model/ # Default location for the trained model artifacts.
│       ├── direct_model.pkl
│       └── pose_label_encoder.pkl
│
├── pose_model_pipeline/      # Python package for the inference pipeline.
│   └── src/
│       └── inference_pipeline.py
│
├── train_pose_inference/     # Python package for the model training pipeline.
│   ├── pose_images/          # <-- PLACE YOUR CUSTOM TRAINING IMAGES HERE
│   └── src/
│       ├── __main__.py       # Entry point for the training script.
│       └── train.py          # Core model training logic.
│
└── venv312/                  # Example virtual environment directory.
```

## Getting Started: Running the Application

Follow these steps to run the real-time pose classification demo.

1.  **Clone the Repository:**
    First, create an empty folder on your local machine and clone the project into it.
    ```bash
    git clone https://github.com/SomtoOnyekwelu/training_pose_recognition_model
    ```

2.  **Navigate to the Parent Directory:**
    All commands should be run from the directory **containing** your `pose_estimation_rough` project folder, not from inside it. For example, if you cloned the project into `C:\dev\`, you should stay in `C:\dev\`.

3.  **Create and Activate Virtual Environment:**
    It is recommended to use Python 3.12.
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r pose_estimation_rough/requirements.txt
    ```

5.  **Run the Application:**
    Execute the project as a Python module. This will launch a window with your webcam feed.
    ```bash
    python -m pose_estimation_rough
    ```

## Training a Custom Pose Model

You can train the classifier on your own set of poses.

1.  **Prepare Your Image Dataset:**
    Place your training images inside the `pose_estimation_rough/train_pose_inference/pose_images/` directory. The structure must follow this format: create one subdirectory for each pose class you want to train.

    For example:
    ```
    pose_estimation_rough/train_pose_inference/pose_images/
    ├── jumping_jacks/
    │   ├── img1.jpg
    │   └── img2.png
    └── push_ups/
        ├── img3.jpg
        └── img4.jpeg
    ```

2.  **Run the Training Script:**
    From the directory **containing** the `pose_estimation_rough` project folder, run the training module. The script will automatically find your images, train the model, and save the new artifacts.

    ```bash
    python -m pose_estimation_rough.train_pose_inference.src
    ```

> **⚠️ Important: Overwriting Model Artifacts**
> Running the training script will **overwrite** the existing `direct_model.pkl` and `pose_label_encoder.pkl` files in the `assets/classification_model/` directory. If you want to keep the original model, make sure to back up these files before starting a new training run.

> **💡 Pro Tip for High Accuracy**
> For best results, ensure the **full body** of the person is visible in your training images and during live inference. The original model was trained on full-body poses, and performance will be significantly better if the input data matches this format. Avoid poses where the body is partially out of frame or occluded by objects.

## Testing the Project

This project was developed with a strong emphasis on quality and reliability through Test-Driven Development (TDD).

> **Commitment to Quality**
> The project includes **296 passing tests** providing **77% code coverage**. This comprehensive test suite covers all critical edge cases and core logic, ensuring the framework is robust and dependable for future development.

**Note:** All test commands must be run from the directory **containing** the `pose_estimation_rough` project folder.

1.  **Run All Tests:**
    To run the entire test suite, use `pytest`. The `-rf` flag shows a summary of failed and errored tests.
    ```bash
    pytest pose_estimation_rough -rf
    ```

2.  **Run Tests with Coverage:**
    To generate a code coverage report, run pytest from the project root using the `--cov` flag.
    ```bash
    cd pose_estimation_rough 
    pytest --cov=pose_estimation_rough
    ```
