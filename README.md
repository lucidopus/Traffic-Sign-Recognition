# Traffic Sign Recognition using CNN

This repository contains a complete pipeline for training and deploying a traffic sign detection system using a Convolutional Neural Network (CNN) with TensorFlow/Keras, along with a FastAPI-based web API for inference.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the API](#running-the-api)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Overview

The project is designed to classify traffic signs into 43 different categories. It includes:

1. **Model Training**: A script (`trainer.py`) to load, preprocess, train, and evaluate the CNN model.
2. **Model Deployment**: A FastAPI-based backend (`main.py`) that exposes the model for inference.
3. **Prediction Pipeline**: Helper functions to process and classify uploaded images.

---

## Features

- **Image Classification**: Classifies traffic signs into 43 predefined categories.
- **Data Visualization**: Plots training and validation accuracy and loss during training.
- **FastAPI API**: Provides an endpoint for predicting the traffic sign class of uploaded images.
- **GPU Acceleration**: Utilizes GPU for faster training and inference.

---

## Requirements

Install the required Python libraries using the provided `requirements.txt`:

```bash
pandas
numpy
tensorflow
scikit-learn
matplotlib
opencv-python
fastapi
pillow
uvicorn
python-multipart
```

To install these dependencies, run:
```bash
pip install -r requirements.txt
```

## Setup

1.	Clone the repository:
```bash
git clone <repository_url>
cd <repository_name>
```

2.	Install the dependencies:
```bash
pip install -r requirements.txt
```

3.	Configure paths and parameters in utils/config.py:
- MODEL_PATH: Path to save/load the trained model.
- DATA_ROOT, TRAIN_DATA_PATH, TEST_DATA_PATH: Paths to the dataset.
- epochs: Number of training epochs.

## Usage

### Training the model

1.	Ensure your dataset is organized with images in the following structure:

```
train/
    0/
    1/
    ...
    42/
```

2.	Run the trainer.py script:

```bash
python3 trainer.py
```

This will:
- Load and preprocess the data.
- Train the CNN model.
- Save the trained model to the specified path.

## Running the API

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Access the API at http://localhost:8000.

3.	Use the /model/predict endpoint to upload an image and get the predicted traffic sign class.

---

## Project Structure
```bash
├── trainer.py         # Model training script
├── main.py            # FastAPI application
├── pipelines.py       # Prediction pipeline logic
├── helper.py          # Helper functions for image processing and prediction
├── utils/
│   ├── config.py      # Configuration file
│   ├── enums.py       # HTTP status code enums
├── requirements.txt   # List of dependencies
├── static/            # Static files (e.g., index.html)
├── data/              # Dataset root directory
└── saved_model/       # Directory to save trained models
```

---

## Acknowledgments

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset. Special thanks to the dataset creators and the open-source libraries that made this project possible.

For access to the TrafficLensv API, please [send a request](mailto:harshilpatel30402@gmail.com?subject=Request%20for%20TrafficLens%20API%20Key) for an API key.