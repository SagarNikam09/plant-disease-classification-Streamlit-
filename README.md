# Plant Disease Classification System

## Overview

The **Plant Disease Classification System** is an AI-powered web application built with Streamlit, TensorFlow, and Keras, aimed at detecting plant diseases from images of leaves. The system uses a trained deep learning model to classify images of plant leaves into different disease categories.

The system allows users to upload an image of a plant, and the model will predict if the plant is healthy or infected, and if infected, which disease it has.

## Features

- **Image Upload**: Upload an image of a plant leaf.
- **Prediction**: The system will analyze the image and predict the disease (if any) using a trained model.
- **User-Friendly Interface**: The system provides a simple and intuitive web interface using Streamlit.
- **Fast and Efficient**: Predictions are generated in seconds.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [License](#license)

## Installation

### Prerequisites

Make sure you have Python 3.x installed and a virtual environment set up.

1. Clone the repository:
   ```bash
   git clone https://github.com/SagarNikam09/plant-disease-classification-Streamlit-
   cd plant-disease-classification-Streamlit-
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

The `requirements.txt` file should contain the following libraries:

```
streamlit
tensorflow
numpy
Pillow
json
```

You can also manually install these dependencies using:

```bash
pip install streamlit tensorflow numpy Pillow
```

## Usage

### Running the Web App

1. After installing the required libraries, run the following command to start the Streamlit web application:

   ```bash
   streamlit run main.py
   ```

2. Open the web browser and navigate to `http://localhost:8501` to interact with the Plant Disease Classification System.

### How It Works

- **Upload an Image**: Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
- **Prediction**: After uploading the image, click on **Predict**, and the system will classify the image and provide the disease label (if any).
  
### Model Prediction

The model used in the system is a deep learning model trained on a dataset of plant leaves. The model predicts the disease based on image classification, and the results are displayed on the web page.

### Supported Pages

- **Home**: Overview of the system and how it works.
- **About**: Details about the project, dataset, and how the model was trained.
- **Disease Recognition**: Upload an image and get predictions for the disease.

## Model Details

The model used in this system is a Convolutional Neural Network (CNN) trained to classify plant diseases based on images of leaves. The model was trained on a large dataset of labeled plant images.

- **Model Format**: The model is saved in the `.h5` format.
- **Input Size**: The model expects images of size 128x128 pixels.
- **Framework**: TensorFlow and Keras.

## Dataset

The dataset used for training this model contains approximately 87,000 RGB images of healthy and diseased crop leaves. The dataset includes 38 classes of different diseases, such as:

- Apple scab
- Potato Early blight
- Tomato Bacterial spot
- And more...

You can access the original dataset [here](https://www.kaggle.com/datasets/emmarex/plantdisease).

The dataset is split into training (80%) and validation (20%) sets, with an additional directory for testing purposes.

## Technologies Used

- **Streamlit**: A framework for building the web interface.
- **TensorFlow/Keras**: For training and inference of the deep learning model.
- **Python**: The primary programming language used.
- **Pillow**: For image preprocessing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
