# Video Anomaly Detection

A deep learning project for detecting abnormal events in video sequences using motion-based preprocessing and a ConvLSTM neural network. The system learns normal motion patterns from video frames and identifies unusual behavior by comparing predicted motion with actual motion.

## Demo

[Watch Demo Video](Testing_Demo_video/Anomaly_detection.mp4)

## Overview

Video anomaly detection is useful in surveillance, safety monitoring, and automated inspection systems. This project focuses on detecting abnormal motion patterns by training a sequence-based model on video frame differences.

The workflow includes:

* Preprocessing video frames
* Generating motion maps using frame differencing
* Creating sequential training samples
* Training a ConvLSTM-based model
* Evaluating anomaly behavior using reconstruction/prediction error

## Repository Structure

```text
Video-Anomaly-Detection/
│
├── script/
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   └── video_anomaly_detection_system.ipynb
│
├── Testing_Demo_video/
│   └── Anomaly_detection.mp4
│
├── .gitignore
└── README.md
```

## Notebooks

### 1. `preprocessing.ipynb`

Handles the preprocessing pipeline, including loading video frames, resizing frames, converting frames to grayscale, normalizing pixel values, and preparing data for motion analysis.

### 2. `model_training.ipynb`

Contains the deep learning model training process. The model is based on ConvLSTM layers, which are suitable for learning both spatial and temporal patterns in video sequences.

### 3. `video_anomaly_detection_system.ipynb`

Runs the complete anomaly detection pipeline, including testing, prediction, anomaly scoring, and visualization of results.

## Methodology

The project follows this main pipeline:

1. Load video frame sequences.
2. Convert frames to grayscale.
3. Resize frames to a fixed resolution.
4. Generate motion maps using the difference between consecutive frames.
5. Create time-series sequences from motion maps.
6. Train a ConvLSTM model to predict the next motion map.
7. Compare the predicted motion map with the actual one.
8. Use the prediction error as an anomaly score.

## Model Concept

The model is trained to understand normal motion behavior. When an abnormal event appears, the model struggles to predict it accurately, producing a higher prediction error. This error is then used to detect anomalies.

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Jupyter Notebook
* ConvLSTM Neural Networks

## Features

* Motion-map based video preprocessing
* Sequential frame modeling
* ConvLSTM deep learning architecture
* Anomaly score calculation
* Visual testing and result analysis
* Demo video included

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Notsuperman7/Video-Anomaly-Detection.git
```

2. Open the project folder:

```bash
cd Video-Anomaly-Detection
```

3. Install the required libraries:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

4. Open Jupyter Notebook:

```bash
jupyter notebook
```

5. Run the notebooks inside the `script/` folder in this order:

```text
1. preprocessing.ipynb
2. model_training.ipynb
3. video_anomaly_detection_system.ipynb
```

## Results

The system detects abnormal motion by measuring the difference between the model's predicted motion map and the actual motion map. Higher prediction error indicates a higher possibility of abnormal activity.

## Applications

This project can be used as a base for:

* Surveillance video analysis
* Security monitoring systems
* Public safety applications
* Industrial monitoring
* Smart camera systems
* Research in video understanding

## Future Improvements

* Add real-time video anomaly detection
* Improve model accuracy with larger datasets
* Add automatic threshold selection
* Export the trained model for deployment
* Build a simple web interface for testing videos
* Add support for multiple video datasets

## Author

**Nour Eldin Mahmoud**

## License

This project is provided for educational and research purposes.
