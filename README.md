# Classification / Anomaly Detection for Automobile Assembly Process

This repository contains the implementation and results for the [Classification / Anomaly Detection] task for Automobile Assembly Process.

## Overview
The project focuses on classifying robot arm data into normal and defective categories. Unlike common datasets such as images or speech signals, this project aims to handle time-series data that includes specific metrics like force and torque, providing valuable experience in managing specialized tasks. The primary goal is to explore various methods and techniques, regardless of achieving optimal results.

## Dataset Description
The project utilized two types of data:

1. **Action Data**:
   - Time-series data obtained from end-effector sensors on the robot arm, measuring forces (`Fx, Fy, Fz`) and torques (`Tx, Ty, Tz`) in each direction at every timestep.
   - Data labels:
     - `OK`: 0
     - `NG`: 1
     - `NG_+Z_inner_path`: 2
     - `NG_-X_Y_offset`: 3
     - `NG_-Z_outer_path`: 4
     - `NG_X_-Y_offset`: 5

2. **Sticker Image Data**:
   - Image data labeled as:
     - `OK`: 0
     - `NG`: 1
   - To be developed...

## Methodology
### Data Preprocessing
- **Action Data**:
  - Noise reduction using low-pass filters.
  - Feature engineering, including rate of change of variables, angles between vectors, and vector directions.

- **Sticker Image Data**:
  - Resizing to match model input size.
  - Standardization using pre-trained dataset statistics.
  - To be developed...

### Model List
- **Action Data**:
  - Classification
    - LSTMClassifier
    - GRUClassifier
    - CNN1DClassifier
    - TCNClassifier
    - TransformerClassifier
  - Anomaly Detection
    - AutoEncoder (Linear)
    - RNNAE (AutoEncoder based on LSTM)
    - CNN1DAE (AutoEncoder based on Conv1D)


- **Sticker Data**:
    To be developed...

### Experiments
1. **Baseline Models**:
   - Initial experiments with basic LSTM without preprocessing: Accuracy = 0.5.
2. **With Preprocessing**:
   - LSTM: Improved but not significant.
   - CNN1D: Loss = 0.4472, Accuracy = 0.8077.
   - Transformer-based model: Loss = 0.0591, Accuracy = 0.9231.
3. **Additional Features**:
   - Expanded features to 21 dimensions: Loss = 0.0679, Accuracy = 0.9423.

### Anomaly Detection Approach
- **Method**:
  - Used CNN1D-based Autoencoder trained only on normal data.
  - Detected anomalies by evaluating reconstruction errors.
- **Results**:
  - Optimal Threshold: 0.026091
  - Precision: 0.8571
  - Recall: 0.7000
  - F1-Score: 0.7706

## Conclusion
- Classification Task:
  - Achieved high accuracy (0.9231) using Transformer models with preprocessing.
- Anomaly Detection:
  - Demonstrated the potential of Autoencoder-based approaches for detecting anomalies.

## Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/Dexoculus/Anomaly-Detection-for-Car-Assemble-Process.git
cd Anomaly-Detection-for-Car-Assemble-Process
```
2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```
Make sure PyTorch and other dependencies (such as torchvision, yaml, matplotlib) are installed. Adjust the requirements as needed.

Also, My code uses a experiment_manager from:
https://github.com/Dexoculus/PyTorch-Experiment-Manager

3. **Run the main code:**
- main.py (for Classification task)
```bash
python main.py --config ./configs/classification_config.yaml
```

- anomal_main.py (for Anomaly Detection task)
```bash
python anomal_main.py --config ./configs/anomal_config.yaml
```

4. Check the Result.
- Checkpoints: Stored in `./checkpoints/` by default.
- Loss plots: Stored in the path you write in config if visualization is enabled.
- results files: Stored in the path you write in config if exporting is enabled.
- Logs: Training and validation logs displayed in the terminal.