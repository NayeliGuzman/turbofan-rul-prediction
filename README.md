# Predictive Maintenance: Turbofan Jet Engine RUL Estimation

In this project, a deep learning hybrid network (1D CNN and LSTM) is designed to solve the regression task of predicting Remaining Useful Life (RUL) using the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset. 
The goal is to avoid overestimation of RUL and consistently under-estimate the remaining life to ensure maintenan    ce is scheduled before catostrophic failure occurs.

## I. Dataset
The data set is  provided by the NASA Ames Prognostics Center of Excellence (PCoE).
Download: [data]( https://data.phmsociety.org/nasa/)
Citation: A. Saxena and K. Goebel (2008). “Turbofan Engine Degradation Simulation Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

## II. Preprocessing Pipeline

Raw CMAPSS `.txt` files are processed via a dedicated pipeline (`src/preprocessing/`) for downstream model training:

**Column filtering** drops constant sensors (std < 1e-9); for multi-condition datasets, this is evaluated per operating condition via KMeans clustering

**RUL labeling** computed as `max_cycle - current_cycle` per unit

**Normalization** MinMaxScaler fit on train only, applied to both train and test to prevent data leakage

**Sequence windowing** sliding window of 30 cycles; short sequences are front-padded

Output is one `.npz` file per dataset containing `X_train`, `y_train`, `X_test`, `y_test`.
See [Pre-processing Only](#pre-processing-only) for instructions to run.

## III. Notebooks
**`01_cnn_lstm_FD001.ipynb`** contains the pre-processing, training, and evaluation on FD001 data (single operating condition, single fault mode). Experiments compare loss functions for their effect on prediction bias and safety characteristics.

**`02_cnn_lstm_FD002.ipynb`** extends the previous experiments to FD002 (six operating conditions, single fault) and investigates asymmetric loss function robustness under more complex operating conditions.

**`03_cnn_lstm_FD003.ipynb`** applies the pipeline to FD003 (single operating condition, two modes of failure) with an emphasis on how asymmetric loss functions perform under increased fault complexity. 

**`04_cnn_lstm_FD004.ipynb`** explores FD004 (six operating conditions, two fault modes) to assess model and asymmetric loss function performance in the most complex settings. 

## VI. Model Architecture Overview, Loss Function Experiments and Metrics
This system fuses spatial feature extraction with temporal sequence modeling to handle multivariate, noisy time-series sensor data. 

**1D-CNN Layer:** Automatically extracts localized, spatial, and cross-channel sensor feature correlations within specific time windows.

**LSTM Layer:** Learns long-term temporal degradation trajectories and sequential dependencies across consecutive operational cycles.

**Regression Head:** A fully connected Dense layer mapping the combined feature maps to a continuous, single real-valued RUL scalar output.

## Loss Function Experiments and Metrics Evaluated

RUL prediction in safety-critical systems requires conservative estimates.

1. **Symmetric MSE**:  baseline, treats over and underestimation equally
2. **Asymmetric MSE**:  penalizes overestimation more heavily than underestimation
3. **NASA S-Score**:  used as an evaluation metric rather than a training loss, consistent with PHM literature

## V. Getting Started

1. Clone the repository
```bash
git clone https://github.com/NayeliGuzman/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset and place in 'nasa_data'. The path to the downloaded data should be `"nasa_data/6. Turbofan Engine Degradation Simulation Data Set-CMAPSSData"`

4. Run any of the notebooks (e.g. `01_cnn_lstm_FD001.ipynb`)

## Requirements

See `requirements.txt`. Key dependencies: TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib.



## Pre-processing Only

1. Make the script executable (run once)
```bash
chmod +x src/preprocessing/run_preprocessing.sh
```

2. Run preprocessing for all datasets at once and save in the desired file path
```bash
./src/preprocessing/run_preprocessing.sh \
  "nasa_data/6. Turbofan Engine Degradation Simulation Data Set-CMAPSSData" \
  nasa_data/clean_data
```

This will generate `FD001.npz`, `FD002.npz`, `FD003.npz`, and `FD004.npz` inside `nasa_data/clean_data/`.

3. Load a dataset into any notebook and you are all set to explore the data
```python
import numpy as np

# Example for FD001
data = np.load('nasa_data/clean_data/FD001.npz')
X_train, y_train = data['X_train'], data['y_train']
X_test,  y_test  = data['X_test'],  data['y_test']
```
