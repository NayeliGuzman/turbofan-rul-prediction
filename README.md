# Predictive Maintenance: Turbofan Jet Engine RUL Estimation

In this project, a deep learning hybrid network (1D CNN and LSTM) is designed to solve the regression task of predicting Remaining Useful Life (RUL) using the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset. 
The goal is to avoid overestimation of RUL and consistently under-estimate the remaining life to ensure maintenan    ce is scheduled before catostrophic failure occurs.

## Dataset
The data set is  provided by the NASA Ames Prognostics Center of Excellence (PCoE).
Download: [data]( https://data.phmsociety.org/nasa/)
 
Citation: A. Saxena and K. Goebel (2008). “Turbofan Engine Degradation Simulation Data Set”, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

## Notebooks
**`01_cnn_lstm_FD001.ipynb`** contains the pre-processing, training, and evaluation on FD001 data (single operating condition, single fault mode). Experiments compare loss functions for their effect on prediction bias and safety characteristics.

**`02_cnn_lstm_FD002.ipynb`** extends the previous experiments to FD002 (six operating conditions, single fault);Investigating asymmetric loss function robustness under more complex operating conditions using the same CNN-LSTM architecture.

**`03_cnn_lstm_FD003.ipynb`** *(in progress)* applies the pipeline to FD003 (single operating condition, two modes of failure) with an emphasis on how asymmetric loss functions perform under increased fault complexity. 

**`04_cnn_lstm_FD004.ipynb`** *(in progress)* explores FD004, featuring six operating conditions and two fault modes, to assess model and asymmetric loss function performance in the most complex settings. 

## Model Architecture Overview
This system fuses spatial feature extraction with temporal sequence modeling to handle multivariate, noisy time-series sensor data. 
* **1D-CNN Layer:** Automatically extracts localized, spatial, and cross-channel sensor feature correlations within specific time windows.
* **LSTM Layer:** Learns long-term temporal degradation trajectories and sequential dependencies across consecutive operational cycles.
* **Regression Head:** A fully connected Dense layer mapping the combined feature maps to a continuous, single real-valued RUL scalar output.

## Loss Function Experiments and Metrics Evaluated

RUL prediction in safety-critical systems requires conservative estimates.

1. **Symmetric MSE**:  baseline, treats over and underestimation equally
2. **Asymmetric MSE**:  penalizes overestimation more heavily than underestimation
3. **NASA S-Score**:  used as an evaluation metric rather than a training loss, consistent with PHM literature

## Getting Started

1. Clone the repository
```bash
git clone https://github.com/NayeliGuzman/turbofan-rul-prediction.git
cd turbofan-rul-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset and place in 'nasa_data'

4. Run `01_cnn_lstm_FD001.ipynb`

## Requirements

See `requirements.txt`. Key dependencies: TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib.
