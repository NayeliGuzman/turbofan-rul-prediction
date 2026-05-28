import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='threadpoolctl')



COLUMNS = (
    ['unit', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    + [f'sensor_{i}' for i in range(1, 22)]
)

VALID_DATASETS = {"FD001", "FD002", "FD003", "FD004"}
STD_THRESHOLD = 1e-9

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_dir: str, dataset_id: str):
    """
    Load the train, test, and RUL files for a given dataset ID
    (e.g. 'FD001').  Returns (train_df, test_df, rul_df).
    """
    if dataset_id not in VALID_DATASETS:
        raise ValueError(
            f"Unknown dataset ID '{dataset_id}'. "
            f"Expected one of: {sorted(VALID_DATASETS)}"
        )

    def read_sensor_file(filename):
        return pd.read_csv(
            os.path.join(input_dir, filename),
            sep=r'\s+',
            header=None,
            names=COLUMNS,
            index_col=False,
            engine='python',
        )

    train = read_sensor_file(f'train_{dataset_id}.txt')
    test  = read_sensor_file(f'test_{dataset_id}.txt')
    rul   = pd.read_csv(
        os.path.join(input_dir, f'RUL_{dataset_id}.txt'),
        header=None,
        names=['RUL'],
    )

    return train, test, rul


# ---------------------------------------------------------------------------
# Preprocessing — one operating condition (FD001, FD003)
# ---------------------------------------------------------------------------

def drop_columns_one_op_condition(df: pd.DataFrame, STD_THRESHOLD) -> pd.DataFrame:
    """hr
    Drop sensor columns whose standard deviation falls below `threshold`.
    Used for single-operating-condition datasets (FD001, FD003).
    """
    ignore_cols = ['unit', 'cycle']
    df_sensors = df.drop(columns=ignore_cols)
    stds = df_sensors.std()
    kept = df_sensors.loc[:, stds >= STD_THRESHOLD]
    return pd.concat([df[ignore_cols], kept], axis=1)


# ---------------------------------------------------------------------------
# Preprocessing — multiple operating conditions (FD002, FD004)
# ---------------------------------------------------------------------------

def drop_columns_multi_op_condition(df: pd.DataFrame, std_threshold: float = 1e-9) -> pd.DataFrame:
    """
    Drop sensor AND op_setting columns that are essentially constant across ALL 6
    operating conditions. A column is only dropped if its std is below std_threshold
    in every condition. If it exceeds the threshold in even one condition, it is kept.
    """
    setting_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    ignore_cols  = ['unit', 'cycle']

    # All columns eligible for dropping: sensors AND op_settings
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    # Cluster rows into 6 operating conditions based on setting columns
    kmeans = KMeans(n_clusters=6, random_state=42)
    df = df.copy()
    df['op_condition'] = kmeans.fit_predict(df[setting_cols])

    # For each feature column, check std in every condition
    # A column is dropped only if std < threshold in ALL 6 conditions
    cols_to_drop = []
    for col in feature_cols:
        if col == 'op_condition':
            continue
        stds = df.groupby('op_condition')[col].std()
        if (stds < std_threshold).all():
            cols_to_drop.append(col)

    # Uncomment to verify multi-condition standard deviation check for all cols:
    #print(f"\nColumns dropped (std < {std_threshold} in all 6 conditions): {cols_to_drop}")
    # print("\nStd per condition for all feature columns:")
    # print(df.groupby('op_condition')[feature_cols].std().T.to_string())

    df = df.drop(columns=cols_to_drop + ['op_condition'])
    return df


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

def choose_preprocessing(dataset_id: str, df: pd.DataFrame, STD_THRESHOLD) -> pd.DataFrame:
    """
    Route a dataframe to the correct preprocessing function based on the
    dataset ID, then apply the given threshold.
    """
    one_op_condition   = {"FD001", "FD003"}
    multi_op_condition = {"FD002", "FD004"}

    if dataset_id in one_op_condition:
        return drop_columns_one_op_condition(df, STD_THRESHOLD)
    if dataset_id in multi_op_condition:
        return drop_columns_multi_op_condition(df, STD_THRESHOLD)

    raise ValueError(f"No preprocessing rule defined for dataset '{dataset_id}'.")

# ---------------------------------------------------------------------------
# Computing RUL labels
# ---------------------------------------------------------------------------

def compute_rul_labels(train: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'RUL' column to the training set.
    For each unit: RUL = max_cycle - current_cycle.
    """
    RUL_cap = 125
    max_cycles = train.groupby('unit')['cycle'].max()
    max_cycles = max_cycles.reset_index()

    max_cycles.columns = ['unit', 'max_cycle']

    # Merge back into train and compute RUL
    train = train.merge(max_cycles, on='unit')
    train['RUL'] = train['max_cycle'] - train['cycle']

    # Drop max_cycle column
    train = train.drop(columns=['max_cycle'])
    train['RUL'] = train['RUL'].clip(upper=RUL_cap)
    return train

# ---------------------------------------------------------------------------
# Data Normalization
# ---------------------------------------------------------------------------

def normalize_data(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """
    Fit a MinMaxScaler on train sensor columns, transform both train and test.
    Returns (train_normalized, test_normalized) with the same column structure.
    """
    scaler = MinMaxScaler() #Transforms features by scaling each feature to a given range
    train[feature_cols] = scaler.fit_transform(train[feature_cols]) #Compute the minimum and maximum to be used for later scaling
    test[feature_cols] = scaler.transform(test[feature_cols]) #Scaling features of the test data according to feature range
    return train, test


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def create_sequences(data: pd.DataFrame, feature_cols: list, window_size: int = 30):
    """
    Sliding window over each unit's time series.
    If a unit has fewer than window_size cycles, it is front-padded by
    repeating its first row so it yields exactly one sequence.
    Label is the RUL at the last timestep of the window.

    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples,)
    """
    X, y = [], []
    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit]
        features  = unit_data[feature_cols].values
        rul_vals  = unit_data['RUL'].values

        if len(features) < window_size:
            pad_length = window_size - len(features)
            padding    = np.repeat(features[0:1], pad_length, axis=0)
            features   = np.vstack([padding, features])
            rul_vals   = np.pad(rul_vals, (pad_length, 0), mode='edge')

        for i in range(len(features) - window_size + 1):
            X.append(features[i : i + window_size])
            y.append(rul_vals[i + window_size - 1])

    return np.array(X), np.array(y)


def create_test_sequences(data: pd.DataFrame, feature_cols: list, window_size: int = 30):
    """
    For each unit in the test set, take the LAST window_size cycles.
    If a unit has fewer than window_size cycles, it is front-padded by
    repeating its first row so every unit yields exactly one sequence.

    Returns:
        X: (n_units, window_size, n_features)
    """
    X = []
    for unit in data['unit'].unique():
        unit_data = data[data['unit'] == unit]
        features  = unit_data[feature_cols].values

        if len(features) < window_size:
            pad_length = window_size - len(features)
            padding    = np.repeat(features[0:1], pad_length, axis=0)
            features   = np.vstack([padding, features])

        X.append(features[-window_size:])
    return np.array(X)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NASA CMAPSS Preprocessing Script")
    parser.add_argument('--input',       required=True,            help="Directory containing raw .txt files")
    parser.add_argument('--output',      required=True,            help="Directory to write .npz output files")
    parser.add_argument('--dataset',   required=True,              help="Dataset ID, e.g. FD001")
    parser.add_argument('--window_size', type=int,   default=30,   help="Sequence window size)")
    args = parser.parse_args()

    dataset_id = args.dataset.upper()
    print(f"Processing {dataset_id} | window={args.window_size}")

    # 1. Load the raw files for this dataset
    train, test, rul = load_data(args.input, dataset_id)

    # print("initial Train shape:", train.shape)
    # print("initial Test shape:", test.shape)
    # print("initial RUL shape:", rul.shape)

    # 2. Apply the appropriate column-dropping logic to train and test
    train = choose_preprocessing(dataset_id, train, STD_THRESHOLD)
    test  = choose_preprocessing(dataset_id, test,  STD_THRESHOLD)
    # print("after dropped cols Train shape:", train.shape)
    # print("after dropped cols Test shape:", test.shape)

    # 3. Compte RUL labels for the training set
    train = compute_rul_labels(train)
    
    # Derive the feature columns
    feature_cols = [col for col in train.columns if col not in ['unit', 'cycle', 'RUL']]
    # 4. Normalize the sensor data (test and train)
    train, test = normalize_data(train, test, feature_cols)

    # 5. Sequences
    X_train, y_train = create_sequences(train, feature_cols, window_size = 30)
    X_test = create_test_sequences(test, feature_cols, window_size = 30)
    y_test = rul['RUL'].values

    # print("X_train shape after windowing", X_train.shape)
    # print("y_train shape after windowing", y_train.shape)
    # print("X_test shape after windowing", X_test.shape)
    # print("y_test shape after windowing", y_test.shape)

    # 6. Save output
    os.makedirs(args.output, exist_ok=True)

    train_out = os.path.join(args.output, f'train_{dataset_id}.csv')
    test_out  = os.path.join(args.output, f'test_{dataset_id}.csv')
    rul_out   = os.path.join(args.output, f'RUL_{dataset_id}.csv')

    # Save as .npz then load in notebook with np.load(path, allow_pickle=False)
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f'{dataset_id}.npz')
    np.savez(out_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"  Saved:  {out_path}")

if __name__ == "__main__":
    main()
