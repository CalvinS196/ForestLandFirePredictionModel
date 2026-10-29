"""Compare Random Forest, XGBoost, and LSTM for MODIS FRP classes.

The target is derived from FRP tertiles and FRP is excluded from every
feature matrix to avoid target leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
SEQUENCE_LENGTH = 12
SPATIAL_GRID_SIZE = 0.25
DATA_PATH = Path(__file__).with_name("modis_2024_Indonesia.csv")
PREPARED_DATA_PATH = Path(__file__).with_name("modis_2024_Indonesia_spatiotemporal.csv")

np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_target(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = result.columns.str.strip()
    text_columns = result.select_dtypes(include="object").columns
    result[text_columns] = result[text_columns].apply(lambda column: column.str.strip())
    result["tingkat_frp"] = pd.qcut(
        result["frp"],
        q=3,
        labels=["Rendah", "Sedang", "Tinggi"],
    )
    return result


def encode_tabular(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = data.drop(columns=["tingkat_frp", "frp"])
    target = data["tingkat_frp"]
    categorical = features.select_dtypes(include="object").columns
    return pd.get_dummies(features, columns=categorical, dtype=float), target


def print_results(name: str, y_true: np.ndarray, predictions: np.ndarray) -> None:
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(y_true, predictions):.4f}")
    print(classification_report(y_true, predictions, target_names=CLASS_NAMES, digits=4))


def run_tree_models(data: pd.DataFrame) -> None:
    encoded_features, target = encode_tabular(data)
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    y = target.map(class_to_id).to_numpy()
    split = int(len(encoded_features) * (1 - TEST_SIZE))
    x_train, x_test = encoded_features.iloc[:split], encoded_features.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    # SMOTE is applied only to independent tabular rows, never to LSTM windows.
    x_train_balanced, y_train_balanced = SMOTE(random_state=SEED).fit_resample(
        x_train, y_train
    )

    random_forest = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, class_weight=None
    )
    random_forest.fit(x_train_balanced, y_train_balanced)
    print_results("Random Forest", y_test, random_forest.predict(x_test))

    xgboost = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(CLASS_NAMES),
        eval_metric="mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )
    xgboost.fit(x_train_balanced, y_train_balanced)
    print_results("XGBoost", y_test, xgboost.predict(x_test))


def build_spatiotemporal_features(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    prepared["acq_datetime"] = pd.to_datetime(
        prepared["acq_date"].astype(str)
        + " "
        + prepared["acq_time"].astype(str).str.zfill(4),
        errors="raise",
    )
    prepared["hour_float"] = (
        prepared["acq_datetime"].dt.hour + prepared["acq_datetime"].dt.minute / 60.0
    )
    prepared["hour_sin"] = np.sin(2 * np.pi * prepared["hour_float"] / 24.0)
    prepared["hour_cos"] = np.cos(2 * np.pi * prepared["hour_float"] / 24.0)
    prepared["day_of_year"] = prepared["acq_datetime"].dt.dayofyear
    prepared["doy_sin"] = np.sin(2 * np.pi * prepared["day_of_year"] / 366.0)
    prepared["doy_cos"] = np.cos(2 * np.pi * prepared["day_of_year"] / 366.0)

    prepared["lat_bin"] = np.floor(prepared["latitude"] / SPATIAL_GRID_SIZE).astype(int)
    prepared["lon_bin"] = np.floor(prepared["longitude"] / SPATIAL_GRID_SIZE).astype(int)
    prepared["lat_bin_center"] = (prepared["lat_bin"] + 0.5) * SPATIAL_GRID_SIZE
    prepared["lon_bin_center"] = (prepared["lon_bin"] + 0.5) * SPATIAL_GRID_SIZE
    prepared["spatial_id"] = (
        prepared["lat_bin"].astype(str) + "_" + prepared["lon_bin"].astype(str)
    )
    prepared = prepared.sort_values(["spatial_id", "acq_datetime"]).reset_index(drop=True)
    return prepared


def make_grouped_sequences(
    features: np.ndarray,
    target: np.ndarray,
    group_values: np.ndarray,
    times: np.ndarray,
    sequence_length: int,
    validation_time: np.datetime64,
    test_time: np.datetime64,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x_train, y_train = [], []
    x_validation, y_validation = [], []
    x_test, y_test = [], []
    unique_groups = np.unique(group_values)
    for group in unique_groups:
        group_idx = np.where(group_values == group)[0]
        if len(group_idx) <= sequence_length:
            continue
        for current in range(sequence_length, len(group_idx)):
            label_index = group_idx[current]
            sequence_indices = group_idx[current - sequence_length : current]
            sequence = features[sequence_indices]
            label = target[label_index]
            label_time = times[label_index]
            if label_time < validation_time:
                x_train.append(sequence)
                y_train.append(label)
            elif label_time < test_time:
                x_validation.append(sequence)
                y_validation.append(label)
            else:
                x_test.append(sequence)
                y_test.append(label)
    if not x_train or not x_validation or not x_test:
        raise ValueError(
            "Spatial-temporal sequence split failed: adjust SEQUENCE_LENGTH or split sizes."
        )
    return (
        np.asarray(x_train, dtype=np.float32),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(x_validation, dtype=np.float32),
        np.asarray(y_validation, dtype=np.int64),
        np.asarray(x_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.int64),
    )


def run_lstm(data: pd.DataFrame) -> None:
    prepared = build_spatiotemporal_features(data)
    prepared.to_csv(PREPARED_DATA_PATH, index=False)

    # Spatial-temporal features preserve location structure for LSTM sequences.
    drop_for_lstm = [
        "tingkat_frp",
        "frp",
        "acq_date",
        "acq_time",
        "acq_datetime",
        "hour_float",
        "day_of_year",
    ]
    features = prepared.drop(columns=drop_for_lstm)
    categorical = ["satellite", "instrument", "daynight", "spatial_id"]
    features = pd.get_dummies(features, columns=categorical, dtype=float)
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    target = prepared["tingkat_frp"].map(class_to_id).to_numpy()
    validation_time = prepared["acq_datetime"].quantile(
        1 - TEST_SIZE - VALIDATION_SIZE
    )
    test_time = prepared["acq_datetime"].quantile(1 - TEST_SIZE)
    validation_time = np.datetime64(validation_time.to_datetime64())
    test_time = np.datetime64(test_time.to_datetime64())
    train_rows = prepared["acq_datetime"] < pd.Timestamp(validation_time)
    if not train_rows.any():
        raise ValueError("No training rows were selected. Check time split.")

    scaler = StandardScaler()
    scaler.fit(features.loc[train_rows])
    scaled_features = scaler.transform(features)
    x_train, y_train, x_validation, y_validation, x_test, y_test = make_grouped_sequences(
        scaled_features,
        target,
        prepared["spatial_id"].to_numpy(),
        prepared["acq_datetime"].to_numpy(dtype="datetime64[ns]"),
        SEQUENCE_LENGTH,
        validation_time,
        test_time,
    )
    print(f"Prepared spatial-temporal dataset: {PREPARED_DATA_PATH.name}")
    print(f"Spatial groups: {prepared['spatial_id'].nunique()}")
    print(
        "LSTM sequences -> "
        f"train: {len(x_train)}, validation: {len(x_validation)}, test: {len(x_test)}"
    )

    class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    total = class_counts.sum()
    class_weight = {
        index: total / (len(CLASS_NAMES) * count)
        for index, count in enumerate(class_counts)
        if count > 0
    }

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, x_train.shape[2])),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=20,
        batch_size=128,
        class_weight=class_weight,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=2,
    )
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print_results("LSTM", y_test, predictions)


CLASS_NAMES = ["Rendah", "Sedang", "Tinggi"]


if __name__ == "__main__":
    modis = make_target(pd.read_csv(DATA_PATH))
    print(f"Rows: {len(modis)}")
    print(f"Features exclude: frp and tingkat_frp")
    run_tree_models(modis)
    run_lstm(modis)
