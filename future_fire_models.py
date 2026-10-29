"""Compare models for next-day fire-detection risk by spatial grid cell.

This script converts MODIS detection events into a dense daily panel:

    grid cell + day t -> detection features on day t
    target            -> any detection in the same cell on day t + 1

The source file contains detections rather than verified no-fire observations,
so zero-detection days are a proxy negative class. Use an observation/quality
mask and independent fire labels before using this for operational warnings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	average_precision_score,
	classification_report,
	confusion_matrix,
	ConfusionMatrixDisplay,
	precision_recall_curve,
	roc_auc_score,
	roc_curve,
)


SEED = 42
FEATURES = [
	"fire_count",
	"frp_sum",
	"frp_max",
	"brightness_mean",
	"bright_t31_mean",
	"confidence_max",
	"month_sin",
	"month_cos",
	"day_of_year_sin",
	"day_of_year_cos",
	"latitude",
	"longitude",
]


def make_daily_panel(raw: pd.DataFrame, grid_size: float) -> pd.DataFrame:
	data = raw.copy()
	data["date"] = pd.to_datetime(data["acq_date"], errors="coerce").dt.normalize()
	data = data.dropna(subset=["date", "latitude", "longitude"])
	data["grid_lat"] = np.floor(data["latitude"] / grid_size).astype("int32")
	data["grid_lon"] = np.floor(data["longitude"] / grid_size).astype("int32")
	data["cell_id"] = data["grid_lat"].astype(str) + "_" + data["grid_lon"].astype(str)

	observed = data.groupby("cell_id", as_index=False).agg(
		grid_lat=("grid_lat", "first"),
		grid_lon=("grid_lon", "first"),
		latitude=("latitude", "mean"),
		longitude=("longitude", "mean"),
	)
	dates = pd.date_range(data["date"].min(), data["date"].max(), freq="D")
	observed["key"] = 1
	date_frame = pd.DataFrame({"date": dates, "key": 1})
	panel = observed.merge(date_frame, on="key").drop(columns="key")

	daily = data.groupby(["cell_id", "date"], as_index=False).agg(
		fire_count=("cell_id", "size"),
		frp_sum=("frp", "sum"),
		frp_max=("frp", "max"),
		brightness_mean=("brightness", "mean"),
		bright_t31_mean=("bright_t31", "mean"),
		confidence_max=("confidence", "max"),
	)
	panel = panel.merge(daily, on=["cell_id", "date"], how="left")
	for column in [
		"fire_count", "frp_sum", "frp_max", "brightness_mean",
		"bright_t31_mean", "confidence_max",
	]:
		panel[column] = panel[column].fillna(0.0)
	panel["month_sin"] = np.sin(2 * np.pi * panel["date"].dt.month / 12)
	panel["month_cos"] = np.cos(2 * np.pi * panel["date"].dt.month / 12)
	panel["day_of_year_sin"] = np.sin(2 * np.pi * panel["date"].dt.dayofyear / 366)
	panel["day_of_year_cos"] = np.cos(2 * np.pi * panel["date"].dt.dayofyear / 366)
	panel = panel.sort_values(["cell_id", "date"]).reset_index(drop=True)
	panel["target"] = (
		panel.groupby("cell_id")["fire_count"].shift(-1).fillna(0).gt(0).astype("int8")
	)
	# The final global date cannot have a known next-day target.
	return panel[panel["date"] < panel["date"].max()].reset_index(drop=True)


def chronological_split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	dates = panel["date"].sort_values().unique()
	train_end = dates[int(len(dates) * 0.70)]
	validation_end = dates[int(len(dates) * 0.85)]
	return (
		panel[panel["date"] < train_end].copy(),
		panel[(panel["date"] >= train_end) & (panel["date"] < validation_end)].copy(),
		panel[panel["date"] >= validation_end].copy(),
	)


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
	precision, recall, thresholds = precision_recall_curve(y_true, probability)
	if len(thresholds) == 0:
		return 0.5
	f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
	return float(thresholds[int(np.nanargmax(f1))])


def evaluate(name, model, x, y, output: Path, threshold: float | None = None):
	probability = model.predict_proba(x)[:, 1]
	if threshold is None:
		threshold = choose_threshold(y, probability)
	prediction = (probability >= threshold).astype("int8")
	matrix = confusion_matrix(y, prediction, labels=[0, 1])
	report = classification_report(y, prediction, output_dict=True, zero_division=0)
	result = {
		"model": name,
		"threshold": threshold,
		"roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else None,
		"average_precision": average_precision_score(y, probability),
		"confusion_matrix": matrix.tolist(),
		"precision_class_1": report["1"]["precision"],
		"recall_class_1": report["1"]["recall"],
		"f1_class_1": report["1"]["f1-score"],
	}
	return result, y, probability, prediction


def plot_validation(name, y, probability, prediction, output: Path):
	output.mkdir(parents=True, exist_ok=True)
	fpr, tpr, _ = roc_curve(y, probability)
	precision, recall, _ = precision_recall_curve(y, probability)
	figure, axes = plt.subplots(1, 3, figsize=(18, 5))
	axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y, probability):.3f}")
	axes[0].plot([0, 1], [0, 1], "k--")
	axes[0].set(title=f"{name} validation ROC", xlabel="False positive rate", ylabel="Recall")
	axes[0].legend()
	axes[1].plot(recall, precision, label=f"AP={average_precision_score(y, probability):.3f}")
	axes[1].set(title=f"{name} validation PR", xlabel="Recall", ylabel="Precision")
	axes[1].legend()
	ConfusionMatrixDisplay.from_predictions(
		y, prediction, labels=[0, 1], display_labels=["No detection", "Detection"],
		cmap="Blues", ax=axes[2], colorbar=False,
	)
	axes[2].set_title(f"{name} validation confusion matrix")
	figure.tight_layout()
	figure.savefig(output / f"{name.lower()}_validation.png", dpi=160)
	plt.close(figure)


def plot_training_curves(curves, output: Path):
	"""Save three black-and-white validation ROC-AUC subplots in one figure."""
	output.mkdir(parents=True, exist_ok=True)
	figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
	labels = {
		"random_forest": "Random Forest",
		"xgboost": "XGBoost",
		"lstm": "LSTM",
	}
	for name, curve in curves.items():
		axis = axes[list(labels).index(name)]
		axis.plot(
			curve["steps"],
			curve["values"],
			color="black",
			linestyle="-",
			marker="o",
			markersize=4,
			linewidth=1.8,
		)
		axis.set_title(labels[name])
		axis.set_xlabel("Trees" if name != "lstm" else "Epochs")
		axis.grid(True, color="0.85", linestyle="--", linewidth=0.7)
	axes[0].set_ylabel("Validation ROC-AUC")
	figure.suptitle("Training Curve Comparison", fontsize=14)
	figure.tight_layout()
	figure.savefig(output / "training_curves_comparison.png", dpi=180)
	plt.close(figure)


def build_lstm_sequences(panel, sequence_length, maximum, seed):
	x_values, y_values = [], []
	metadata = []
	for _, group in panel.groupby("cell_id", sort=False):
		values = group[FEATURES].to_numpy(dtype="float32")
		labels = group["target"].to_numpy(dtype="int8")
		for end in range(sequence_length, len(group)):
			x_values.append(values[end - sequence_length:end])
			y_values.append(labels[end])
			metadata.append(group.iloc[end][["cell_id", "date", "latitude", "longitude"]].to_dict())
	if len(x_values) > maximum:
		rng = np.random.default_rng(seed)
		indices = rng.choice(len(x_values), maximum, replace=False)
		x_values = [x_values[index] for index in indices]
		y_values = [y_values[index] for index in indices]
		metadata = [metadata[index] for index in indices]
	return (
		np.asarray(x_values, dtype="float32"),
		np.asarray(y_values, dtype="int8"),
		pd.DataFrame(metadata),
	)


def train(args):
	output = Path(args.output)
	output.mkdir(parents=True, exist_ok=True)
	raw = pd.read_csv(args.data)
	panel = make_daily_panel(raw, args.grid_size)
	train_panel, validation_panel, test_panel = chronological_split(panel)
	train_x, validation_x, test_x = (
		train_panel[FEATURES].to_numpy(),
		validation_panel[FEATURES].to_numpy(),
		test_panel[FEATURES].to_numpy(),
	)
	train_y, validation_y, test_y = (
		train_panel["target"].to_numpy(),
		validation_panel["target"].to_numpy(),
		test_panel["target"].to_numpy(),
	)
	metadata = {
		"target": "any MODIS detection in the same grid cell on the next day",
		"negative_label_warning": "zero detection is a proxy negative because source data lacks observation coverage",
		"grid_size_degrees": args.grid_size,
		"sequence_length_days": args.sequence_length,
		"split": "70/15/15 chronological by target date",
		"features": FEATURES,
	}
	panel.to_csv(output / "future_fire_daily_panel.csv", index=False)
	(output / "future_fire_metadata.json").write_text(json.dumps(metadata, indent=2))

	models = {
		"random_forest": RandomForestClassifier(
			n_estimators=args.trees, min_samples_leaf=5, class_weight="balanced",
			random_state=SEED, n_jobs=-1,
		),
	}
	model_parameters = {
		"random_forest": {
			"n_estimators": args.trees,
			"min_samples_leaf": 5,
			"class_weight": "balanced",
		},
	}
	try:
		from xgboost import XGBClassifier
	except ImportError as error:
		raise SystemExit("Install xgboost to run the three-model comparison.") from error
	positive_weight = max((train_y == 0).sum() / max((train_y == 1).sum(), 1), 1.0)
	models["xgboost"] = XGBClassifier(
		n_estimators=args.trees, max_depth=5, learning_rate=0.05,
		min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
		scale_pos_weight=positive_weight, eval_metric="logloss",
		random_state=SEED, n_jobs=-1,
	)
	model_parameters["xgboost"] = {
		"n_estimators": args.trees,
		"max_depth": 5,
		"learning_rate": 0.05,
		"min_child_weight": 5,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"scale_pos_weight": positive_weight,
	}
	results = []
	training_curves = {}
	validation_map = validation_panel[
		["cell_id", "date", "latitude", "longitude", "target"]
	].copy()
	for name, model in models.items():
		if name == "xgboost":
			model.fit(
				train_x,
				train_y,
				eval_set=[(validation_x, validation_y)],
				verbose=False,
			)
		else:
			model.fit(train_x, train_y)
		if name == "random_forest":
			training_curves[name] = {
				"steps": list(range(1, args.trees + 1)),
				"values": [
					roc_auc_score(
						validation_y,
						np.mean(
							[
								estimator.predict_proba(validation_x)[:, 1]
								for estimator in model.estimators_[:step]
							],
							axis=0,
						),
					)
					for step in range(1, args.trees + 1)
				],
			}
		elif name == "xgboost":
			evaluation_history = model.evals_result()
			training_curves[name] = {
				"steps": list(range(1, len(evaluation_history["validation_0"]["logloss"]) + 1)),
				"values": [
					roc_auc_score(
						validation_y,
						model.predict_proba(validation_x, iteration_range=(0, step))[:, 1],
					)
					for step in range(1, args.trees + 1)
				],
			}
		result, y, probability, prediction = evaluate(name, model, validation_x, validation_y, output)
		result["parameters"] = model_parameters[name]
		results.append(result)
		validation_map[f"{name}_probability"] = model.predict_proba(validation_x)[:, 1]
		validation_map[f"{name}_prediction"] = prediction
		plot_validation(name, y, probability, prediction, output / "plots")
		joblib.dump(model, output / f"{name}_future.joblib")

	try:
		import tensorflow as tf
	except ImportError as error:
		raise SystemExit("Install tensorflow to run the three-model comparison.") from error
	lstm_train = build_lstm_sequences(train_panel, args.sequence_length, args.max_lstm_samples, SEED)
	lstm_validation = build_lstm_sequences(validation_panel, args.sequence_length, args.max_lstm_samples, SEED)
	lstm_test = build_lstm_sequences(test_panel, args.sequence_length, args.max_lstm_samples, SEED)
	lstm = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(args.sequence_length, len(FEATURES))),
		tf.keras.layers.LSTM(args.units, dropout=args.dropout, recurrent_dropout=0.0),
		tf.keras.layers.Dense(1, activation="sigmoid"),
	])
	lstm.compile(
		optimizer=tf.keras.optimizers.Adam(args.learning_rate),
		loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")],
	)
	counts = np.bincount(lstm_train[1], minlength=2)
	weights = {i: len(lstm_train[1]) / (2 * count) for i, count in enumerate(counts) if count}
	lstm.fit(
		lstm_train[0], lstm_train[1],
		validation_data=(lstm_validation[0], lstm_validation[1]),
		epochs=args.epochs, batch_size=args.batch_size, class_weight=weights,
		shuffle=False, verbose=2,
		callbacks=[tf.keras.callbacks.EarlyStopping(
			monitor="val_auc", mode="max", patience=args.patience, restore_best_weights=True,
		)],
	)
	lstm_history = lstm.history.history
	training_curves["lstm"] = {
		"steps": list(range(1, len(lstm_history["val_auc"]) + 1)),
		"values": lstm_history["val_auc"],
	}
	lstm_validation_probability = lstm.predict(lstm_validation[0], verbose=0).ravel()
	lstm_threshold = choose_threshold(lstm_validation[1], lstm_validation_probability)
	lstm_prediction = (lstm_validation_probability >= lstm_threshold).astype("int8")
	lstm_map = lstm_validation[2].copy()
	lstm_map["target"] = lstm_validation[1]
	lstm_map["lstm_probability"] = lstm_validation_probability
	lstm_map["lstm_prediction"] = lstm_prediction
	validation_map = validation_map.merge(
		lstm_map[
			["cell_id", "date", "lstm_probability", "lstm_prediction"]
		],
		on=["cell_id", "date"],
		how="left",
	)
	lstm_report = classification_report(
		lstm_validation[1], lstm_prediction, output_dict=True, zero_division=0,
	)
	results.append({
		"model": "lstm",
		"parameters": {
			"sequence_length": args.sequence_length,
			"units": args.units,
			"dropout": args.dropout,
			"learning_rate": args.learning_rate,
			"epochs_max": args.epochs,
			"early_stopping_patience": args.patience,
			"class_weight": "balanced",
		},
		"threshold": lstm_threshold,
		"roc_auc": roc_auc_score(lstm_validation[1], lstm_validation_probability),
		"average_precision": average_precision_score(lstm_validation[1], lstm_validation_probability),
		"confusion_matrix": confusion_matrix(lstm_validation[1], lstm_prediction, labels=[0, 1]).tolist(),
		"precision_class_1": lstm_report["1"]["precision"],
		"recall_class_1": lstm_report["1"]["recall"],
		"f1_class_1": lstm_report["1"]["f1-score"],
	})
	plot_validation(
		"lstm", lstm_validation[1], lstm_validation_probability,
		lstm_prediction, output / "plots",
	)
	plot_training_curves(training_curves, output / "plots")
	lstm.save(output / "lstm_future.keras")
	(output / "future_fire_validation_metrics.json").write_text(json.dumps(results, indent=2))
	validation_map.to_csv(output / "tableau_validation_grid_risk.csv", index=False)
	pd.DataFrame(results).drop(columns="confusion_matrix").to_csv(
		output / "future_fire_validation_metrics.csv", index=False,
	)
	print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="modis_2024_Indonesia.csv")
	parser.add_argument("--output", default="prepared_data")
	parser.add_argument("--grid-size", type=float, default=0.5)
	parser.add_argument("--sequence-length", type=int, default=7)
	parser.add_argument("--trees", type=int, default=250)
	parser.add_argument("--units", type=int, default=32)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--learning-rate", type=float, default=5e-4)
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--max-lstm-samples", type=int, default=120_000)
	train(parser.parse_args())
