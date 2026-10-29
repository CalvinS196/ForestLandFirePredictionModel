"""Train a binary LSTM from artifacts created by datapreparation.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
	average_precision_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_recall_curve,
	roc_auc_score,
	roc_curve,
)


def sequences(x, y, length):
	if len(x) <= length:
		return np.empty((0, length, x.shape[1])), np.empty((0,), dtype="int8")
	return np.asarray([x[i - length:i] for i in range(length, len(x))]), np.asarray(y[length:])


def grouped_sequences(x, y, groups, length):
	"""Create windows within each spatial cell, never across cell boundaries."""
	features = []
	labels = []
	for group in np.unique(groups):
		indices = np.flatnonzero(groups == group)
		if len(indices) <= length:
			continue
		for current in range(length, len(indices)):
			features.append(x[indices[current - length:current]])
			labels.append(y[indices[current]])
	if not features:
		return np.empty((0, length, x.shape[1]), dtype=x.dtype), np.empty((0,), dtype=y.dtype)
	return np.asarray(features, dtype=x.dtype), np.asarray(labels, dtype=y.dtype)


def make_sequences(split, prefix, length):
	"""Prefer spatially grouped windows and support older prepared artifacts."""
	group_key = f"groups_{prefix}"
	if group_key in split:
		return grouped_sequences(split[f"X_{prefix}"], split[f"y_{prefix}"], split[group_key], length)
	return sequences(split[f"X_{prefix}"], split[f"y_{prefix}"], length)


def validation_threshold(y_true, probability):
	"""Choose the threshold with the best validation F1 without touching test labels."""
	precision, recall, thresholds = precision_recall_curve(y_true, probability)
	if not len(thresholds):
		return 0.5
	f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
	return float(thresholds[int(np.nanargmax(f1))])


def plot_evaluation_curves(y_true, probability, output):
	"""Save ROC and Precision-Recall plots for binary predictions."""
	import matplotlib.pyplot as plt

	y_true = np.asarray(y_true)
	probability = np.asarray(probability)
	if len(np.unique(y_true)) != 2:
		raise ValueError("ROC and Precision-Recall curves require both target classes.")
	output = Path(output)
	output.mkdir(parents=True, exist_ok=True)

	fpr, tpr, _ = roc_curve(y_true, probability)
	roc_auc = roc_auc_score(y_true, probability)
	precision, recall, _ = precision_recall_curve(y_true, probability)
	average_precision = average_precision_score(y_true, probability)

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	axes[0].plot(fpr, tpr, label=f"LSTM (AUC = {roc_auc:.3f})")
	axes[0].plot([0, 1], [0, 1], "k--", label="Random")
	axes[0].set_title("ROC curve")
	axes[0].set_xlabel("False positive rate")
	axes[0].set_ylabel("True positive rate")
	axes[0].legend(loc="lower right")
	axes[0].grid(alpha=.3)

	axes[1].plot(recall, precision, label=f"LSTM (AP = {average_precision:.3f})")
	axes[1].set_title("Precision-Recall curve")
	axes[1].set_xlabel("Recall")
	axes[1].set_ylabel("Precision")
	axes[1].legend(loc="lower left")
	axes[1].grid(alpha=.3)

	fig.tight_layout()
	fig.savefig(output / "lstm_evaluation_curves.png", dpi=160)
	plt.close(fig)


def plot_confusion_matrix(y_true, prediction, probability, output):
	"""Save a confusion matrix for binary predictions."""
	import matplotlib.pyplot as plt

	output = Path(output)
	output.mkdir(parents=True, exist_ok=True)
	matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
	f1 = f1_score(y_true, prediction, pos_label=1, zero_division=0)
	roc_auc = (
		roc_auc_score(y_true, probability)
		if len(np.unique(y_true)) == 2
		else None
	)
	fig, axis = plt.subplots(figsize=(6, 6))
	image = axis.imshow(matrix, cmap="Blues")
	fig.colorbar(image, ax=axis)
	axis.set(
		xticks=[0, 1],
		yticks=[0, 1],
		xticklabels=["Non-fire (0)", "Fire (1)"],
		yticklabels=["Non-fire (0)", "Fire (1)"],
		xlabel="Predicted label",
		ylabel="True label",
		title="LSTM confusion matrix",
	)
	threshold = matrix.max() / 2
	for row in range(matrix.shape[0]):
		for column in range(matrix.shape[1]):
			axis.text(
				column,
				row,
				str(matrix[row, column]),
				ha="center",
				va="center",
				color="white" if matrix[row, column] > threshold else "black",
			)
	metric_text = f"Fire F1-score: {f1:.3f}"
	if roc_auc is not None:
		metric_text += f"    ROC-AUC: {roc_auc:.3f}"
	fig.text(
		0.5,
		0.02,
		metric_text,
		ha="center",
		va="bottom",
		fontsize=11,
		bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round,pad=0.4"},
	)
	fig.tight_layout(rect=(0, 0.07, 1, 1))
	fig.savefig(output / "lstm_confusion_matrix.png", dpi=160)
	plt.close(fig)


def plot_training_history(history, output):
	"""Save training and validation loss and AUC curves."""
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	epochs = range(1, len(history["loss"]) + 1)
	axes[0].plot(epochs, history["loss"], label="Training loss")
	axes[0].plot(epochs, history["val_loss"], label="Validation loss")
	axes[0].set_title("Training and validation loss")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Binary cross-entropy")
	axes[0].legend()
	axes[0].grid(alpha=.3)

	metric = "auc" if "auc" in history and "val_auc" in history else "accuracy"
	axes[1].plot(epochs, history[metric], label=f"Training {metric.upper()}")
	axes[1].plot(epochs, history[f"val_{metric}"], label=f"Validation {metric.upper()}")
	axes[1].set_title(f"Training and validation {metric.upper()}")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel(metric.upper())
	axes[1].legend()
	axes[1].grid(alpha=.3)

	fig.tight_layout()
	fig.savefig(output / "lstm_training_history.png", dpi=160)
	plt.close(fig)


def train(
	output,
	sequence_length,
	epochs,
	batch_size,
	units,
	dropout,
	learning_rate,
	patience,
	balance_classes,
):
	try:
		import tensorflow as tf  # type: ignore[reportMissingImports, reportMissingModuleSource]
	except ImportError as error:
		raise SystemExit("Install tensorflow to train the LSTM model.") from error
	output = Path(output)
	splits = np.load(output / "tabular_splits.npz")
	train_data = make_sequences(splits, "train", sequence_length)
	validation_data = make_sequences(splits, "validation", sequence_length)
	test_data = make_sequences(splits, "test", sequence_length)
	if not all(len(data[0]) for data in (train_data, validation_data, test_data)):
		raise ValueError("A split has too few rows for the selected sequence length.")
	x_train, y_train = train_data
	x_validation, y_validation = validation_data
	x_test, y_test = test_data
	tf.random.set_seed(42)
	class_weight = None
	if balance_classes:
		counts = np.bincount(y_train.astype("int64"), minlength=2)
		total = counts.sum()
		class_weight = {i: total / (2 * count) for i, count in enumerate(counts) if count}
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(sequence_length, x_train.shape[2])),
		tf.keras.layers.LSTM(
			units,
			kernel_regularizer=tf.keras.regularizers.l2(1e-4),
			recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
		),
		tf.keras.layers.Dropout(max(dropout, 0.3)),
		tf.keras.layers.Dense(
			max(units // 2, 8),
			activation="relu",
			kernel_regularizer=tf.keras.regularizers.l2(1e-4),
		),
		tf.keras.layers.Dropout(max(dropout, 0.3)),
		tf.keras.layers.Dense(1, activation="sigmoid"),
	])
	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="binary_crossentropy",
		metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
	)
	history = model.fit(
		x_train, y_train, validation_data=(x_validation, y_validation),
		epochs=epochs, batch_size=batch_size, class_weight=class_weight,
		callbacks=[
			tf.keras.callbacks.ReduceLROnPlateau(
				monitor="val_auc", mode="max", factor=0.5,
				patience=max(2, patience // 2), min_lr=1e-5,
			),
			tf.keras.callbacks.EarlyStopping(
				monitor="val_auc", mode="max", patience=patience,
				restore_best_weights=True,
			),
		],
		shuffle=False, verbose=2,
	)
	probability = model.predict(x_test, verbose=0).ravel()
	validation_probability = model.predict(x_validation, verbose=0).ravel()
	threshold = validation_threshold(y_validation, validation_probability)
	prediction = (probability >= threshold).astype("int8")
	print("\nLSTM test\n", classification_report(y_test, prediction, zero_division=0))
	print("Validation-selected threshold:", round(threshold, 4))
	plots = output / "plots"
	plots.mkdir(parents=True, exist_ok=True)
	plot_confusion_matrix(y_test, prediction, probability, plots)
	plot_training_history(history.history, plots)
	if len(np.unique(y_test)) == 2:
		print("ROC-AUC:", round(roc_auc_score(y_test, probability), 4))
		plot_evaluation_curves(y_test, probability, plots)
	model.save(output / "lstm.keras")
	(output / "lstm_history.json").write_text(
		json.dumps({**history.history, "decision_threshold": threshold}, indent=2)
	)
	np.savez_compressed(
		output / "lstm_sequences.npz",
		X_train=x_train, X_validation=x_validation, X_test=x_test,
		y_train=y_train, y_validation=y_validation, y_test=y_test,
		sequence_length=sequence_length,
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", default="prepared_data")
	parser.add_argument("--sequence-length", type=int, default=7)
	parser.add_argument("--epochs", type=int, default=40)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--units", type=int, default=64)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument(
		"--balance-classes",
		action="store_true",
		help="Apply inverse-frequency class weights; disabled by default for calibrated validation loss.",
	)
	args = parser.parse_args()
	train(
		args.output,
		args.sequence_length,
		args.epochs,
		args.batch_size,
		args.units,
		args.dropout,
		args.learning_rate,
		args.patience,
		args.balance_classes,
	)
