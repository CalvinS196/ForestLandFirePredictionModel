"""Train tabular models from artifacts created by datapreparation.py."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline


def report(model, x, y, title):
	prediction = model.predict(x)
	print(f"\n{title}\n", classification_report(y, prediction, zero_division=0))
	if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
		print("ROC-AUC:", round(roc_auc_score(y, model.predict_proba(x)[:, 1]), 4))


def train(output):
	output = Path(output)
	splits = np.load(output / "tabular_splits.npz")
	x_train, x_validation, x_test = splits["X_train"], splits["X_validation"], splits["X_test"]
	y_train, y_validation, y_test = splits["y_train"], splits["y_validation"], splits["y_test"]

	models = {
		"random_forest": Pipeline([("model", RandomForestClassifier(
			n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
		))]),
	}
	try:
		XGBClassifier = import_module("xgboost").XGBClassifier
	except ImportError:
		print("Install xgboost to train the XGBoost model.")
	else:
		models["xgboost"] = XGBClassifier(
			n_estimators=300, max_depth=6, learning_rate=.05,
			subsample=.8, colsample_bytree=.8, eval_metric="logloss",
			random_state=42,
		)
	for name, model in models.items():
		model.fit(x_train, y_train)
		report(model, x_validation, y_validation, name + " validation")
		report(model, x_test, y_test, name + " test")
		joblib.dump(model, output / f"{name}.joblib")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", default="prepared_data")
	train(parser.parse_args().output)
