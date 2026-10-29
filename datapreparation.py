"""Prepare modis_2024_Indonesia.csv for wildfire classification models.

Example:
	python datapreparation.py --data modis_2024_Indonesia.csv

Outputs are written to ``prepared_data``: EDA plots, engineered CSV, fitted
preprocessor, and chronological tabular splits. Train models separately with
``train_models.py`` or ``train_lstm.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def find_column(frame, names):
	normalized = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in frame.columns}
	for name in names:
		key = re.sub(r"[^a-z0-9]", "", name.lower())
		if key in normalized:
			return normalized[key]
	return None


def target_column(frame, requested, threshold):
	if requested and requested in frame:
		values = frame[requested]
		return (pd.to_numeric(values, errors="coerce").fillna(0) > 0).astype("int8"), requested
	name = find_column(frame, ["fire", "wildfire", "is_fire", "target", "label", "occurrence"])
	if name:
		values = frame[name]
		if values.dtype == object:
			return values.astype(str).str.lower().isin(["1", "true", "yes", "fire"]).astype("int8"), name
		return (pd.to_numeric(values, errors="coerce").fillna(0) > 0).astype("int8"), name
	name = find_column(frame, ["confidence", "conf"])
	if name:
		values = pd.to_numeric(frame[name].astype(str).str.rstrip("%"), errors="coerce")
		return (values >= threshold).astype("int8"), name
	raise ValueError("No target found. Use --target-column or add a fire/non-fire label.")


def engineer(frame):
	data = frame.copy()
	date_name = find_column(data, ["acq_date", "acw_date", "acquisition_date", "date"])
	time_name = find_column(data, ["acq_time", "acg_time", "acquisition_time", "time"])
	dates = pd.to_datetime(data[date_name], errors="coerce") if date_name else pd.Series(pd.NaT, index=data.index)
	if time_name:
		raw_time = data[time_name].astype(str).str.extract(r"(\d{1,4})", expand=False).fillna("0").str.zfill(4)
		offsets = pd.to_timedelta(raw_time.str[:2] + ":" + raw_time.str[2:] + ":00", errors="coerce")
		data["acq_datetime"] = dates.dt.normalize() + offsets
	else:
		data["acq_datetime"] = dates
	dt = pd.to_datetime(data["acq_datetime"], errors="coerce")
	data["year"] = dt.dt.year
	data["month"] = dt.dt.month
	data["day_of_year"] = dt.dt.dayofyear
	data["day_of_week"] = dt.dt.dayofweek
	data["hour"] = dt.dt.hour
	data["month_sin"] = np.sin(2 * np.pi * data["month"].fillna(1) / 12)
	data["month_cos"] = np.cos(2 * np.pi * data["month"].fillna(1) / 12)
	data["hour_sin"] = np.sin(2 * np.pi * data["hour"].fillna(0) / 24)
	data["hour_cos"] = np.cos(2 * np.pi * data["hour"].fillna(0) / 24)
	lat_name = find_column(data, ["latitude", "lat"])
	lon_name = find_column(data, ["longitude", "lon", "long"])
	if not lat_name or not lon_name:
		raise ValueError("Latitude and longitude columns are required.")
	data["latitude"] = pd.to_numeric(data[lat_name], errors="coerce")
	data["longitude"] = pd.to_numeric(data[lon_name], errors="coerce")
	data["spatial_cell"] = data["latitude"].round(1).astype(str) + "_" + data["longitude"].round(1).astype(str)
	return data.dropna(subset=["acq_datetime", "latitude", "longitude"])


def make_plots(data, output):
	try:
		import matplotlib.pyplot as plt
		import importlib
		sns = importlib.import_module("seaborn")
	except ImportError:
		print("Install matplotlib and seaborn for EDA plots.")
		return
	output.mkdir(parents=True, exist_ok=True)
	sns.set_theme(style="whitegrid")
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	sns.countplot(data=data, x="target", ax=axes[0, 0]); axes[0, 0].set_title("Target balance")
	sns.countplot(data=data, x="month", hue="target", ax=axes[0, 1]); axes[0, 1].set_title("Wildfire by month")
	sns.countplot(data=data, x="hour", hue="target", ax=axes[1, 0]); axes[1, 0].set_title("Wildfire by hour")
	sample = data.sample(min(len(data), 20000), random_state=42)
	sns.scatterplot(data=sample, x="longitude", y="latitude", hue="target", s=10, ax=axes[1, 1]); axes[1, 1].set_title("Spatial distribution")
	fig.tight_layout(); fig.savefig(output / "eda_patterns.png", dpi=160); plt.close(fig)
	numeric = data.select_dtypes(include=np.number)
	plt.figure(figsize=(12, 9)); sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
	plt.tight_layout(); plt.savefig(output / "eda_correlation.png", dpi=160); plt.close()


def chronological_split(data):
	data = data.sort_values("acq_datetime").reset_index(drop=True)
	first, second = int(len(data) * .70), int(len(data) * .85)
	return data.iloc[:first], data.iloc[first:second], data.iloc[second:]


def build_preprocessor(train, source_target):
	# Raw date/time values are identifiers rather than stable predictive
	# features. The engineered calendar fields below retain their signal
	# without allowing the model to memorize individual dates.
	excluded = {
		"target", "acq_datetime", "spatial_cell", source_target,
		"acq_date", "acq_time",
	}
	columns = [c for c in train.columns if c not in excluded]
	categorical = train[columns].select_dtypes(include=["object", "category"]).columns.tolist()
	numerical = [c for c in columns if c not in categorical]
	encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	prep = ColumnTransformer([
		("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numerical),
		("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)]), categorical),
	])
	return prep, columns


def main(args):
	output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
	raw = pd.read_csv(args.data)
	labels, source = target_column(raw, args.target_column, args.confidence_threshold)
	raw["_label"] = labels
	data = engineer(raw); data["target"] = data["_label"].astype("int8"); data.drop(columns="_label", inplace=True)
	make_plots(data, output / "plots")
	train, validation, test = chronological_split(data)
	if train["target"].nunique() < 2:
		raise ValueError("Training split has one class; provide real positive and negative observations.")
	prep, columns = build_preprocessor(train, source)
	x_train = prep.fit_transform(train[columns]); x_val = prep.transform(validation[columns]); x_test = prep.transform(test[columns])
	y_train, y_val, y_test = train.target.to_numpy(), validation.target.to_numpy(), test.target.to_numpy()
	joblib.dump(prep, output / "preprocessor.joblib")
	np.savez_compressed(
		output / "tabular_splits.npz",
		X_train=x_train,
		X_validation=x_val,
		X_test=x_test,
		y_train=y_train,
		y_validation=y_val,
		y_test=y_test,
		groups_train=np.asarray(train["spatial_cell"], dtype="U"),
		groups_validation=np.asarray(validation["spatial_cell"], dtype="U"),
		groups_test=np.asarray(test["spatial_cell"], dtype="U"),
	)
	data.to_csv(output / "engineered_modis.csv", index=False)
	(output / "feature_info.json").write_text(json.dumps({"features": columns, "target_source": source, "split": "70/15/15 chronological"}, indent=2))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="modis_2024_Indonesia.csv")
	parser.add_argument("--output", default="prepared_data")
	parser.add_argument("--target-column")
	parser.add_argument("--confidence-threshold", type=float, default=50)
	main(parser.parse_args())
