"""Create a static map dashboard from the Tableau-ready grid-risk CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_COLUMNS = {
	"Actual next-day detection rate": "target",
	"Random Forest risk": "random_forest_probability",
	"XGBoost risk": "xgboost_probability",
	"LSTM risk": "lstm_probability",
}


def create_dashboard(csv_path: str, output_path: str) -> None:
	data = pd.read_csv(csv_path, parse_dates=["date"])
	aggregations = data.groupby(
		["cell_id", "latitude", "longitude"], as_index=False
	).agg(
		target=("target", "mean"),
		random_forest_probability=("random_forest_probability", "mean"),
		xgboost_probability=("xgboost_probability", "mean"),
		lstm_probability=("lstm_probability", "mean"),
	)
	figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
	figure.suptitle(
		"Indonesia Next-Day Fire-Detection Risk by Grid Cell",
		fontsize=17,
		fontweight="bold",
	)
	for axis, (title, column) in zip(axes.ravel(), MODEL_COLUMNS.items()):
		mapped = aggregations.dropna(subset=[column])
		points = axis.scatter(
			mapped["longitude"],
			mapped["latitude"],
			c=mapped[column],
			cmap="YlOrRd",
			vmin=0,
			vmax=1,
			s=20,
			alpha=0.85,
			edgecolors="none",
		)
		axis.set_title(title)
		axis.set_xlabel("Longitude")
		axis.set_ylabel("Latitude")
		axis.set_xlim(95, 141)
		axis.set_ylim(-11, 6)
		axis.grid(True, color="0.85", linewidth=0.6)
		axis.set_aspect("equal", adjustable="box")
		figure.colorbar(points, ax=axis, label="Rate / predicted probability")
	axis_note = (
		"Each point is a 0.5° grid cell. Values are averaged over the validation period. "
		"LSTM values are shown only where a complete 7-day sequence exists."
	)
	figure.text(0.5, 0.01, axis_note, ha="center", fontsize=9)
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--csv",
		default="prepared_data/tableau_validation_grid_risk.csv",
	)
	parser.add_argument(
		"--output",
		default="prepared_data/plots/indonesia_risk_dashboard.png",
	)
	args = parser.parse_args()
	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	create_dashboard(args.csv, args.output)
	print(f"Saved visualization to {args.output}")
