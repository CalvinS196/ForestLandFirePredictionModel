"""Create a static exploratory dashboard for the raw MODIS fire data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_dashboard(data_path: str, output_path: str, grid_size: float) -> None:
	data = pd.read_csv(data_path)
	data["date"] = pd.to_datetime(data["acq_date"], errors="coerce")
	data["month"] = data["date"].dt.month
	data["month_name"] = data["date"].dt.strftime("%b")
	data["acq_time_text"] = data["acq_time"].astype(str).str.zfill(4)
	data["hour"] = pd.to_numeric(data["acq_time_text"].str[:2], errors="coerce")
	data["grid_lat"] = np.floor(data["latitude"] / grid_size).astype(int)
	data["grid_lon"] = np.floor(data["longitude"] / grid_size).astype(int)

	grid = data.groupby(["grid_lat", "grid_lon"], as_index=False).agg(
		latitude=("latitude", "mean"),
		longitude=("longitude", "mean"),
		mean_frp=("frp", "mean"),
		detections=("frp", "size"),
	)
	monthly = data.groupby(["month", "month_name"], as_index=False).size()
	monthly = monthly.sort_values("month")
	daynight = data["daynight"].value_counts().rename(index={"D": "Day", "N": "Night"})
	type_counts = data["type"].value_counts().sort_index()
	correlation_columns = [
		"brightness", "scan", "track", "confidence",
		"bright_t31", "frp", "latitude", "longitude",
	]
	correlation = data[correlation_columns].corr()

	sns.set_theme(style="whitegrid")
	figure, axes = plt.subplots(3, 2, figsize=(18, 19), constrained_layout=True)
	figure.suptitle("MODIS Indonesia Fire-Detection Data Understanding", fontsize=18, fontweight="bold")

	map_axis = axes[0, 0]
	points = map_axis.scatter(
		grid["longitude"], grid["latitude"], c=grid["mean_frp"],
		cmap="YlOrRd", s=18, alpha=0.85, edgecolors="none",
	)
	map_axis.set_title("Mean FRP by 0.5° grid cell")
	map_axis.set_xlabel("Longitude")
	map_axis.set_ylabel("Latitude")
	map_axis.set_xlim(95, 141)
	map_axis.set_ylim(-11, 6)
	map_axis.set_aspect("equal", adjustable="box")
	figure.colorbar(points, ax=map_axis, label="Mean FRP")

	monthly_axis = axes[0, 1]
	monthly_axis.bar(monthly["month_name"], monthly["size"], color="black")
	monthly_axis.set_title("Detections by month")
	monthly_axis.set_xlabel("Month")
	monthly_axis.set_ylabel("Number of detections")

	frp_axis = axes[1, 0]
	frp_axis.hist(data["frp"].clip(upper=data["frp"].quantile(0.99)), bins=40, color="black")
	frp_axis.set_title("FRP distribution (99th percentile clipped)")
	frp_axis.set_xlabel("FRP")
	frp_axis.set_ylabel("Frequency")

	confidence_axis = axes[1, 1]
	confidence_axis.hist(data["confidence"], bins=20, color="dimgray", edgecolor="black")
	confidence_axis.axvline(50, color="black", linestyle="--", label="Confidence = 50")
	confidence_axis.set_title("Confidence distribution")
	confidence_axis.set_xlabel("Confidence")
	confidence_axis.set_ylabel("Frequency")
	confidence_axis.legend()

	composition_axis = axes[2, 0]
	daynight.plot.bar(ax=composition_axis, color="black")
	composition_axis.set_title("Day/night detection composition")
	composition_axis.set_xlabel("")
	composition_axis.set_ylabel("Number of detections")
	composition_axis.tick_params(axis="x", rotation=0)

	heatmap_axis = axes[2, 1]
	sns.heatmap(correlation, cmap="Greys", center=0, vmin=-1, vmax=1, ax=heatmap_axis)
	heatmap_axis.set_title("Numeric feature correlation")

	figure.text(
		0.5, 0.005,
		f"Rows: {len(data):,} | Grid cells: {len(grid):,} | "
		f"Date range: {data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d} | "
		f"Fire type counts: {type_counts.to_dict()}",
		ha="center", fontsize=9,
	)
	Path(output_path).parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)

	individual_dir = Path(output_path).parent
	map_figure, map_axis = plt.subplots(figsize=(9, 7))
	points = map_axis.scatter(
		grid["longitude"], grid["latitude"], c=grid["mean_frp"],
		cmap="YlOrRd", s=24, alpha=0.85, edgecolors="none",
	)
	map_axis.set_title("Mean FRP by 0.5° Grid Cell")
	map_axis.set_xlabel("Longitude")
	map_axis.set_ylabel("Latitude")
	map_axis.set_xlim(95, 141)
	map_axis.set_ylim(-11, 6)
	map_axis.set_aspect("equal", adjustable="box")
	map_figure.colorbar(points, ax=map_axis, label="Mean FRP")
	map_figure.tight_layout()
	map_figure.savefig(individual_dir / "01_mean_frp_spatial_map.png", dpi=180)
	plt.close(map_figure)

	temporal_figure, temporal_axis = plt.subplots(figsize=(9, 6))
	temporal_axis.bar(monthly["month_name"], monthly["size"], color="black")
	temporal_axis.set_title("MODIS Fire Detections by Month")
	temporal_axis.set_xlabel("Month")
	temporal_axis.set_ylabel("Number of detections")
	temporal_figure.tight_layout()
	temporal_figure.savefig(individual_dir / "02_monthly_detection_pattern.png", dpi=180)
	plt.close(temporal_figure)

	distribution_figure, distribution_axis = plt.subplots(figsize=(9, 6))
	distribution_axis.hist(
		data["frp"].clip(upper=data["frp"].quantile(0.99)),
		bins=40,
		color="black",
	)
	distribution_axis.set_title("FRP Distribution (99th Percentile Clipped)")
	distribution_axis.set_xlabel("FRP")
	distribution_axis.set_ylabel("Frequency")
	distribution_figure.tight_layout()
	distribution_figure.savefig(individual_dir / "03_frp_distribution.png", dpi=180)
	plt.close(distribution_figure)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", default="modis_2024_Indonesia.csv")
	parser.add_argument(
		"--output",
		default="prepared_data/plots/data_understanding_dashboard.png",
	)
	parser.add_argument("--grid-size", type=float, default=0.5)
	args = parser.parse_args()
	create_dashboard(args.data, args.output, args.grid_size)
	print(f"Saved visualization to {args.output}")
