from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

DATA_DIR = Path("infos/mlops_biomass_data")
IMAGE_DIR = DATA_DIR / "images_med_res"
LABELS_PATH = DATA_DIR / "digital_biomass_labels.xlsx"
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")


def _ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Could not find {LABELS_PATH}")
    df = pd.read_excel(LABELS_PATH)

    # converts the timestamp column from Unix seconds to proper pandas datetime64, and errors="coerce" silently turns any invalid entries into NaT so downstream code can rely on consistent datetime types without crashing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    return df


def save_target_distribution(df: pd.DataFrame) -> None:
    # remove NaN values for plotting
    target = df["fresh_weight_total"].dropna()

    plt.figure(figsize=(8, 6))
    sns.histplot(target, bins=40, kde=True, color="#2c7fb8")
    plt.title("Fresh Biomass Distribution")
    plt.xlabel("fresh_weight_total [g]")
    plt.ylabel("Plants count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close()


def save_sample_images(df: pd.DataFrame) -> None:
    labeled = df.dropna(subset=["fresh_weight_total"])
    sample_size = min(9, len(labeled))
    if sample_size == 0:
        return
    rng = np.random.default_rng(seed=42)
    sample = labeled.iloc[rng.choice(len(labeled), size=sample_size, replace=False)]
    grid_size = math.ceil(sample_size ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = np.atleast_1d(axes).flatten()
    for ax, (_, row) in zip(axes, sample.iterrows()):
        image_path = IMAGE_DIR / row["filename"]
        if not image_path.exists():
            ax.axis("off")
            continue
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{row['fresh_weight_total']:.1f} g")
        ax.axis("off")
    for ax in axes[len(sample) :]:
        ax.axis("off")
    plt.suptitle("Sample Training Images with Biomass Labels")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(FIGURES_DIR / "sample_images.png", dpi=200)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    candidate_columns = [
        "temperature",
        "humidity",
        "illuminancelux",
        "illuminanceinfrared",
        "illuminancevisible",
        "illuminancefullspectrum",
        "age_days",
        "small_leaves",
        "big_leaves",
        "total_leaves",
        "plant_surface_area",
        "leaf_surface_area",
        "shoot_root_ratio_fresh",
        "experiment_id", # it is not a numeric variable per se, but including it to see if any correlation exists
    ]
    available = [col for col in candidate_columns if col in df.columns]
    data = df[available + ["fresh_weight_total"]]
    corr = data.corr(numeric_only=True)["fresh_weight_total"].dropna()
    corr = corr.sort_values()
    plt.figure(figsize=(6, max(4, len(corr) * 0.4)))
    sns.heatmap(
        corr.to_frame(name="corr"),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar=True,
        fmt=".2f",
    )
    plt.title("Correlation with Fresh Biomass")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_age_vs_biomass(df: pd.DataFrame) -> None:
    subset = df[["age_days", "fresh_weight_total"]].dropna()
    if subset.empty:
        return
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=subset,
        x="age_days",
        y="fresh_weight_total",
        hue="age_days",
        palette="viridis",
        alpha=0.6,
        edgecolor="none",
    )
    sns.regplot(
        data=subset,
        x="age_days",
        y="fresh_weight_total",
        scatter=False,
        color="black",
        line_kws={"linewidth": 1.5, "alpha": 0.7},
    )
    plt.title("Biomass vs Plant Age")
    plt.xlabel("age_days")
    plt.ylabel("fresh_weight_total [g]")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "age_vs_biomass.png", dpi=200)
    plt.close()


def save_pixel_analysis(df: pd.DataFrame, max_images: int = 500) -> None:
    filenames = df["filename"].dropna().unique().tolist()
    if not filenames:
        return
    random.seed(42)
    random.shuffle(filenames)
    selected = filenames[: max_images or len(filenames)]
    channel_means: List[List[float]] = []
    for name in selected:
        image_path = IMAGE_DIR / name
        if not image_path.exists():
            continue
        with Image.open(image_path).convert("RGB") as image:
            arr = np.asarray(image, dtype=np.float32) / 255.0
            mean_rgb = arr.reshape(-1, 3).mean(axis=0)
            channel_means.append(mean_rgb.tolist())
    if not channel_means:
        return
    samples = pd.DataFrame(channel_means, columns=["R", "G", "B"])
    melted = samples.melt(var_name="channel", value_name="mean_intensity")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=melted,
        x="mean_intensity",
        hue="channel",
        fill=True,
        common_norm=False,
        alpha=0.35,
        palette={"R": "red", "G": "green", "B": "blue"},
    )
    plt.title("Distribution of Mean RGB Intensities per Image")
    plt.xlabel("Mean channel intensity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_pixel_analysis.png", dpi=200)
    plt.close()


def collect_summary(df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary["num_records"] = int(len(df))
    summary["num_images_on_disk"] = len(list(IMAGE_DIR.glob("*.png")))
    target = df["fresh_weight_total"]
    summary["num_labeled"] = int(target.notna().sum())
    summary["pct_labeled"] = float(target.notna().mean())
    summary["fresh_weight_stats"] = {
        k: (float(v) if not math.isnan(v) else None)
        for k, v in target.describe().to_dict().items()
    }
    summary["age_missing_pct"] = float(df["age_days"].isna().mean())
    summary["target_missing_pct"] = float(target.isna().mean())
    plant_counts = df["plant_number"].value_counts()
    summary["plants_with_multiple_samples"] = int((plant_counts > 1).sum())
    summary["max_images_per_plant"] = int(plant_counts.max()) if not plant_counts.empty else 0
    summary["timestamp_span_days"] = (
        (df["timestamp"].max() - df["timestamp"].min()).days
        if df["timestamp"].notna().any()
        else None
    )
    issues: List[str] = []
    if summary["target_missing_pct"] > 0.1:
        issues.append(
            "Roughly 14% of the rows have no fresh_weight_total label, so a notable part of the imagery cannot be used for supervised training."
        )
    if summary["age_missing_pct"] > 0.1:
        issues.append("age_days is missing for more than 10% of entries, limiting the reliability of temporal analysis.")
    if summary["max_images_per_plant"] > 5:
        issues.append(
            "Multiple images per plant exist, so random train/test splits risk leakage without grouping by plant_number."
        )
    summary["data_quality_issues"] = issues
    return summary


def save_summary(summary: Dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def main() -> None:
    _ensure_dirs()
    df = load_dataset()
    save_target_distribution(df)
    save_sample_images(df)
    save_correlation_heatmap(df)
    save_age_vs_biomass(df)
    save_pixel_analysis(df)
    summary = collect_summary(df)
    save_summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
