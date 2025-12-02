# Exploratory Data Analysis (EDA) für Biomasse-Vorhersage
# Dieses Skript führt eine umfassende EDA durch:
# - Datenqualität prüfen (fehlende Werte, Outliers)
# - Verteilungen visualisieren
# - Korrelationen analysieren
# - Potenzielle Probleme identifizieren (z.B. Data Leakage)

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

# Pfade zu Daten und Ausgabeverzeichnissen
DATA_DIR = Path("infos/mlops_biomass_data")
IMAGE_DIR = DATA_DIR / "images_med_res"
LABELS_PATH = DATA_DIR / "digital_biomass_labels.xlsx"
FIGURES_DIR = Path("figures")   # EDA-Plots werden hier gespeichert
RESULTS_DIR = Path("results")   # Numerische Zusammenfassungen als JSON


def _ensure_dirs() -> None:
    """Erstellt Ausgabeverzeichnisse, falls sie nicht existieren.
    Automatisches Anlegen der Verzeichnisse verhindert Fehler.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Lädt den Biomasse-Datensatz aus Excel.
    DATA LOADING:
    - Prüft zuerst, ob die Datei existiert (Fehlerbehandlung)
    - Konvertiert Unix-Timestamps zu lesbaren Datetime-Objekten
    - errors="coerce" wandelt ungültige Timestamps zu NaT (Not a Time)
    """
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Could not find {LABELS_PATH}")
    df = pd.read_excel(LABELS_PATH)

    # converts the timestamp column from Unix seconds to proper pandas datetime64, and errors="coerce" silently turns any invalid entries into NaT so downstream code can rely on consistent datetime types without crashing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    return df


def save_target_distribution(df: pd.DataFrame) -> None:
    """Erstellt ein Histogramm der Biomasse-Verteilung.
    DISTRIBUTION ANALYSIS:
    - Histogramme zeigen die Verteilung der Zielvariable
    - KDE (Kernel Density Estimate) glättet die Verteilung
    - Hier: Rechtsschief = viele kleine Pflanzen, wenige große
    - Diese Information ist wichtig für die Wahl der Loss-Funktion (MSE vs. Huber)
    """
    # Entferne NaN-Werte für saubere Visualisierung
    target = df["fresh_weight_total"].dropna()

    plt.figure(figsize=(8, 6))
    sns.histplot(target, bins=40, kde=True, color="#2c7fb8")  # 40 Bins + geglättete KDE-Kurve
    plt.title("Fresh Biomass Distribution")
    plt.xlabel("fresh_weight_total [g]")
    plt.ylabel("Plants count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close()


def save_sample_images(df: pd.DataFrame) -> None:
    """Zeigt zufällige Beispielbilder mit ihren Biomasse-Labels.
    VISUAL INSPECTION:
    - Hilft, die Datenqualität zu überprüfen (Fehlende Bilder? Korrupte Dateien?)
    - Zeigt die Variabilität der Daten (Beleuchtung, Hintergrund, Zoom)
    - Gibt Hinweise auf sinnvolle Data Augmentation Strategien
    """
    labeled = df.dropna(subset=["fresh_weight_total"])  # Nur gelabelte Samples
    sample_size = min(9, len(labeled))  # Maximal 9 Bilder (3x3 Grid)
    if sample_size == 0:
        return  # Keine Daten zum Plotten
    
    # Zufällige Auswahl mit festem Seed (Reproduzierbarkeit)
    rng = np.random.default_rng(seed=42)
    sample = labeled.iloc[rng.choice(len(labeled), size=sample_size, replace=False)]
    
    # Grid-Größe berechnen (z.B. 9 Bilder -> 3x3)
    grid_size = math.ceil(sample_size ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = np.atleast_1d(axes).flatten()  # Einheitliches 1D-Array
    
    # Plotte jedes Bild mit Label
    for ax, (_, row) in zip(axes, sample.iterrows()):
        image_path = IMAGE_DIR / row["filename"]
        if not image_path.exists():
            ax.axis("off")  # Verstecke Subplot bei fehlendem Bild
            continue
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"{row['fresh_weight_total']:.1f} g")  # Biomasse als Titel
        ax.axis("off")  # Keine Achsen für saubereres Layout
    
    # Verstecke übrige leere Subplots
    for ax in axes[len(sample) :]:
        ax.axis("off")
    
    plt.suptitle("Sample Training Images with Biomass Labels")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Platz für Titel
    plt.savefig(FIGURES_DIR / "sample_images.png", dpi=200)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Erstellt eine Korrelationsmatrix zwischen Metadaten und Biomasse.
    FEATURE ENGINEERING:
    - Korrelationen zeigen, welche Metadaten-Features nützlich für Vorhersagen wären
    - Starke Korrelationen (z.B. leaf_surface_area) könnten später als zusätzliche Inputs verwendet werden
    - Schwache Korrelationen (z.B. Temperatur, Luftfeuchtigkeit) können ignoriert werden
    - experiment_id zeigt negative Korrelation -> potenzielles Data Leakage Risk
    """
    # Listen der verfügbaren numerischen Features
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
    # Filtere nur vorhandene Spalten (falls Dataset nicht alle enthält)
    available = [col for col in candidate_columns if col in df.columns]
    data = df[available + ["fresh_weight_total"]]
    
    # Berechne Korrelation nur mit Zielvariable (reduziert Clutter)
    corr = data.corr(numeric_only=True)["fresh_weight_total"].dropna()
    corr = corr.sort_values()  # Sortiere für bessere Übersichtlichkeit
    
    # Heatmap mit Annotationen (Zahlenwerte in Zellen)
    plt.figure(figsize=(6, max(4, len(corr) * 0.4)))  # Dynamische Höhe basierend auf Feature-Anzahl
    sns.heatmap(
        corr.to_frame(name="corr"),
        annot=True,              # Zeige Korrelationswerte
        cmap="coolwarm",         # Rot = positiv, Blau = negativ
        vmin=-1,                 # Korrelation von -1 (perfekt negativ)
        vmax=1,                  # bis +1 (perfekt positiv)
        cbar=True,               # Farbskala anzeigen
        fmt=".2f",               # 2 Dezimalstellen
    )
    plt.title("Correlation with Fresh Biomass")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_age_vs_biomass(df: pd.DataFrame) -> None:
    """Scatter-Plot: Pflanzenalter vs. Biomasse.
    TEMPORAL ANALYSIS:
    - Zeigt Wachstumstrend über die Zeit
    - Scatterplot + Regressionslinie = visueller Trend
    - Mehrere Punkte pro Pflanze sichtbar (mehrfache Messungen)
      -> Bestätigt die Notwendigkeit von GroupShuffleSplit beim Training
    """
    subset = df[["age_days", "fresh_weight_total"]].dropna()  # Nur vollständige Zeilen
    if subset.empty:
        return
    
    plt.figure(figsize=(8, 6))
    # Scatterplot mit Farbcodierung nach Alter (viridis: gelb=jung, dunkelblau=alt)
    sns.scatterplot(
        data=subset,
        x="age_days",
        y="fresh_weight_total",
        hue="age_days",
        palette="viridis",
        alpha=0.6,          # Transparenz für überlappende Punkte
        edgecolor="none",   # Keine Umrandung
    )
    # Überlagere lineare Regression (Trendlinie)
    sns.regplot(
        data=subset,
        x="age_days",
        y="fresh_weight_total",
        scatter=False,      # Keine Punkte, nur Linie
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
    """Analysiert RGB-Kanal-Verteilungen der Bilder.
    IMAGE STATISTICS:
    - Berechnet mittlere RGB-Intensitäten für jedes Bild
    - KDE-Plot zeigt Verteilung der Farbwerte
    """
    filenames = df["filename"].dropna().unique().tolist()  # Alle eindeutigen Bilddateinamen
    if not filenames:
        return
    
    # Zufällige Auswahl
    random.seed(42)  # Reproduzierbarkeit
    random.shuffle(filenames)
    selected = filenames[: max_images or len(filenames)]
    
    # Sammle mittlere RGB-Werte für jedes Bild
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
    
    # Konvertiere zu DataFrame für Seaborn-Plotting
    samples = pd.DataFrame(channel_means, columns=["R", "G", "B"])
    melted = samples.melt(var_name="channel", value_name="mean_intensity")
    
    # KDE-Plot (geglättete Dichte-Schätzung)
    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=melted,
        x="mean_intensity",
        hue="channel",
        fill=True,                # Gefüllte Flächen
        common_norm=False,        # Separate Normierung pro Kanal
        alpha=0.35,               # Transparenz
        palette={"R": "red", "G": "green", "B": "blue"},  # Kanal-spezifische Farben
    )
    plt.title("Distribution of Mean RGB Intensities per Image")
    plt.xlabel("Mean channel intensity")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "image_pixel_analysis.png", dpi=200)
    plt.close()


def collect_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Sammelt numerische Statistiken über den Datensatz.
    DATA QUALITY REPORT:
    - Zählt Anzahl der Samples, gelabelten Daten, fehlenden Werten
    - Identifiziert potenzielle Probleme:
      1. Fehlende Labels -> weniger Trainingsdaten
      2. Fehlende Metadaten -> Features nicht nutzbar
      3. Mehrfachaufnahmen pro Pflanze -> Data Leakage Risk
    - Diese Informationen fließen direkt in die Modellierungs-Entscheidungen ein
    """
    summary: Dict[str, object] = {}
    
    # Grundstatistiken
    summary["num_records"] = int(len(df))  # Gesamtanzahl Zeilen
    summary["num_images_on_disk"] = len(list(IMAGE_DIR.glob("*.png")))  # Tatsächliche Bilddateien
    
    # Label-Statistiken
    target = df["fresh_weight_total"]
    summary["num_labeled"] = int(target.notna().sum())  # Anzahl gelabelter Samples
    summary["pct_labeled"] = float(target.notna().mean())  # Prozentsatz
    
    # Biomasse-Statistiken
    summary["fresh_weight_stats"] = {
        k: (float(v) if not math.isnan(v) else None)
        for k, v in target.describe().to_dict().items()
    }
    
    # Fehlende Werte in wichtigen Features
    summary["age_missing_pct"] = float(df["age_days"].isna().mean())
    summary["target_missing_pct"] = float(target.isna().mean())
    
    # Mehrfachmessungen pro Pflanze (Data Leakage Risk)
    plant_counts = df["plant_number"].value_counts()
    summary["plants_with_multiple_samples"] = int((plant_counts > 1).sum())  # Anzahl Pflanzen mit >1 Bild
    summary["max_images_per_plant"] = int(plant_counts.max()) if not plant_counts.empty else 0
    
    # Zeitspanne des Experiments
    summary["timestamp_span_days"] = (
        (df["timestamp"].max() - df["timestamp"].min()).days
        if df["timestamp"].notna().any()
        else None
    )
    
    # Identifiziere Datenqualitäts-Probleme
    issues: List[str] = []
    if summary["target_missing_pct"] > 0.1:
        issues.append(
            "Roughly 14% of the rows have no fresh_weight_total label, so a notable part of the imagery cannot be used for supervised training."
        )
    if summary["age_missing_pct"] > 0.1:
        issues.append("age_days is missing for more than 10% of entries, limiting the reliability of temporal analysis.")
    if summary["max_images_per_plant"] > 5:
        issues.append(
            "Multiple images per plant exist, so random train/test splits risk leakage without grouping by plant_number. Same goes for experiment_id."
        )
    summary["data_quality_issues"] = issues
    
    return summary


def save_summary(summary: Dict[str, object]) -> None:
    """Speichert die Zusammenfassung als JSON.
    DOCUMENTATION: JSON-Format ermöglicht späteres programmatisches Lesen (z.B. für automatische Reports oder Dashboards).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)


def main() -> None:
    """Hauptfunktion: Führt alle EDA-Schritte aus.
    EDA PIPELINE:
    1. Verzeichnisse erstellen
    2. Daten laden
    3. Alle Visualisierungen generieren
    4. Zusammenfassung erstellen und speichern
    5. Zusammenfassung in Console ausgeben
    """
    _ensure_dirs()  # Erstelle figures/ und results/
    df = load_dataset()  # Lade Excel-Datei
    
    # Generiere alle Plots
    save_target_distribution(df)   # Histogramm der Biomasse
    save_sample_images(df)         # Zufällige Beispielbilder
    save_correlation_heatmap(df)   # Korrelationsmatrix
    save_age_vs_biomass(df)        # Alter vs. Biomasse Scatterplot
    save_pixel_analysis(df)        # RGB-Kanal-Verteilungen
    
    # Erstelle und speichere Zusammenfassung
    summary = collect_summary(df)
    save_summary(summary)
    
    # Gib Zusammenfassung auch in Console aus (für schnelles Feedback)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    # Setze Seaborn-Theme für schönere Plots
    sns.set_theme(style="whitegrid")  # Whitegrid-Stil mit Gitterlinien
    main()
