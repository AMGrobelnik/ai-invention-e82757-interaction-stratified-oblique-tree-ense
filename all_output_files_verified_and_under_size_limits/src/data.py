# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26.0,<2.3.0",
#   "scikit-learn>=1.3.0",
# ]
# ///
"""
ISO-FIGS Benchmark Suite: Breast Cancer Wisconsin Dataset Preparation

Loads the Breast Cancer Wisconsin (Diagnostic) dataset from sklearn — the canonical
FIGS benchmark dataset with 30 highly-correlated features ideal for oblique splits.
Outputs exactly 200 examples in exp_sel_data_out.json schema format.

Selected as THE BEST dataset for ISO-FIGS evaluation because:
1. Canonical FIGS/RO-FIGS benchmark (Singh et al. 2022, imodels library)
2. 30 correlated features (radius↔perimeter↔area) ideal for oblique splits
3. Known feature interaction structure for ANOVA decomposition validation
4. Binary classification — primary FIGS benchmark task type
5. sklearn built-in — perfect reproducibility, zero preprocessing variance
"""

import json
import random
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# AIDEV-NOTE: Fixed seed for reproducibility across all operations
RANDOM_SEED = 42
EXAMPLES_PER_DATASET = 200
TRAIN_RATIO = 0.8

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

WORKSPACE = Path(__file__).parent
OUTPUT_FILE = WORKSPACE / "full_data_out.json"


def format_features_text(feature_names: list[str], values: list) -> str:
    """Format feature names and values into a readable string."""
    parts = []
    for name, val in zip(feature_names, values):
        if isinstance(val, float):
            if val == int(val):
                parts.append(f"{name}={int(val)}")
            else:
                parts.append(f"{name}={val:.4g}")
        else:
            parts.append(f"{name}={val}")
    return ", ".join(parts)


def create_example(
    feature_names: list[str],
    feature_values: list,
    target_value: int,
    target_name: str,
    dataset_name: str,
    split: str,
    task_type: str,
    known_interactions: str = "",
    source: str = "",
    n_features: int = 0,
    n_samples: int = 0,
) -> dict:
    """Create a single example in exp_sel_data_out.json schema format."""
    features_text = format_features_text(feature_names, feature_values)

    input_text = (
        f"Predict {target_name} (binary classification) given these features: "
        f"{features_text}"
    )
    output_text = str(int(target_value))

    context = {
        "dataset_name": dataset_name,
        "task_type": task_type,
        "n_features": n_features,
        "n_samples_total": n_samples,
        "target_name": target_name,
        "source": source,
        "feature_names": feature_names,
        "feature_values": [
            round(v, 6) if isinstance(v, float) else v for v in feature_values
        ],
    }
    if known_interactions:
        context["known_interactions"] = known_interactions

    return {
        "input": input_text,
        "context": context,
        "output": output_text,
        "dataset": dataset_name,
        "split": split,
    }


def process_breast_cancer() -> list[dict]:
    """
    Breast Cancer Wisconsin (Diagnostic) from sklearn.

    569 samples, 30 features, binary classification.
    Features computed from cell nuclei images — many are highly correlated:
    - Size group: mean_radius, mean_perimeter, mean_area (r > 0.99)
    - Worst group: worst_radius, worst_perimeter, worst_area (r > 0.99)
    - Cross-group: mean_radius ↔ worst_radius (r > 0.97)

    These strong correlations make oblique splits significantly more effective
    than axis-aligned splits, which is exactly what ISO-FIGS is designed for.

    Target: 0 = malignant, 1 = benign (sklearn convention)
    Class balance: ~63% benign, ~37% malignant — no extreme imbalance.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = list(data.feature_names)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y,
    )

    # Sample exactly EXAMPLES_PER_DATASET examples (160 train + 40 test)
    n_train_sample = int(EXAMPLES_PER_DATASET * TRAIN_RATIO)  # 160
    n_test_sample = EXAMPLES_PER_DATASET - n_train_sample       # 40

    rng = np.random.RandomState(RANDOM_SEED)
    train_idx = rng.choice(len(X_train), size=n_train_sample, replace=False)
    test_idx = rng.choice(len(X_test), size=n_test_sample, replace=False)

    dataset_name = "breast_cancer"
    target_name = "malignant"
    task_type = "classification"
    source = "sklearn.datasets.load_breast_cancer"
    known_interactions = (
        "mean_radius*mean_perimeter correlation (r>0.99), "
        "mean_area*worst_area cross-group correlation (r>0.97), "
        "texture*smoothness interaction, "
        "concavity*concave_points correlation — "
        "these feature groups form natural interaction tiers "
        "ideal for ISO-FIGS oblique-within-tier splits"
    )

    examples = []
    for split_name, X_split, y_split, idx in [
        ("train", X_train, y_train, train_idx),
        ("test", X_test, y_test, test_idx),
    ]:
        for i in idx:
            feature_values = [float(v) for v in X_split[i]]
            target_value = int(y_split[i])
            examples.append(create_example(
                feature_names=feature_names,
                feature_values=feature_values,
                target_value=target_value,
                target_name=target_name,
                dataset_name=dataset_name,
                split=split_name,
                task_type=task_type,
                known_interactions=known_interactions,
                source=source,
                n_features=n_features,
                n_samples=n_samples,
            ))

    return examples


def main():
    print("Processing breast_cancer (Breast Cancer Wisconsin Diagnostic)...")
    examples = process_breast_cancer()

    n_train = sum(1 for e in examples if e["split"] == "train")
    n_test = sum(1 for e in examples if e["split"] == "test")
    print(f"  -> {len(examples)} examples ({n_train} train, {n_test} test)")

    # Verify counts
    assert len(examples) == EXAMPLES_PER_DATASET, (
        f"Expected {EXAMPLES_PER_DATASET}, got {len(examples)}"
    )
    assert n_train == int(EXAMPLES_PER_DATASET * TRAIN_RATIO), (
        f"Expected {int(EXAMPLES_PER_DATASET * TRAIN_RATIO)} train, got {n_train}"
    )

    # Verify class balance in output
    targets = [int(e["output"]) for e in examples]
    n_malignant = sum(1 for t in targets if t == 0)
    n_benign = sum(1 for t in targets if t == 1)
    print(f"  Class distribution: {n_malignant} malignant (0), {n_benign} benign (1)")

    # Build output in schema format
    output = {"examples": examples}

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nTotal examples: {len(examples)}")
    print(f"Dataset: breast_cancer")
    print(f"Task: binary classification (30 features, 569 samples)")
    print(f"Source: sklearn.datasets.load_breast_cancer")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
