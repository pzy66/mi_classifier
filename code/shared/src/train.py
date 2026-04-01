"""Training entry point for the supported MI classifier pipelines."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

from src.data_loader import extract_dataset, load_subject, load_subject_moabb
from src.models import (
    DeepConvNet,
    EEGNet,
    ShallowConvNet,
    build_classical_pipeline,
    build_optimized_candidates,
)
from src.preprocessing import preprocess


CLASSICAL_MODELS = {"lda", "svm"}
OPTIMIZED_MODEL = "optimized"
DEEP_MODEL_FACTORIES = {
    "eegnet": EEGNet,
    "shallow": ShallowConvNet,
    "deep": DeepConvNet,
}


def resolve_path(project_root: Path, raw_path: str | None) -> Path | None:
    """Resolve a config path relative to the project root when needed."""
    if not raw_path:
        return None

    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def seed_everything(seed: int) -> None:
    """Make training runs deterministic where practical."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Load the YAML config file."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dataset_available(config: dict, project_root: Path) -> Path:
    """Return the extracted dataset directory, extracting the archive if needed."""
    data_source = config["data"].get("source", "local_gdf").lower()
    if data_source == "moabb":
        moabb_data_dir = resolve_path(project_root, config["data"].get("moabb_data_dir", "moabb_data"))
        if moabb_data_dir is None:
            raise ValueError("data.moabb_data_dir must be configured when using MOABB.")
        moabb_data_dir.mkdir(parents=True, exist_ok=True)
        return moabb_data_dir
    if data_source != "local_gdf":
        raise ValueError(f"Unsupported data source: {data_source}")

    dataset_dir = resolve_path(project_root, config["data"].get("dataset_dir"))
    if dataset_dir is not None and dataset_dir.exists():
        return dataset_dir

    archive_path = resolve_path(project_root, config["data"].get("archive_path"))
    extract_dir = resolve_path(project_root, config["data"].get("extract_dir", "data"))

    if archive_path is None or not archive_path.exists():
        missing_path = dataset_dir if dataset_dir is not None else Path("<unset>")
        raise FileNotFoundError(
            "Dataset directory was not found and no valid archive_path is configured. "
            f"Expected dataset directory: {missing_path}"
        )

    if extract_dir is None:
        raise ValueError("extract_dir must be configured when archive extraction is required.")

    extract_dataset(archive_path, extract_dir)

    if dataset_dir is not None and dataset_dir.exists():
        return dataset_dir

    discovered = next(extract_dir.rglob("A01T.gdf"), None)
    if discovered is None:
        raise FileNotFoundError("Dataset extraction finished, but no GDF files were found.")
    return discovered.parent


def to_band_tuple(raw_band: tuple[float, float] | list[float] | None) -> tuple[float, float] | None:
    """Convert a config band definition into a float tuple."""
    if raw_band is None:
        return None
    return tuple(float(value) for value in raw_band)


def load_subject_data(config: dict, dataset_dir: Path, subject_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one subject from the configured data source."""
    epoch_window = config["preprocessing"]["epoch_window"]
    data_source = config["data"].get("source", "local_gdf").lower()

    if data_source == "moabb":
        paradigm_band = to_band_tuple(config["data"].get("moabb_paradigm_band", [0.5, 100.0]))
        if paradigm_band is None:
            raise ValueError("data.moabb_paradigm_band must be configured when using MOABB.")
        return load_subject_moabb(
            subject_id,
            tmin=float(epoch_window[0]),
            tmax=float(epoch_window[1]),
            n_classes=len(config["dataset"]["class_names"]),
            fmin=paradigm_band[0],
            fmax=paradigm_band[1],
            class_names=config["dataset"]["class_names"],
        )

    return load_subject(
        dataset_dir,
        subject_id,
        tmin=float(epoch_window[0]),
        tmax=float(epoch_window[1]),
    )


def build_classical_model(model_type: str, config: dict):
    """Build a classical sklearn-style model."""
    if model_type not in CLASSICAL_MODELS:
        raise ValueError(f"Unsupported classical model: {model_type}")
    return build_classical_pipeline(
        model_type,
        kernel=config["model"].get("svm_kernel", "rbf"),
        C=float(config["model"].get("svm_c", 1.0)),
    )


def build_optimized_model_candidates(config: dict) -> dict[str, object]:
    """Build the optimized candidate pipelines used during validation."""
    return build_optimized_candidates(
        candidate_names=config["model"].get("optimized_candidates"),
        fs=float(config["dataset"]["sampling_rate"]),
        bands=config["model"].get("fbcsp_bands"),
        n_components=int(config["model"].get("fbcsp_components", 4)),
        riemann_band=config["model"].get("riemann_band", [4.0, 40.0]),
        estimator=config["model"].get("riemann_estimator", "lwf"),
        metric=config["model"].get("riemann_metric", "riemann"),
        kernel=config["model"].get("svm_kernel", "rbf"),
        C=float(config["model"].get("svm_c", 1.0)),
    )


def build_deep_model(model_type: str, n_channels: int, n_classes: int):
    """Build a torch model."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for deep models. Install `torch` to use eegnet, shallow, or deep."
        )
    if model_type not in DEEP_MODEL_FACTORIES:
        raise ValueError(f"Unsupported deep model: {model_type}")
    return DEEP_MODEL_FACTORIES[model_type](n_channels=n_channels, n_classes=n_classes)


def evaluate_deep_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Compute validation loss and accuracy for a torch model."""
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch_y.cpu().numpy())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(labels, predictions)
    return average_loss, accuracy


def train_deep_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    model_type: str,
    config: dict,
) -> tuple[nn.Module, float]:
    """Train a deep model with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_deep_model(model_type, n_channels=X_train.shape[1], n_classes=len(np.unique(y_train)))
    model.to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_state = None
    best_val_acc = -1.0
    patience = int(config["training"]["early_stopping_patience"])
    patience_counter = 0

    for epoch in range(int(config["training"]["epochs"])):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_deep_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


def predict_deep_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Run inference with a trained torch model."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
    return torch.argmax(logits, dim=1).cpu().numpy()


def preprocess_trials(X: np.ndarray, config: dict, *, model_type: str) -> np.ndarray:
    """Apply the configured preprocessing for the requested model family."""
    if model_type == OPTIMIZED_MODEL:
        bandpass = to_band_tuple(config["model"].get("optimized_input_bandpass", [4.0, 40.0]))
    else:
        bandpass = to_band_tuple(config["preprocessing"].get("bandpass"))

    return preprocess(
        X,
        fs=float(config["dataset"]["sampling_rate"]),
        bandpass=bandpass,
        notch=config["preprocessing"].get("notch"),
        apply_car=bool(config["preprocessing"].get("apply_car", True)),
        standardize_data=bool(config["preprocessing"].get("standardize", False)),
    )


def run_optimized_subject(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> dict:
    """Select the best optimized pipeline on a validation split, then evaluate on test."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=float(config["training"]["validation_split"]),
        random_state=int(config["training"]["random_state"]),
        stratify=y_train,
    )

    best_name = None
    best_val_acc = -1.0
    best_pipeline = None

    for name, pipeline in build_optimized_model_candidates(config).items():
        pipeline.fit(X_tr, y_tr)
        val_predictions = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_predictions)
        print(f"  candidate={name:<12} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_name = name
            best_val_acc = val_acc
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No optimized candidate pipelines were configured.")

    best_pipeline.fit(X_train, y_train)
    test_predictions = best_pipeline.predict(X_test)

    return {
        "model": OPTIMIZED_MODEL,
        "selected_pipeline": best_name,
        "val_acc": float(best_val_acc),
        "test_acc": float(accuracy_score(y_test, test_predictions)),
        "kappa": float(cohen_kappa_score(y_test, test_predictions)),
    }


def run_subject(config: dict, dataset_dir: Path, subject_id: int) -> dict:
    """Train and evaluate one subject."""
    X_train, y_train, X_test, y_test = load_subject_data(config, dataset_dir, subject_id)

    model_type = config["model"]["type"].lower()
    X_train = preprocess_trials(X_train, config, model_type=model_type)
    X_test = preprocess_trials(X_test, config, model_type=model_type)

    if model_type == OPTIMIZED_MODEL:
        result = run_optimized_subject(X_train, y_train, X_test, y_test, config)
        result["subject"] = subject_id
        return result

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=float(config["training"]["validation_split"]),
        random_state=int(config["training"]["random_state"]),
        stratify=y_train,
    )

    if model_type in CLASSICAL_MODELS:
        model = build_classical_model(model_type, config)
        model.fit(X_tr, y_tr)
        val_predictions = model.predict(X_val)
        test_predictions = model.predict(X_test)
        val_acc = accuracy_score(y_val, val_predictions)
    else:
        model, val_acc = train_deep_model(X_tr, y_tr, X_val, y_val, model_type=model_type, config=config)
        test_predictions = predict_deep_model(model, X_test)

    test_acc = accuracy_score(y_test, test_predictions)
    kappa = cohen_kappa_score(y_test, test_predictions)

    return {
        "subject": subject_id,
        "model": model_type,
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "kappa": float(kappa),
    }


def save_results(results: list[dict], output_dir: Path) -> tuple[Path, Path]:
    """Persist detailed results and a summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    pipeline_counts = {}
    for entry in results:
        pipeline_name = entry.get("selected_pipeline")
        if pipeline_name is None:
            continue
        pipeline_counts[pipeline_name] = pipeline_counts.get(pipeline_name, 0) + 1

    summary = {
        "subjects": len(results),
        "mean_test_acc": float(np.mean([entry["test_acc"] for entry in results])) if results else None,
        "mean_val_acc": float(np.mean([entry["val_acc"] for entry in results])) if results else None,
        "mean_kappa": float(np.mean([entry["kappa"] for entry in results])) if results else None,
        "selected_pipeline_counts": pipeline_counts,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return results_path, summary_path


def main(config_path: str | Path = "config.yaml") -> list[dict]:
    """Load config, run all subjects, and save outputs."""
    config_path = Path(config_path).resolve()
    project_root = config_path.parent
    config = load_config(config_path)
    seed_everything(int(config["training"]["random_state"]))

    dataset_dir = ensure_dataset_available(config, project_root)
    output_dir = resolve_path(project_root, config["results"]["output_dir"])
    if output_dir is None:
        raise ValueError("results.output_dir must be configured.")

    print("=" * 72)
    print("Motor Imagery EEG Classification")
    print(f"Data source: {config['data'].get('source', 'local_gdf')}")
    print(f"Data path: {dataset_dir}")
    print(f"Requested model: {config['model']['type']}")
    if config["model"]["type"].lower() == OPTIMIZED_MODEL:
        candidates = config["model"].get("optimized_candidates", ["hybrid+lda", "hybrid+svm"])
        print(f"Optimized candidates: {', '.join(candidates)}")
    print("=" * 72)

    results = []
    for subject_id in config["dataset"]["subjects"]:
        print(f"\n[Subject {subject_id}]")
        try:
            result = run_subject(config, dataset_dir, int(subject_id))
            results.append(result)
            message = (
                f"val_acc={result['val_acc']:.4f} | "
                f"test_acc={result['test_acc']:.4f} | "
                f"kappa={result['kappa']:.4f}"
            )
            if "selected_pipeline" in result:
                message += f" | selected={result['selected_pipeline']}"
            print(message)
        except Exception as error:
            print(f"Failed subject {subject_id}: {error}")

    results_path, summary_path = save_results(results, output_dir)
    if results:
        print("\nSummary")
        print(f"Mean accuracy: {np.mean([entry['test_acc'] for entry in results]):.4f}")
        print(f"Mean kappa:    {np.mean([entry['kappa'] for entry in results]):.4f}")

    print(f"\nSaved detailed results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    return results
