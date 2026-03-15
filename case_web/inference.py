from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ARTIFACTS_DIR = Path(os.environ.get("MODEL_ARTIFACTS_DIR", BASE_DIR / "artifacts")).expanduser()
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
RESULTS_DIR = RUNTIME_DIR / "results"
PLOTS_DIR = RUNTIME_DIR / "plots"

MODEL_PATH = ARTIFACTS_DIR / "tiny_cnn_model.h5"
CONFIG_PATH = ARTIFACTS_DIR / "tiny_cnn_config.json"
LABEL_MAPPING_PATH = ARTIFACTS_DIR / "tiny_cnn_label_mapping.csv"
HISTORY_PATH = ARTIFACTS_DIR / "tiny_cnn_history.csv"
ANALYTICS_PATH = ARTIFACTS_DIR / "tiny_cnn_analytics.json"
FALLBACK_MODEL_PATH = ROOT_DIR / "tiny_cnn_model.h5"
FALLBACK_LABEL_MAPPING_PATH = ROOT_DIR / "data" / "extracted_audio" / "label_mapping.csv"

LATEST_RESULT_PATH = RESULTS_DIR / "latest_test_summary.json"
LATEST_PREDICTIONS_PATH = RESULTS_DIR / "latest_test_predictions.csv"
LATEST_CONFIDENCE_PLOT = PLOTS_DIR / "latest_test_confidence.html"
LATEST_CONFUSION_PLOT = PLOTS_DIR / "latest_test_confusion_matrix.html"

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "expected_num_samples": 80000,
    "frame_length": 1024,
    "frame_step": 256,
    "fft_length": 1024,
    "n_mels": 96,
    "img_height": 96,
    "img_width": 160,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 3e-4,
    "use_wave_aug": True,
    "use_spec_aug": True,
    "train_accuracy": None,
    "val_accuracy": None,
    "train_macro_f1": None,
    "val_macro_f1": None,
    "train_loss_eval": None,
    "val_loss_eval": None,
}


class ArtifactError(RuntimeError):
    pass


class InferenceError(RuntimeError):
    pass


def ensure_runtime_dirs() -> None:
    for directory in [ARTIFACTS_DIR, UPLOADS_DIR, RESULTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_model_path() -> Path:
    for path in [MODEL_PATH, FALLBACK_MODEL_PATH]:
        if path.exists():
            return path
    raise ArtifactError(
        "Не найдена h5-модель. Положите tiny_cnn_model.h5 в "
        f"{ARTIFACTS_DIR} или в корень проекта"
    )


def _resolve_label_mapping_path() -> Path:
    for path in [LABEL_MAPPING_PATH, FALLBACK_LABEL_MAPPING_PATH]:
        if path.exists():
            return path
    raise ArtifactError(
        "Не найден label_mapping.csv. Положите tiny_cnn_label_mapping.csv в "
        f"{ARTIFACTS_DIR} или сохраните data/extracted_audio/label_mapping.csv"
    )


def _read_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        config.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return config


def _read_history() -> pd.DataFrame:
    if HISTORY_PATH.exists():
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame(columns=["loss", "val_loss", "accuracy", "val_accuracy"])


def _read_analytics() -> dict:
    if ANALYTICS_PATH.exists():
        return json.loads(ANALYTICS_PATH.read_text(encoding="utf-8"))
    return {}


@lru_cache(maxsize=1)
def load_artifact_metadata() -> dict:
    ensure_runtime_dirs()
    model_path = _resolve_model_path()
    label_mapping_path = _resolve_label_mapping_path()
    config = _read_config()
    mapping_df = pd.read_csv(label_mapping_path)
    if "class_id" not in mapping_df.columns and "internal_class_id" in mapping_df.columns:
        mapping_df = mapping_df.rename(columns={"internal_class_id": "class_id"})
    mapping_df = mapping_df.sort_values("class_id")
    class_names = mapping_df["label"].astype(str).tolist()
    history_df = _read_history()
    analytics = _read_analytics()
    return {
        "config": config,
        "mapping_df": mapping_df,
        "class_names": class_names,
        "history_df": history_df,
        "analytics": analytics,
        "model_path": model_path,
        "label_mapping_path": label_mapping_path,
    }


@lru_cache(maxsize=1)
def load_prediction_bundle() -> dict:
    meta = load_artifact_metadata()
    model = tf.keras.models.load_model(str(meta["model_path"]))
    return {**meta, "model": model}


def _fit_wave_length(waves: np.ndarray, expected_num_samples: int) -> np.ndarray:
    if waves.shape[1] == expected_num_samples:
        return waves
    if waves.shape[1] > expected_num_samples:
        return waves[:, :expected_num_samples]
    pad_width = expected_num_samples - waves.shape[1]
    return np.pad(waves, ((0, 0), (0, pad_width)), mode="constant")


def _prepare_wave_array(raw_x: np.ndarray, expected_num_samples: int) -> np.ndarray:
    waves = np.asarray(raw_x, dtype=np.float32)
    if waves.ndim == 3 and waves.shape[-1] == 1:
        waves = waves[..., 0]
    if waves.ndim != 2:
        raise InferenceError(f"Ожидался массив формы (N, T), получено {waves.shape}")
    return _fit_wave_length(waves, expected_num_samples)


def _normalize_label_value(value: object, class_names: list[str]) -> str:
    label = str(value).strip()
    if label in class_names:
        return label
    if len(label) > 32 and label[32:] in class_names:
        return label[32:]
    return label


def _prepare_targets(raw_y: np.ndarray | None, class_names: list[str]) -> tuple[np.ndarray | None, str]:
    if raw_y is None:
        return None, "missing"

    targets = np.asarray(raw_y)
    targets = np.squeeze(targets)
    if targets.ndim != 1:
        raise InferenceError(f"Ожидался одномерный массив test_y, получено {targets.shape}")

    if np.issubdtype(targets.dtype, np.integer):
        y = targets.astype(np.int32)
        if (y < 0).any() or (y >= len(class_names)).any():
            raise InferenceError("test_y содержит class_id вне диапазона label_mapping")
        return y, "numeric"

    if np.issubdtype(targets.dtype, np.floating) and np.allclose(targets, np.round(targets)):
        y = np.round(targets).astype(np.int32)
        if (y < 0).any() or (y >= len(class_names)).any():
            raise InferenceError("test_y содержит class_id вне диапазона label_mapping")
        return y, "numeric"

    series = pd.Series(targets).astype(str).str.strip()
    if series.str.fullmatch(r"\d+").all():
        y = series.astype(np.int32).to_numpy()
        if (y < 0).any() or (y >= len(class_names)).any():
            raise InferenceError("test_y содержит class_id вне диапазона label_mapping")
        return y, "numeric_text"

    normalized = series.map(lambda value: _normalize_label_value(value, class_names))
    unknown = normalized[~normalized.isin(class_names)]
    if not unknown.empty:
        raise InferenceError(
            "Не удалось сопоставить test_y с label_mapping: "
            + ", ".join(unknown.unique()[:5].tolist())
        )

    label_to_id = {label: idx for idx, label in enumerate(class_names)}
    return normalized.map(label_to_id).to_numpy(np.int32), "label_text"


def _extract_npz_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    npz = np.load(npz_path, allow_pickle=False)
    x_key = next((key for key in ["test_x", "valid_x", "train_x"] if key in npz.files), None)
    y_key = next((key for key in ["test_y", "valid_y", "train_y"] if key in npz.files), None)

    if x_key is None:
        raise InferenceError(f"В {npz_path.name} не найден массив test_x/valid_x/train_x")

    return npz[x_key], npz[y_key] if y_key else None, {"x_key": x_key, "y_key": y_key}


def _preprocess_waves(waves: np.ndarray, config: dict, batch_size: int = 32) -> np.ndarray:
    expected_num_samples = int(config["expected_num_samples"])
    frame_length = int(config["frame_length"])
    frame_step = int(config["frame_step"])
    fft_length = int(config["fft_length"])
    n_mels = int(config["n_mels"])
    img_height = int(config["img_height"])
    img_width = int(config["img_width"])
    sample_rate = int(config["sample_rate"])

    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=fft_length // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=20.0,
        upper_edge_hertz=sample_rate / 2.0,
    )

    images = []
    for start in range(0, len(waves), batch_size):
        batch = tf.convert_to_tensor(waves[start:start + batch_size], dtype=tf.float32)
        batch = tf.reshape(batch, [-1, expected_num_samples])
        max_abs = tf.reduce_max(tf.abs(batch), axis=1, keepdims=True)
        batch = tf.where(max_abs > 0, batch / max_abs, batch)

        stft = tf.signal.stft(
            batch,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            pad_end=True,
        )
        power = tf.abs(stft) ** 2
        mel = tf.tensordot(power, mel_weight_matrix, axes=[[2], [0]])
        logmel = tf.math.log(mel + 1e-6)
        logmel = tf.transpose(logmel, perm=[0, 2, 1])
        mean = tf.reduce_mean(logmel, axis=[1, 2], keepdims=True)
        std = tf.math.reduce_std(logmel, axis=[1, 2], keepdims=True)
        image = (logmel - mean) / (std + 1e-6)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, [img_height, img_width])
        images.append(image.numpy().astype(np.float32))

    return np.concatenate(images, axis=0)


def _style_figure(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str, hovermode: str = "x") -> go.Figure:
    theme_color = "#00BCD4"
    bg_color = "#000066"
    grid_color = "#1DAAE6"

    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 24, "family": "Arial Black", "color": theme_color},
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        hovermode=hovermode,
        bargap=0.1,
        height=600,
        width=1000,
        font={"family": "Roboto, sans-serif", "size": 14, "color": theme_color},
        legend={"font": {"color": theme_color}},
    )
    fig.update_xaxes(showgrid=False, gridcolor=grid_color, linecolor=theme_color, tickfont={"size": 10, "color": theme_color}, title_font={"color": theme_color})
    fig.update_yaxes(showgrid=True, gridcolor=grid_color, linecolor=theme_color, tickfont={"color": theme_color}, title_font={"color": theme_color})
    return fig


def _save_figure(fig: go.Figure, output_path: Path) -> None:
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def _build_confidence_plot(predictions_df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=predictions_df["record_id"],
            y=predictions_df["confidence"],
            mode="markers",
            name="Confidence",
            marker={
                "size": 8,
                "opacity": 0.8,
                "color": predictions_df["correct"].map({True: "#00E676", False: "#FF5252"}).fillna("#00BCD4"),
            },
            customdata=predictions_df[["pred_class_id", "pred_label", "true_class_id", "true_label", "correct"]].fillna("").to_numpy(),
            hovertemplate="<b>Запись:</b> %{x}<br><b>Confidence:</b> %{y:.4f}<br><b>Pred ID:</b> %{customdata[0]}<br><b>Pred Label:</b> %{customdata[1]}<br><b>True ID:</b> %{customdata[2]}<br><b>True Label:</b> %{customdata[3]}<br><b>Correct:</b> %{customdata[4]}<extra></extra>",
        )
    )
    _style_figure(fig, title, "Номер записи", "Confidence")
    fig.update_yaxes(range=[0, 1])
    _save_figure(fig, output_path)


def _build_confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    np.add.at(matrix, (y_true, y_pred), 1)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=np.arange(num_classes),
            y=np.arange(num_classes),
            colorscale="Blues",
            hovertemplate="<b>Pred:</b> %{x}<br><b>True:</b> %{y}<br><b>Count:</b> %{z}<extra></extra>",
            colorbar={"title": "Count"},
        )
    )
    _style_figure(fig, "Test confusion matrix", "Predicted class_id", "True class_id", hovermode="closest")
    _save_figure(fig, output_path)


def build_dashboard_payload() -> dict:
    meta = load_artifact_metadata()
    config = meta["config"]
    mapping_df = meta["mapping_df"].copy()
    history_df = meta["history_df"]
    analytics = meta["analytics"].copy()

    if "train_class_counts" not in analytics:
        if "train_count" in mapping_df.columns:
            analytics["train_class_counts"] = dict(
                zip(mapping_df["class_id"].astype(int).tolist(), mapping_df["train_count"].fillna(0).astype(int).tolist())
            )
        else:
            analytics["train_class_counts"] = {}

    if "top5_validation_classes" not in analytics:
        if "valid_count" in mapping_df.columns:
            top5_df = mapping_df.sort_values("valid_count", ascending=False).head(5)
            analytics["top5_validation_classes"] = top5_df[["class_id", "label", "valid_count"]].rename(columns={"valid_count": "count"}).to_dict(orient="records")
        else:
            analytics["top5_validation_classes"] = []

    if "epochs" not in analytics:
        analytics["epochs"] = list(range(1, len(history_df) + 1))
    if "train_loss_curve" not in analytics:
        analytics["train_loss_curve"] = history_df["loss"].astype(float).tolist() if "loss" in history_df else []
    if "val_loss_curve" not in analytics:
        analytics["val_loss_curve"] = history_df["val_loss"].astype(float).tolist() if "val_loss" in history_df else []
    if "train_accuracy_curve" not in analytics:
        analytics["train_accuracy_curve"] = history_df["accuracy"].astype(float).tolist() if "accuracy" in history_df else []
    if "val_accuracy_curve" not in analytics:
        analytics["val_accuracy_curve"] = history_df["val_accuracy"].astype(float).tolist() if "val_accuracy" in history_df else []
    if "final_accuracy" not in analytics:
        analytics["final_accuracy"] = config.get("val_accuracy") or config.get("val_accuracy_eval") or config.get("val_accuracy")
    if "final_loss" not in analytics:
        analytics["final_loss"] = config.get("val_loss_eval")
    analytics["class_names"] = meta["class_names"]

    latest_test = None
    if LATEST_RESULT_PATH.exists():
        latest_test = json.loads(LATEST_RESULT_PATH.read_text(encoding="utf-8"))

    return {
        "ready": True,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model": {
            "num_classes": len(meta["class_names"]),
            "class_names": meta["class_names"],
            "validation_accuracy": analytics.get("final_accuracy"),
            "validation_loss": analytics.get("final_loss"),
        },
        "training": analytics,
        "test": latest_test,
        "plots": {
            "training_curves": "tiny_cnn_training_curves.html" if (ARTIFACTS_DIR / "tiny_cnn_training_curves.html").exists() else None,
            "class_distribution": "tiny_cnn_class_distribution.html" if (ARTIFACTS_DIR / "tiny_cnn_class_distribution.html").exists() else None,
            "confidence": LATEST_CONFIDENCE_PLOT.name if LATEST_CONFIDENCE_PLOT.exists() else ("tiny_cnn_validation_confidence.html" if (ARTIFACTS_DIR / "tiny_cnn_validation_confidence.html").exists() else None),
            "top5_validation": "tiny_cnn_top5_validation_classes.html" if (ARTIFACTS_DIR / "tiny_cnn_top5_validation_classes.html").exists() else None,
            "confusion_matrix": LATEST_CONFUSION_PLOT.name if LATEST_CONFUSION_PLOT.exists() else ("tiny_cnn_confusion_matrix.html" if (ARTIFACTS_DIR / "tiny_cnn_confusion_matrix.html").exists() else None),
        },
        "downloads": {
            "predictions_csv": LATEST_PREDICTIONS_PATH.name if LATEST_PREDICTIONS_PATH.exists() else None,
            "summary_json": LATEST_RESULT_PATH.name if LATEST_RESULT_PATH.exists() else None,
        },
    }


def run_test_inference(npz_path: Path) -> dict:
    ensure_runtime_dirs()
    bundle = load_prediction_bundle()
    config = bundle["config"]
    class_names = bundle["class_names"]
    model = bundle["model"]

    raw_x, raw_y, keys = _extract_npz_arrays(npz_path)
    waves = _prepare_wave_array(raw_x, int(config["expected_num_samples"]))
    y_true, target_mode = _prepare_targets(raw_y, class_names)
    images = _preprocess_waves(waves, config, batch_size=int(config.get("batch_size", 32)))

    probabilities = model.predict(images, batch_size=int(config.get("batch_size", 32)), verbose=0)
    pred_ids = probabilities.argmax(axis=1).astype(np.int32)
    confidence = probabilities.max(axis=1).astype(float)

    predictions_df = pd.DataFrame(
        {
            "record_id": np.arange(1, len(pred_ids) + 1),
            "pred_class_id": pred_ids,
            "pred_label": [class_names[idx] for idx in pred_ids],
            "confidence": confidence,
        }
    )

    result = {
        "file_name": npz_path.name,
        "num_samples": int(len(predictions_df)),
        "x_key": keys["x_key"],
        "y_key": keys["y_key"],
        "targets_present": y_true is not None,
        "target_mode": target_mode,
        "final_accuracy": None,
        "final_loss": None,
        "preview_rows": [],
    }

    if y_true is not None:
        true_ids = y_true.astype(np.int32)
        predictions_df["true_class_id"] = true_ids
        predictions_df["true_label"] = [class_names[idx] for idx in true_ids]
        predictions_df["correct"] = predictions_df["pred_class_id"] == predictions_df["true_class_id"]
        result["final_accuracy"] = float(predictions_df["correct"].mean())
        losses = tf.keras.losses.sparse_categorical_crossentropy(true_ids, probabilities).numpy()
        result["final_loss"] = float(np.mean(losses))
        _build_confusion_plot(true_ids, pred_ids, LATEST_CONFUSION_PLOT)
    else:
        predictions_df["true_class_id"] = None
        predictions_df["true_label"] = None
        predictions_df["correct"] = None

    _build_confidence_plot(
        predictions_df,
        "Уверенность модели по каждой test-записи",
        LATEST_CONFIDENCE_PLOT,
    )

    predictions_df.to_csv(LATEST_PREDICTIONS_PATH, index=False)
    result["preview_rows"] = predictions_df.head(20).to_dict(orient="records")
    result["mean_confidence"] = float(predictions_df["confidence"].mean())
    result["predicted_class_counts"] = (
        predictions_df["pred_class_id"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    LATEST_RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def get_plot_path(filename: str) -> Path:
    ensure_runtime_dirs()
    candidates = [PLOTS_DIR / filename, ARTIFACTS_DIR / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(filename)


def get_result_path(filename: str) -> Path:
    ensure_runtime_dirs()
    candidate = RESULTS_DIR / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(filename)
