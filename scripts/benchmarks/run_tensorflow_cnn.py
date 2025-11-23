#!/usr/bin/env python3
"""TensorFlow CNN mini benchmark for CUDA WSL installer."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TF CNN training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--result-file", type=Path)
    return parser.parse_args()


def configure_env(device: str) -> None:
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


ARGS = parse_args()
configure_env(ARGS.device)

import tensorflow as tf  # noqa: E402


def ensure_device(device: str) -> None:
    if device == "cuda":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise SystemExit("TensorFlow did not detect a GPU")


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def load_data():
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    return x_train, y_train


def run_benchmark() -> dict:
    ensure_device(ARGS.device)
    x_train, y_train = load_data()
    model = build_model()

    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=ARGS.epochs,
        batch_size=ARGS.batch_size,
        verbose=0,
    )
    duration = time.perf_counter() - start

    return {
        "device": ARGS.device,
        "epochs": ARGS.epochs,
        "batch_size": ARGS.batch_size,
        "seconds": duration,
        "history": {"loss": history.history["loss"]},
    }


def main() -> None:
    result = run_benchmark()
    print(
        f"[benchmark] tf_cnn device={result['device']} epochs={result['epochs']} "
        f"time={result['seconds']:.2f}s"
    )
    if ARGS.result_file:
        ARGS.result_file.parent.mkdir(parents=True, exist_ok=True)
        ARGS.result_file.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
