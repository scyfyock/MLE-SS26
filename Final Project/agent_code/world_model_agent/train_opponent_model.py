"""Offline warm-start and validation for the factored opponent-action model."""

import argparse
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np

from .evaluate_world_model import chronological_replay, confusion_metrics
from .world_model import OPPONENT_ACTIONS


def _fit(model, inputs, targets, epochs, batch_size, seed):
    rng = random.Random(seed)
    indices = list(range(len(targets)))
    classes = np.arange(len(OPPONENT_ACTIONS), dtype=np.int8)
    for _epoch in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selection = indices[start:start + batch_size]
            batch_targets = targets[selection]
            weights = np.ones(len(selection), dtype=np.float32)
            weights[batch_targets == 5] = 2.0
            weights[batch_targets == 6] = 3.0
            model.opponent_model.partial_fit(
                inputs[selection],
                batch_targets,
                classes=classes,
                sample_weight=weights,
            )
            model.opponent_training_steps += 1
            model.opponent_last_loss = float(model.opponent_model.loss_)


def _arrays(model, transitions):
    inputs, targets = model._opponent_training_samples(transitions)
    if not inputs:
        return (
            np.empty((0, 48), dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )
    return np.stack(inputs), np.asarray(targets, dtype=np.int8)


def evaluate_classifier(model, inputs, targets):
    predicted = model.opponent_model.predict(inputs).astype(np.int8)
    report = {
        "samples": int(len(targets)),
        "accuracy": float(np.mean(predicted == targets)),
        "majority_baseline": float(
            np.max(np.bincount(targets, minlength=len(OPPONENT_ACTIONS)))
            / len(targets)
        ),
        "by_action": {},
    }
    for index, action in enumerate(OPPONENT_ACTIONS):
        metrics = confusion_metrics(targets == index, predicted == index)
        metrics["support"] = int(np.sum(targets == index))
        report["by_action"][action] = metrics
    return report


def print_report(report):
    print(
        f"Held-out opponent actions: {report['samples']} samples, "
        f"accuracy={100 * report['accuracy']:.1f}%, "
        f"majority={100 * report['majority_baseline']:.1f}%"
    )
    print("action    support  precision  recall    F1")
    for action, metrics in report["by_action"].items():
        print(
            f"{action:<8} {metrics['support']:>7}  "
            f"{100 * metrics['precision']:>8.1f}%  "
            f"{100 * metrics['recall']:>5.1f}%  "
            f"{100 * metrics['f1']:>5.1f}%"
        )


def atomic_save(saved_data, checkpoint):
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{checkpoint.name}.",
            suffix=".tmp",
            dir=checkpoint.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            pickle.dump(
                saved_data,
                temporary,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        os.replace(temporary_path, checkpoint)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warm-start the local opponent-action classifier."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fit-all", action="store_true")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with args.checkpoint.open("rb") as file:
        saved_data = pickle.load(file)
    model = saved_data["world_model"]
    model.ensure_opponent_model_state()
    transitions = chronological_replay(model)
    split = max(
        1,
        min(
            len(transitions) - 1,
            int(len(transitions) * (1.0 - args.validation_fraction)),
        ),
    )
    train_inputs, train_targets = _arrays(model, transitions[:split])
    validation_inputs, validation_targets = _arrays(
        model,
        transitions[split:],
    )
    if not len(train_targets) or not len(validation_targets):
        raise ValueError("Replay does not contain enough opponent transitions.")
    _fit(
        model,
        train_inputs,
        train_targets,
        args.epochs,
        args.batch_size,
        args.seed,
    )
    report = evaluate_classifier(
        model,
        validation_inputs,
        validation_targets,
    )
    print_report(report)

    if args.fit_all:
        _fit(
            model,
            validation_inputs,
            validation_targets,
            max(1, args.epochs // 2),
            args.batch_size,
            args.seed + 1,
        )
        print(
            f"Deployment fit complete: "
            f"{model.opponent_training_steps} batches, "
            f"last_loss={model.opponent_last_loss:.4f}"
        )
    if args.save:
        atomic_save(saved_data, args.checkpoint)
        print(f"Updated checkpoint written atomically to {args.checkpoint}")


if __name__ == "__main__":
    main()
