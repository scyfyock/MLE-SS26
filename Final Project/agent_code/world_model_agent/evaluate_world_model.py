"""Evaluate recursive full-state predictions of a saved Dyna world model.

The first state of every rollout is real. Afterwards the model consumes its
own generated full state while actions come from the recorded episode. This
open-loop evaluation exposes compounding dynamics errors independently of the
policy used to collect the data.
"""

import argparse
import json
from pathlib import Path
import pickle
import random

import numpy as np

from .callbacks import (
    ACTIONS,
    MODEL_SCHEMA_VERSION,
)
from .world_model import (
    LearnedWorldModel,
    N_BOARD_CHANNELS,
    N_WORLD_SCALARS,
    world_state_to_game_state,
)


WORLD_CHANNEL_NAMES = (
    "field",
    "coins",
    "bombs",
    "explosions",
    "self_position",
    "opponents",
    "opponent_bombs_left",
)

WORLD_OBJECT_NAMES = (
    "field_layout",
    "coins",
    "bombs",
    "explosions",
    "self_position",
    "self_bombs_left",
    "opponents",
)


def chronological_replay(world_model):
    """Return ring-buffer experiences from oldest to newest."""

    replay = list(world_model.replay)
    if len(replay) < world_model.replay_capacity:
        return replay
    position = int(world_model.replay_position)
    return replay[position:] + replay[:position]


def completed_episodes(world_model):
    """Recover complete episodes from terminal transition boundaries."""

    replay = chronological_replay(world_model)
    episodes = []
    current_episode = []
    prefix_may_be_incomplete = len(replay) == world_model.replay_capacity

    for experience in replay:
        current_episode.append(experience)
        if experience.done:
            if not prefix_may_be_incomplete:
                episodes.append(current_episode)
            prefix_may_be_incomplete = False
            current_episode = []
    return episodes


def confusion_metrics(true_values, predicted_values):
    true_values = np.asarray(true_values, dtype=bool)
    predicted_values = np.asarray(predicted_values, dtype=bool)
    true_positive = int(np.sum(true_values & predicted_values))
    false_positive = int(np.sum(~true_values & predicted_values))
    false_negative = int(np.sum(true_values & ~predicted_values))
    true_negative = int(np.sum(~true_values & ~predicted_values))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    return {
        "accuracy": (
            (true_positive + true_negative) / len(true_values)
            if len(true_values)
            else 0.0
        ),
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def empty_accumulator():
    return {
        "world_matches": 0,
        "world_total": 0,
        "world_exact": 0,
        "world_states": 0,
        "policy_matches": 0,
        "policy_total": 0,
        "policy_exact": 0,
        "channels": {
            name: {"matches": 0, "total": 0}
            for name in WORLD_CHANNEL_NAMES
        },
        "objects": {
            name: {"matches": 0, "total": 0}
            for name in WORLD_OBJECT_NAMES
        },
        "scalars": {"matches": 0, "total": 0},
        "reward_errors": [],
        "terminal_reward_errors": [],
        "nonterminal_reward_errors": [],
        "done_true": [],
        "done_probabilities": [],
        "death_true": [],
        "death_probabilities": [],
        "suicide_true": [],
        "suicide_probabilities": [],
        "legal_true": [],
        "legal_predicted": [],
        "samples": 0,
    }


def _legal_mask(action_indices):
    mask = np.zeros(len(ACTIONS), dtype=bool)
    if action_indices:
        mask[list(action_indices)] = True
    return mask


def add_prediction(
    accumulator,
    prediction,
    predicted_policy_state,
    predicted_valid_actions,
    experience,
):
    accumulator["samples"] += 1
    reward_error = abs(prediction.reward - experience.reward)
    accumulator["reward_errors"].append(reward_error)
    if experience.done:
        accumulator["terminal_reward_errors"].append(reward_error)
    else:
        accumulator["nonterminal_reward_errors"].append(reward_error)
    accumulator["done_true"].append(experience.done)
    accumulator["done_probabilities"].append(
        prediction.done_probability
    )
    if experience.death_within_horizon is not None:
        accumulator["death_true"].append(
            experience.death_within_horizon
        )
        accumulator["death_probabilities"].append(
            prediction.death_probability
        )
    if experience.suicide_within_horizon is not None:
        accumulator["suicide_true"].append(
            experience.suicide_within_horizon
        )
        accumulator["suicide_probabilities"].append(
            prediction.suicide_probability
        )
    accumulator["legal_true"].extend(
        _legal_mask(experience.next_valid_actions).tolist()
    )
    accumulator["legal_predicted"].extend(
        _legal_mask(predicted_valid_actions).tolist()
    )

    if experience.next_world_state is None:
        return

    actual_world = np.asarray(
        experience.next_world_state.vector,
        dtype=np.float32,
    )
    predicted_world = np.asarray(
        prediction.next_world_state.vector,
        dtype=np.float32,
    )
    matches = np.isclose(actual_world, predicted_world, atol=1e-6)
    accumulator["world_matches"] += int(np.sum(matches))
    accumulator["world_total"] += len(matches)
    accumulator["world_exact"] += int(np.all(matches))
    accumulator["world_states"] += 1

    actual_game = world_state_to_game_state(
        experience.next_world_state
    )
    predicted_game = world_state_to_game_state(
        prediction.next_world_state
    )
    actual_opponents = {
        other[3]: bool(other[2])
        for other in actual_game["others"]
    }
    predicted_opponents = {
        other[3]: bool(other[2])
        for other in predicted_game["others"]
    }
    object_matches = {
        "field_layout": np.array_equal(
            actual_game["field"],
            predicted_game["field"],
        ),
        "coins": set(actual_game["coins"]) == set(predicted_game["coins"]),
        "bombs": set(actual_game["bombs"]) == set(predicted_game["bombs"]),
        "explosions": np.array_equal(
            actual_game["explosion_map"],
            predicted_game["explosion_map"],
        ),
        "self_position": (
            actual_game["self"][3] == predicted_game["self"][3]
        ),
        "self_bombs_left": (
            bool(actual_game["self"][2])
            == bool(predicted_game["self"][2])
        ),
        "opponents": actual_opponents == predicted_opponents,
    }
    for name, matched in object_matches.items():
        accumulator["objects"][name]["matches"] += int(matched)
        accumulator["objects"][name]["total"] += 1

    board_size = int(np.prod(prediction.next_world_state.shape))
    for channel, name in enumerate(WORLD_CHANNEL_NAMES):
        start = channel * board_size
        channel_matches = matches[start:start + board_size]
        accumulator["channels"][name]["matches"] += int(
            np.sum(channel_matches)
        )
        accumulator["channels"][name]["total"] += board_size

    scalar_matches = matches[
        N_BOARD_CHANNELS * board_size:
        N_BOARD_CHANNELS * board_size + N_WORLD_SCALARS
    ]
    accumulator["scalars"]["matches"] += int(np.sum(scalar_matches))
    accumulator["scalars"]["total"] += len(scalar_matches)

    actual_policy = np.asarray(experience.next_state)
    predicted_policy = np.asarray(predicted_policy_state)
    policy_matches = actual_policy == predicted_policy
    accumulator["policy_matches"] += int(np.sum(policy_matches))
    accumulator["policy_total"] += len(policy_matches)
    accumulator["policy_exact"] += int(np.all(policy_matches))


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def finalize(accumulator):
    done_true = np.asarray(accumulator["done_true"], dtype=bool)
    done_probabilities = np.asarray(
        accumulator["done_probabilities"],
        dtype=np.float64,
    )
    done_threshold_sweep = {
        str(threshold): confusion_metrics(
            done_true,
            done_probabilities >= threshold,
        )
        for threshold in (0.5, 0.7, 0.9)
    }
    suicide_true = np.asarray(accumulator["suicide_true"], dtype=bool)
    suicide_probabilities = np.asarray(
        accumulator["suicide_probabilities"],
        dtype=np.float64,
    )
    suicide_threshold_sweep = {
        str(threshold): confusion_metrics(
            suicide_true,
            suicide_probabilities >= threshold,
        )
        for threshold in (0.5, 0.7, 0.9)
    }
    death_true = np.asarray(accumulator["death_true"], dtype=bool)
    death_probabilities = np.asarray(
        accumulator["death_probabilities"],
        dtype=np.float64,
    )
    death_threshold_sweep = {
        str(threshold): confusion_metrics(
            death_true,
            death_probabilities >= threshold,
        )
        for threshold in (0.5, 0.7, 0.9)
    }
    legal = confusion_metrics(
        accumulator["legal_true"],
        accumulator["legal_predicted"],
    )
    return {
        "samples": accumulator["samples"],
        "world_feature_accuracy": (
            accumulator["world_matches"] / accumulator["world_total"]
            if accumulator["world_total"]
            else None
        ),
        "world_exact_accuracy": (
            accumulator["world_exact"] / accumulator["world_states"]
            if accumulator["world_states"]
            else None
        ),
        "policy_feature_accuracy": (
            accumulator["policy_matches"] / accumulator["policy_total"]
            if accumulator["policy_total"]
            else None
        ),
        "policy_exact_accuracy": (
            accumulator["policy_exact"] / accumulator["world_states"]
            if accumulator["world_states"]
            else None
        ),
        "world_channel_accuracy": {
            name: (
                values["matches"] / values["total"]
                if values["total"]
                else None
            )
            for name, values in accumulator["channels"].items()
        },
        "world_object_exact_accuracy": {
            name: (
                values["matches"] / values["total"]
                if values["total"]
                else None
            )
            for name, values in accumulator["objects"].items()
        },
        "world_scalar_accuracy": (
            accumulator["scalars"]["matches"]
            / accumulator["scalars"]["total"]
            if accumulator["scalars"]["total"]
            else None
        ),
        "reward_mae": _mean_or_none(accumulator["reward_errors"]),
        "terminal_reward_mae": _mean_or_none(
            accumulator["terminal_reward_errors"]
        ),
        "nonterminal_reward_mae": _mean_or_none(
            accumulator["nonterminal_reward_errors"]
        ),
        "done_threshold_sweep": done_threshold_sweep,
        "death_risk_samples": len(death_true),
        "death_threshold_sweep": death_threshold_sweep,
        "suicide_risk_samples": len(suicide_true),
        "suicide_threshold_sweep": suicide_threshold_sweep,
        "legal_actions": legal,
    }


def select_starts(episodes, max_starts, rng):
    starts = [
        (episode_index, step_index)
        for episode_index, episode in enumerate(episodes)
        for step_index in range(len(episode))
    ]
    if max_starts is not None and len(starts) > max_starts:
        starts = rng.sample(starts, max_starts)
    return starts


def evaluate_bomb_outcomes(world_model):
    """Report calibration and classification of delayed bomb outcomes."""

    world_model.ensure_bomb_outcome_state()
    if not world_model.is_bomb_outcome_ready:
        return None
    outcomes = list(world_model.bomb_outcome_replay)
    predictions = [
        world_model.predict_bomb_outcome(outcome.state)
        for outcome in outcomes
    ]
    targets = {
        "survived": [outcome.survived for outcome in outcomes],
        "crate": [outcome.destroyed_crate for outcome in outcomes],
        "kill": [outcome.killed_opponent for outcome in outcomes],
        "escape": [outcome.opponent_escaped for outcome in outcomes],
    }
    probabilities = {
        "survived": [item.survived_probability for item in predictions],
        "crate": [item.crate_probability for item in predictions],
        "kill": [item.kill_probability for item in predictions],
        "escape": [item.escape_probability for item in predictions],
    }
    report = {"samples": len(outcomes), "targets": {}}
    for name in targets:
        truth = np.asarray(targets[name], dtype=bool)
        probability = np.asarray(probabilities[name], dtype=np.float64)
        report["targets"][name] = {
            "positive_samples": int(np.sum(truth)),
            "positive_rate": float(np.mean(truth)),
            "brier_score": float(np.mean((probability - truth) ** 2)),
            "mean_probability_positive": (
                float(np.mean(probability[truth]))
                if np.any(truth)
                else None
            ),
            "mean_probability_negative": (
                float(np.mean(probability[~truth]))
                if np.any(~truth)
                else None
            ),
            "threshold_sweep": {
                str(threshold): confusion_metrics(
                    truth,
                    probability >= threshold,
                )
                for threshold in (0.1, 0.15, 0.25, 0.5)
            },
        }
    return report


def evaluate(world_model, max_horizon=5, max_starts=5_000, seed=0):
    if not isinstance(world_model, LearnedWorldModel):
        raise TypeError("Checkpoint does not contain a LearnedWorldModel.")
    if not world_model.is_ready:
        raise ValueError("The saved world model has not completed warm-up.")

    episodes = completed_episodes(world_model)
    if not episodes:
        raise ValueError("No complete replay episodes could be reconstructed.")

    starts = select_starts(episodes, max_starts, random.Random(seed))
    horizons = {
        horizon: empty_accumulator()
        for horizon in range(1, max_horizon + 1)
    }
    by_action = {
        action: empty_accumulator()
        for action in ACTIONS
    }

    for episode_index, step_index in starts:
        episode = episodes[episode_index]
        predicted_policy_state = episode[step_index].state
        predicted_world_state = episode[step_index].world_state

        for offset in range(max_horizon):
            actual_index = step_index + offset
            if actual_index >= len(episode):
                break
            experience = episode[actual_index]
            prediction = world_model.predict(
                predicted_world_state,
                predicted_policy_state,
                experience.action_index,
            )
            next_policy_state = prediction.next_state
            predicted_valid_actions = prediction.next_valid_actions
            accumulator = horizons[offset + 1]
            add_prediction(
                accumulator,
                prediction,
                next_policy_state,
                predicted_valid_actions,
                experience,
            )
            if offset == 0:
                add_prediction(
                    by_action[ACTIONS[experience.action_index]],
                    prediction,
                    next_policy_state,
                    predicted_valid_actions,
                    experience,
                )

            predicted_policy_state = next_policy_state
            predicted_world_state = prediction.next_world_state
            if experience.done:
                break

    return {
        "replay_transitions": len(world_model),
        "complete_episodes": len(episodes),
        "evaluated_starts": len(starts),
        "max_horizon": max_horizon,
        "training_steps": world_model.training_steps,
        "last_training_loss": world_model.last_loss,
        "bomb_outcomes": evaluate_bomb_outcomes(world_model),
        "horizons": {
            str(horizon): finalize(accumulator)
            for horizon, accumulator in horizons.items()
        },
        "one_step_by_action": {
            action: finalize(accumulator)
            for action, accumulator in by_action.items()
        },
    }


def _percentage(value):
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _number(value):
    return "n/a" if value is None else f"{value:.3f}"


def print_report(report):
    print(
        f"Replay: {report['replay_transitions']} transitions, "
        f"{report['complete_episodes']} complete episodes, "
        f"{report['evaluated_starts']} rollout starts"
    )
    print()
    print("H  world    policy   exact   reward-MAE  done-P/R    legal-F1")
    for horizon, metrics in report["horizons"].items():
        done = metrics["done_threshold_sweep"]["0.5"]
        print(
            f"{int(horizon):>1}  "
            f"{_percentage(metrics['world_feature_accuracy']):>7}  "
            f"{_percentage(metrics['policy_feature_accuracy']):>7}  "
            f"{_percentage(metrics['policy_exact_accuracy']):>6}  "
            f"{_number(metrics['reward_mae']):>10}  "
            f"{_percentage(done['precision'])}/"
            f"{_percentage(done['recall'])}  "
            f"{_percentage(metrics['legal_actions']['f1']):>8}"
        )

    print()
    print("Learned death/suicide risk (current action + four future steps)")
    print("H  target    samples  precision  recall    F1")
    for horizon, metrics in report["horizons"].items():
        for target in ("death", "suicide"):
            risk = metrics[f"{target}_threshold_sweep"]["0.5"]
            print(
                f"{int(horizon):>1}  {target:<7}  "
                f"{metrics[f'{target}_risk_samples']:>7}  "
                f"{_percentage(risk['precision']):>9}  "
                f"{_percentage(risk['recall']):>6}  "
                f"{_percentage(risk['f1']):>6}"
            )

    bomb_outcomes = report.get("bomb_outcomes")
    if bomb_outcomes is not None:
        print()
        print("Delayed bomb outcomes (in-replay calibration)")
        print("target    pos/all  p(pos)  p(neg)  Brier   P/R/F1 @ 0.5")
        for target, metrics in bomb_outcomes["targets"].items():
            threshold = metrics["threshold_sweep"]["0.5"]
            print(
                f"{target:<8}  {metrics['positive_samples']:>4}/"
                f"{bomb_outcomes['samples']:<4}  "
                f"{_number(metrics['mean_probability_positive']):>6}  "
                f"{_number(metrics['mean_probability_negative']):>6}  "
                f"{_number(metrics['brier_score']):>5}  "
                f"{_percentage(threshold['precision'])}/"
                f"{_percentage(threshold['recall'])}/"
                f"{_percentage(threshold['f1'])}"
            )

    print()
    print("One-step world channels")
    for name, accuracy in report["horizons"]["1"][
        "world_channel_accuracy"
    ].items():
        print(f"  {name:<22} {_percentage(accuracy)}")

    print()
    print("One-step world objects (whole object exactly correct)")
    for name, accuracy in report["horizons"]["1"][
        "world_object_exact_accuracy"
    ].items():
        print(f"  {name:<22} {_percentage(accuracy)}")

    print()
    print("One-step by action")
    print("action  samples  policy   reward-MAE  done-R")
    for action, metrics in report["one_step_by_action"].items():
        done = metrics["done_threshold_sweep"]["0.5"]
        print(
            f"{action:<6}  {metrics['samples']:>7}  "
            f"{_percentage(metrics['policy_feature_accuracy']):>7}  "
            f"{_number(metrics['reward_mae']):>10}  "
            f"{_percentage(done['recall']):>6}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved full-state Dyna world model."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-horizon", type=int, default=5)
    parser.add_argument("--max-starts", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    with args.checkpoint.open("rb") as file:
        saved_data = pickle.load(file)
    schema = saved_data.get("model_schema_version")
    if schema != MODEL_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema {schema} is incompatible with current "
            f"schema {MODEL_SCHEMA_VERSION}."
        )

    report = evaluate(
        saved_data["world_model"],
        max_horizon=args.max_horizon,
        max_starts=args.max_starts,
        seed=args.seed,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["model_schema_version"] = schema
    print_report(report)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        print(f"\nJSON written to {args.json_output}")


if __name__ == "__main__":
    main()
