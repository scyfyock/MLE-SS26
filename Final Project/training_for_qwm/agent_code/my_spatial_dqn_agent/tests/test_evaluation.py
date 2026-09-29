"""
Tests for evaluate.py and evaluation metrics calculation.
"""
import pytest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate import compute_metrics, render_table, visible_len, resolve_model_path, evaluate_model


def test_visible_len_strips_ansi():
    raw_text = "\033[92m+15.5%\033[0m"
    assert visible_len(raw_text) == 6
    assert visible_len("regular text") == 12


def test_render_table():
    headers = ["Agent", "Win %", "Score"]
    rows = [
        ["agent_a", "60.0%", "5.5"],
        ["agent_b", "40.0%", "3.0"],
    ]
    table_str = render_table(headers, rows, title="Test Table")
    assert "AGENT" in table_str or "Agent" in table_str
    assert "agent_a" in table_str
    assert "60.0%" in table_str


def test_resolve_model_path():
    path, desc = resolve_model_path("my_spatial_dqn_agent")
    assert path.is_file()
    assert path.suffix == ".pt"
    assert "Stage 4" in desc or "final model" in desc or "my-saved-model" in desc


def test_compute_metrics():
    mock_stats = {
        "by_agent": {
            "my_agent": {"survival_wins": 1, "score_wins": 1},
            "rule_based_agent_0": {"survival_wins": 0, "score_wins": 0},
            "rule_based_agent_1": {"survival_wins": 0, "score_wins": 0},
            "rule_based_agent_2": {"survival_wins": 0, "score_wins": 0},
        },
        "by_round": {
            "Round 1": {
                "steps": 100,
                "by_agent": {
                    "my_agent": {
                        "score": 5, "coins": 3, "kills": 1, "suicides": 0,
                        "got_killed": 0, "bombs": 4, "waited": 6, "moves": 90,
                        "crates": 12, "invalid": 0, "steps": 100, "survived": 1,
                        "won": 1, "dead": False
                    },
                    "rule_based_agent_0": {
                        "score": 1, "coins": 1, "kills": 0, "suicides": 0,
                        "got_killed": 1, "bombs": 2, "waited": 0, "moves": 40,
                        "crates": 4, "invalid": 1, "steps": 43, "survived": 0,
                        "won": 0, "dead": True
                    },
                    "rule_based_agent_1": {
                        "score": 0, "coins": 0, "kills": 0, "suicides": 1,
                        "got_killed": 0, "bombs": 1, "waited": 0, "moves": 20,
                        "crates": 2, "invalid": 0, "steps": 21, "survived": 0,
                        "won": 0, "dead": True
                    },
                    "rule_based_agent_2": {
                        "score": 2, "coins": 2, "kills": 0, "suicides": 0,
                        "got_killed": 0, "bombs": 3, "waited": 1, "moves": 96,
                        "crates": 8, "invalid": 0, "steps": 100, "survived": 1,
                        "won": 0, "dead": False
                    },
                }
            }
        }
    }

    metrics = compute_metrics(mock_stats, "my_agent")
    ea = metrics["evaluated_agent"]
    rb = metrics["rule_based_avg"]

    assert metrics["n_rounds"] == 1
    assert ea["win_rate_pct"] == 100.0
    assert ea["survival_rate_pct"] == 100.0
    assert ea["total_score"] == 5
    assert ea["total_coins"] == 3
    assert ea["total_kills"] == 1
    assert ea["total_suicides"] == 0
    assert ea["total_got_killed"] == 0
    assert ea["kdr"] == 1.0
    assert ea["waited_total"] == 6
    assert ea["bombs_total"] == 4
    assert ea["moves_total"] == 90
    assert ea["total_actions"] == 100

    # Rule based composite checks
    assert rb["coins_per_round"] == pytest.approx((1 + 0 + 2) / 3)
    assert metrics["deltas"]["win_rate_delta"] == 100.0


def test_evaluate_model_fast_run():
    metrics = evaluate_model(
        agent_name="my_spatial_dqn_agent",
        n_rounds=1,
        no_save=True,
    )
    assert metrics["n_rounds"] == 1
    assert "evaluated_agent" in metrics
    assert "rule_based_avg" in metrics

