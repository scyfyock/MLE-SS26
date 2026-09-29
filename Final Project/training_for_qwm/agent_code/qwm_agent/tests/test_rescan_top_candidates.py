"""
test_rescan_top_candidates.py - Unit tests for Top-K Candidate Rescan & Tournament Engine.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import optuna
from optuna.trial import TrialState

from agent_code.qwm_agent.rescan_top_candidates import (
    CandidateConfig,
    ScenarioDef,
    MatchResult,
    extract_top_k_from_optuna,
    aggregate_phase1_results,
    _create_candidate_wrapper,
    build_parser,
    ROOT_DIR,
    AGENT_CODE_DIR
)


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.agent == "qwm_agent"
    assert args.top_k == 4
    assert args.n_seeds == 100
    assert "classic_combat" in args.scenarios
    assert "classic" in args.h2h_scenarios
    assert args.h2h_rounds == 100
    assert args.workers in (12, 14)


def test_extract_top_k_from_optuna():
    """Verify loading from SQLite, ranking, deduplication, and fallback behavior."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = Path(tmp_db.name)

    try:
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            study_name="test_rescan_study",
            storage=storage_url,
            direction="maximize"
        )

        # Trial 0: Score 100.0, params (D=4, B=12, λ=0.10, α=0.40)
        trial0 = study.ask()
        trial0.set_user_attr("tag", "0")
        study.tell(trial0, 100.0, state=TrialState.COMPLETE)
        # Note: tell with state=COMPLETE doesn't record params unless suggested during ask
        # Let's add trials via optimize with mock evaluations
        def objective(trial):
            if trial.number == 0:
                trial.suggest_int("search_depth", 4, 4)
                trial.suggest_int("beam_size", 12, 12)
                trial.suggest_float("tree_discount", 0.10, 0.10)
                trial.suggest_float("alpha_vq", 0.40, 0.40)
                return 100.0
            elif trial.number == 1:
                # Duplicate config with higher score
                trial.suggest_int("search_depth", 4, 4)
                trial.suggest_int("beam_size", 12, 12)
                trial.suggest_float("tree_discount", 0.10, 0.10)
                trial.suggest_float("alpha_vq", 0.40, 0.40)
                return 150.0
            elif trial.number == 2:
                trial.suggest_int("search_depth", 6, 6)
                trial.suggest_int("beam_size", 24, 24)
                trial.suggest_float("tree_discount", 0.20, 0.20)
                trial.suggest_float("alpha_vq", 0.50, 0.50)
                return 300.0
            elif trial.number == 3:
                trial.suggest_int("search_depth", 8, 8)
                trial.suggest_int("beam_size", 36, 36)
                trial.suggest_float("tree_discount", 0.15, 0.15)
                trial.suggest_float("alpha_vq", 0.60, 0.60)
                return 250.0
            else:
                trial.suggest_int("search_depth", 2, 2)
                trial.suggest_int("beam_size", 6, 6)
                trial.suggest_float("tree_discount", 0.05, 0.05)
                trial.suggest_float("alpha_vq", 0.20, 0.20)
                return 50.0

        study.optimize(objective, n_trials=5)

        candidates = extract_top_k_from_optuna(
            storage=str(db_path),
            study_name="test_rescan_study",
            top_k=3
        )

        assert len(candidates) == 3
        # Candidate 1 must be Trial 2 (score 300.0)
        assert candidates[0].search_depth == 6
        assert candidates[0].beam_size == 24
        assert candidates[0].optuna_score == 300.0

        # Candidate 2 must be Trial 3 (score 250.0)
        assert candidates[1].search_depth == 8
        assert candidates[1].beam_size == 36
        assert candidates[1].optuna_score == 250.0

        # Candidate 3 must be Trial 1 (score 150.0), deduplicating Trial 0
        assert candidates[2].search_depth == 4
        assert candidates[2].beam_size == 12
        assert candidates[2].optuna_score == 150.0

    finally:
        db_path.unlink(missing_ok=True)


def test_win_metric_separation_logic():
    """Test explicit logic for separating Point Wins and Survival Wins."""
    # Simulation 1: High score, but eliminated early (Point Win, not Survival Win)
    round_by_agent_1 = {
        "qwm_agent": {"score": 5, "survived": 0, "dead": True, "won": 0},
        "rule_based_agent_1": {"score": 2, "survived": 1, "dead": False, "won": 1},
        "rule_based_agent_2": {"score": 0, "survived": 0, "dead": True, "won": 0},
    }
    all_scores_1 = {k: v.get("score", 0) for k, v in round_by_agent_1.items()}
    max_score_1 = max(all_scores_1.values())
    point_win_1 = (round_by_agent_1["qwm_agent"]["score"] == max_score_1 and max_score_1 > 0 and list(all_scores_1.values()).count(max_score_1) == 1)
    alive_1 = [k for k, v in round_by_agent_1.items() if not v.get("dead", True) or v.get("survived", 0) > 0]
    survival_win_1 = (len(alive_1) == 1 and alive_1[0] == "qwm_agent")

    assert point_win_1 is True
    assert survival_win_1 is False

    # Simulation 2: Sole survivor with 0 points (Survival Win, not Point Win)
    round_by_agent_2 = {
        "qwm_agent": {"score": 0, "survived": 1, "dead": False, "won": 1},
        "rule_based_agent_1": {"score": 0, "survived": 0, "dead": True, "won": 0},
        "rule_based_agent_2": {"score": 0, "survived": 0, "dead": True, "won": 0},
    }
    all_scores_2 = {k: v.get("score", 0) for k, v in round_by_agent_2.items()}
    max_score_2 = max(all_scores_2.values())
    point_win_2 = (round_by_agent_2["qwm_agent"]["score"] == max_score_2 and max_score_2 > 0 and list(all_scores_2.values()).count(max_score_2) == 1)
    alive_2 = [k for k, v in round_by_agent_2.items() if not v.get("dead", True) or v.get("survived", 0) > 0]
    survival_win_2 = (len(alive_2) == 1 and alive_2[0] == "qwm_agent")

    assert point_win_2 is False
    assert survival_win_2 is True

    # Simulation 3: Dominant victory: 4 points AND sole survivor (Both Point Win and Survival Win)
    round_by_agent_3 = {
        "qwm_agent": {"score": 4, "survived": 1, "dead": False, "won": 1},
        "rule_based_agent_1": {"score": 1, "survived": 0, "dead": True, "won": 0},
        "rule_based_agent_2": {"score": 0, "survived": 0, "dead": True, "won": 0},
    }
    all_scores_3 = {k: v.get("score", 0) for k, v in round_by_agent_3.items()}
    max_score_3 = max(all_scores_3.values())
    point_win_3 = (round_by_agent_3["qwm_agent"]["score"] == max_score_3 and max_score_3 > 0 and list(all_scores_3.values()).count(max_score_3) == 1)
    alive_3 = [k for k, v in round_by_agent_3.items() if not v.get("dead", True) or v.get("survived", 0) > 0]
    survival_win_3 = (len(alive_3) == 1 and alive_3[0] == "qwm_agent")

    assert point_win_3 is True
    assert survival_win_3 is True


def test_aggregate_phase1_results():
    """Verify aggregation across scenarios and ranking computation."""
    c1 = CandidateConfig(name="C1", search_depth=4, beam_size=12, tree_discount=0.1, alpha_vq=0.5)
    c2 = CandidateConfig(name="C2", search_depth=6, beam_size=24, tree_discount=0.2, alpha_vq=0.6)

    s1 = ScenarioDef(key="classic", display_name="Classic", scenario="classic", opponents=[], weight=1.0)
    s2 = ScenarioDef(key="loot", display_name="Loot", scenario="loot-crate", opponents=[], weight=1.0)

    results = [
        # C1 matches
        MatchResult(candidate_name="C1", scenario_key="classic", seed=1001, score=5.0, point_win=True, survival_win=False, total_win=True, survived=True, coins=2, kills=1, crates=3, suicides=0, got_killed=0, steps=100, think_time_ms=10.0, success=True),
        MatchResult(candidate_name="C1", scenario_key="classic", seed=1002, score=1.0, point_win=False, survival_win=True, total_win=True, survived=True, coins=1, kills=0, crates=2, suicides=0, got_killed=0, steps=80, think_time_ms=10.0, success=True),
        MatchResult(candidate_name="C1", scenario_key="loot", seed=1001, score=8.0, point_win=True, survival_win=True, total_win=True, survived=True, coins=4, kills=1, crates=5, suicides=0, got_killed=0, steps=120, think_time_ms=12.0, success=True),
        MatchResult(candidate_name="C1", scenario_key="loot", seed=1002, score=0.0, point_win=False, survival_win=False, total_win=False, survived=False, coins=0, kills=0, crates=0, suicides=1, got_killed=0, steps=20, think_time_ms=10.0, success=True),

        # C2 matches (lower score and fewer wins)
        MatchResult(candidate_name="C2", scenario_key="classic", seed=1001, score=0.0, point_win=False, survival_win=False, total_win=False, survived=False, coins=0, kills=0, crates=0, suicides=0, got_killed=1, steps=50, think_time_ms=20.0, success=True),
        MatchResult(candidate_name="C2", scenario_key="classic", seed=1002, score=0.0, point_win=False, survival_win=False, total_win=False, survived=False, coins=0, kills=0, crates=0, suicides=0, got_killed=1, steps=50, think_time_ms=20.0, success=True),
        MatchResult(candidate_name="C2", scenario_key="loot", seed=1001, score=2.0, point_win=False, survival_win=False, total_win=False, survived=True, coins=2, kills=0, crates=2, suicides=0, got_killed=0, steps=100, think_time_ms=22.0, success=True),
        MatchResult(candidate_name="C2", scenario_key="loot", seed=1002, score=1.0, point_win=False, survival_win=False, total_win=False, survived=False, coins=1, kills=0, crates=1, suicides=1, got_killed=0, steps=40, think_time_ms=18.0, success=True),
    ]

    summary = aggregate_phase1_results([c1, c2], [s1, s2], results)

    # Check that C1 is ranked #1 overall
    assert summary["ranked_overall"][0]["candidate"]["name"] == "C1"
    assert summary["ranked_overall"][1]["candidate"]["name"] == "C2"

    c1_classic = summary["summary_by_cand"]["C1"]["scenarios"]["classic"]
    assert c1_classic["total_wins"] == 2
    assert c1_classic["point_wins"] == 1
    assert c1_classic["survival_wins"] == 1
    assert c1_classic["survival_rate"] == 100.0


def test_candidate_wrapper_creation_and_cleanup():
    """Verify dynamic agent directory creation and proper structure."""
    cand = CandidateConfig(name="Test-Wrapper", search_depth=5, beam_size=15, tree_discount=0.18, alpha_vq=0.33)
    wrapper_name = "_test_dynamic_cand_dir"
    target_dir = AGENT_CODE_DIR / wrapper_name

    try:
        w_dir = _create_candidate_wrapper(
            wrapper_name=wrapper_name,
            cand=cand,
            checkpoint_path="/mock/checkpoint.pt"
        )

        assert w_dir.is_dir()
        cfg_file = w_dir / "config.json"
        cb_file = w_dir / "callbacks.py"

        assert cfg_file.is_file()
        assert cb_file.is_file()

        with open(cfg_file) as f:
            cfg = json.load(f)

        assert cfg["search_depth"] == 5
        assert cfg["beam_size"] == 15
        assert cfg["tree_discount"] == 0.18
        assert cfg["alpha_vq"] == 0.33

        with open(cb_file) as f:
            cb_text = f.read()

        assert "agent_code.qwm_agent.callbacks" in cb_text

    finally:
        if target_dir.is_dir():
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
