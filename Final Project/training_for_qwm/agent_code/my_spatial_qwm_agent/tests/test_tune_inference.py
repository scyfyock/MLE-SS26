import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import pytest

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_code.my_spatial_qwm_agent.tune_inference import (
    InferenceCandidate,
    ScenarioDef,
    get_candidate_presets,
    DEFAULT_SCENARIOS,
    _create_candidate_agent_wrapper,
    evaluate_candidates_matched,
    resolve_checkpoint,
    ROOT_DIR,
    AGENT_CODE_DIR
)


def test_candidate_presets():
    quick_cands = get_candidate_presets("quick")
    assert len(quick_cands) >= 4
    for c in quick_cands:
        assert 1 <= c.search_depth <= 8
        assert 6 <= c.beam_size <= 60
        assert 0.05 <= c.tree_discount <= 0.5
        assert 0.1 <= c.alpha_vq <= 0.9

    std_cands = get_candidate_presets("standard")
    assert len(std_cands) >= 8

    thorough_cands = get_candidate_presets("thorough")
    assert len(thorough_cands) >= 12


def test_scenarios_include_dqn_adversary():
    dqn_scenarios = [s for s in DEFAULT_SCENARIOS if "my_spatial_dqn_agent" in s.opponents]
    assert len(dqn_scenarios) >= 2, "Expected at least 2 scenarios with my_spatial_dqn_agent as enemy"


def test_candidate_wrapper_lifecycle():
    cand = InferenceCandidate(
        name="Test-D2-B8",
        search_depth=2,
        beam_size=8,
        tree_discount=0.14,
        alpha_vq=0.48,
        predicted_wait_penalty=1.2,
        predicted_loop_penalty=2.4
    )
    checkpoint_p = resolve_checkpoint()
    wrapper_name = "_test_wrapper_cand"
    target_dir = AGENT_CODE_DIR / wrapper_name

    try:
        created = _create_candidate_agent_wrapper(wrapper_name, cand, str(checkpoint_p))
        assert created.is_dir()
        assert (created / "config.json").is_file()
        assert (created / "callbacks.py").is_file()

        with open(created / "config.json") as f:
            cfg = json.load(f)
        assert cfg["search_depth"] == 2
        assert cfg["beam_size"] == 8
        assert cfg["tree_discount"] == 0.14
        assert cfg["alpha_vq"] == 0.48
        assert cfg["predicted_wait_penalty"] == 1.2
        assert cfg["predicted_loop_penalty"] == 2.4

    finally:
        if target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
        assert not target_dir.exists()


def test_resolve_checkpoint():
    ckpt = resolve_checkpoint()
    assert Path(ckpt).is_file()
    assert Path(ckpt).suffix == ".pt"


def test_optuna_integration():
    from agent_code.my_spatial_qwm_agent.tune_inference import (
        OPTUNA_AVAILABLE,
        tune_with_optuna,
        DEFAULT_SCENARIOS
    )
    assert OPTUNA_AVAILABLE, "Expected optuna to be installed and available"

    ckpt = resolve_checkpoint()
    scens = [DEFAULT_SCENARIOS[0]]  # classic_combat
    seeds = [1000]

    # Run 1 Optuna trial with 1 worker as a quick unit test with isolated db
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "unit_test.db"
        res = tune_with_optuna(
            checkpoint_path=str(ckpt),
            scenarios=scens,
            seeds=seeds,
            n_trials=1,
            n_workers=1,
            storage=f"sqlite:///{test_db}",
            study_name="test_integration",
            prune=False,
            verbose=False
        )
        assert res["strategy"] == "optuna"
        assert len(res["ranked_candidates"]) == 1
        best_cand = res["ranked_candidates"][0]["candidate"]
        assert 1 <= best_cand["search_depth"] <= 8
        assert 6 <= best_cand["beam_size"] <= 36
        assert 0.05 <= best_cand["tree_discount"] <= 0.40
        assert 0.10 <= best_cand["alpha_vq"] <= 0.90


def test_optuna_persistent_storage_and_resume():
    from agent_code.my_spatial_qwm_agent.tune_inference import (
        OPTUNA_AVAILABLE,
        tune_with_optuna,
        DEFAULT_SCENARIOS
    )
    assert OPTUNA_AVAILABLE

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_study.db"
        storage_url = f"sqlite:///{db_path}"
        ckpt = resolve_checkpoint()
        scens = [DEFAULT_SCENARIOS[0]]  # classic_combat
        seeds = [1000]

        # First run: 1 trial
        res1 = tune_with_optuna(
            checkpoint_path=str(ckpt),
            scenarios=scens,
            seeds=seeds,
            n_trials=1,
            n_workers=1,
            storage=storage_url,
            study_name="test_persistent_study",
            prune=False,
            verbose=False
        )
        assert db_path.is_file(), "Database file should exist after first run"
        assert len(res1["ranked_candidates"]) == 1

        # Second run: 1 more trial on the same persistent DB
        res2 = tune_with_optuna(
            checkpoint_path=str(ckpt),
            scenarios=scens,
            seeds=seeds,
            n_trials=1,
            n_workers=1,
            storage=storage_url,
            study_name="test_persistent_study",
            prune=False,
            verbose=False
        )
        assert len(res2["ranked_candidates"]) == 2, "Should accumulate trials across runs from persistent DB"
        cand_names = [c["candidate"]["name"] for c in res2["ranked_candidates"]]
        assert "Optuna-000" in cand_names
        assert "Optuna-001" in cand_names


def test_optuna_concurrent_trial_parallelism():
    from agent_code.my_spatial_qwm_agent.tune_inference import (
        OPTUNA_AVAILABLE,
        tune_with_optuna,
        DEFAULT_SCENARIOS
    )
    assert OPTUNA_AVAILABLE

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "concurrent.db"
        storage_url = f"sqlite:///{db_path}"
        ckpt = resolve_checkpoint()
        scens = [DEFAULT_SCENARIOS[0]]
        seeds = [1000]

        # Run 2 trials concurrently with n_jobs=2 sharing 2 workers
        res = tune_with_optuna(
            checkpoint_path=str(ckpt),
            scenarios=scens,
            seeds=seeds,
            n_trials=2,
            n_jobs=2,
            n_workers=2,
            storage=storage_url,
            study_name="test_concurrent_study",
            prune=False,
            verbose=False
        )
        assert len(res["ranked_candidates"]) == 2

