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

from agent_code.qwm_agent.tune_inference import (
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
        assert 1 <= c.search_depth <= 10
        assert 6 <= c.beam_size <= 48
        assert 0.05 <= c.tree_discount <= 0.40
        assert 0.05 <= c.alpha_vq <= 0.95
        # Ensure no heuristic penalties are present on the candidate
        assert not hasattr(c, "predicted_wait_penalty")
        assert not hasattr(c, "predicted_loop_penalty")

    std_cands = get_candidate_presets("standard")
    assert len(std_cands) >= 8

    thorough_cands = get_candidate_presets("thorough")
    assert len(thorough_cands) >= 12


def test_scenarios_include_dqn_adversary():
    dqn_scenarios = [s for s in DEFAULT_SCENARIOS if "my_spatial_dqn_agent" in s.opponents]
    assert len(dqn_scenarios) >= 2, "Expected at least 2 scenarios with my_spatial_dqn_agent as enemy"


def test_candidate_wrapper_lifecycle():
    cand = InferenceCandidate(
        name="Test-Pure-D3-B12",
        search_depth=3,
        beam_size=12,
        tree_discount=0.15,
        alpha_vq=0.50
    )
    checkpoint_p = resolve_checkpoint()
    wrapper_name = "_test_qwm_wrapper_cand"
    target_dir = AGENT_CODE_DIR / wrapper_name

    try:
        created = _create_candidate_agent_wrapper(wrapper_name, cand, str(checkpoint_p))
        assert created.is_dir()
        assert (created / "config.json").is_file()
        assert (created / "callbacks.py").is_file()

        with open(created / "config.json") as f:
            cfg = json.load(f)
        assert cfg["search_depth"] == 3
        assert cfg["beam_size"] == 12
        assert cfg["tree_discount"] == 0.15
        assert cfg["alpha_vq"] == 0.50
        assert "predicted_wait_penalty" not in cfg
        assert "predicted_loop_penalty" not in cfg

        with open(created / "callbacks.py") as f:
            cb_text = f.read()
        assert "agent_code.qwm_agent.callbacks" in cb_text

    finally:
        if target_dir.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
        assert not target_dir.exists()


def test_resolve_checkpoint():
    ckpt = resolve_checkpoint()
    assert Path(ckpt).is_file()
    assert Path(ckpt).suffix == ".pt"
    assert "qwm_agent" in str(ckpt)


def test_callbacks_env_var_override():
    import types
    import agent_code.qwm_agent.callbacks as qwm_cb

    dummy_self = types.SimpleNamespace()

    old_env = {
        "MY_QWM_SEARCH_DEPTH": os.environ.get("MY_QWM_SEARCH_DEPTH"),
        "MY_QWM_BEAM_SIZE": os.environ.get("MY_QWM_BEAM_SIZE"),
        "MY_QWM_TREE_DISCOUNT": os.environ.get("MY_QWM_TREE_DISCOUNT"),
        "MY_QWM_ALPHA_VQ": os.environ.get("MY_QWM_ALPHA_VQ"),
    }

    try:
        os.environ["MY_QWM_SEARCH_DEPTH"] = "5"
        os.environ["MY_QWM_BEAM_SIZE"] = "18"
        os.environ["MY_QWM_TREE_DISCOUNT"] = "0.22"
        os.environ["MY_QWM_ALPHA_VQ"] = "0.65"

        qwm_cb.setup(dummy_self)

        assert dummy_self.search_depth == 5
        assert dummy_self.beam_size == 18
        assert abs(dummy_self.tree_discount - 0.22) < 1e-5
        assert abs(dummy_self.alpha_vq - 0.65) < 1e-5
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_optuna_integration():
    from agent_code.qwm_agent.tune_inference import (
        OPTUNA_AVAILABLE,
        tune_with_optuna,
        DEFAULT_SCENARIOS
    )
    assert OPTUNA_AVAILABLE, "Expected optuna to be installed and available"

    ckpt = resolve_checkpoint()
    scens = [DEFAULT_SCENARIOS[0]]  # classic_combat
    seeds = [1000]

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "unit_test_qwm.db"
        res = tune_with_optuna(
            checkpoint_path=str(ckpt),
            scenarios=scens,
            seeds=seeds,
            n_trials=1,
            n_workers=1,
            storage=f"sqlite:///{test_db}",
            study_name="test_integration_qwm",
            prune=False,
            verbose=False
        )
        assert res["strategy"] == "optuna"
        assert len(res["ranked_candidates"]) == 1
        best_cand = res["ranked_candidates"][0]["candidate"]
        assert 3 <= best_cand["search_depth"] <= 12
        assert 6 <= best_cand["beam_size"] <= 48
        assert 0.00 <= best_cand["tree_discount"] <= 0.40
        assert 0.00 <= best_cand["alpha_vq"] <= 0.96
        assert "predicted_wait_penalty" not in best_cand


def test_optuna_persistent_storage_and_resume():
    from agent_code.qwm_agent.tune_inference import (
        OPTUNA_AVAILABLE,
        tune_with_optuna,
        DEFAULT_SCENARIOS
    )
    assert OPTUNA_AVAILABLE

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_study_qwm.db"
        storage_url = f"sqlite:///{db_path}"
        ckpt = resolve_checkpoint()
        scens = [DEFAULT_SCENARIOS[0]]
        seeds = [1000]

        # First run: 1 trial
        res1 = tune_with_optuna(
            checkpoint_path=str(ckpt),
            scenarios=scens,
            seeds=seeds,
            n_trials=1,
            n_workers=1,
            storage=storage_url,
            study_name="test_persistent_study_qwm",
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
            study_name="test_persistent_study_qwm",
            prune=False,
            verbose=False
        )
        assert len(res2["ranked_candidates"]) == 2, "Should accumulate trials across runs from persistent DB"
        cand_names = [c["candidate"]["name"] for c in res2["ranked_candidates"]]
        assert "Optuna-000" in cand_names
        assert "Optuna-001" in cand_names


def test_optuna_concurrent_trial_parallelism():
    from agent_code.qwm_agent.tune_inference import (
        OPTUNA_AVAILABLE,
        tune_with_optuna,
        DEFAULT_SCENARIOS
    )
    assert OPTUNA_AVAILABLE

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "concurrent_qwm.db"
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
            study_name="test_concurrent_study_qwm",
            prune=False,
            verbose=False
        )
        assert len(res["ranked_candidates"]) == 2

