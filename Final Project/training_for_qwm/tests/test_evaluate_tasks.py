#!/usr/bin/env python3
"""
Unit tests for evaluate_tasks.py (Multi-Task Evaluation Benchmark).
"""

import json
from pathlib import Path
import tempfile
import pytest

from evaluate_tasks import (
    TASKS_CATALOG,
    TASK_ALIASES,
    TaskSpec,
    H2HMatchupSpec,
    H2H_MATCHUPS,
    RoundExecutionResult,
    H2HRoundResult,
    build_parser,
    resolve_tasks,
    resolve_agent_model,
    grade_metric,
    summarize_task_results,
    summarize_h2h_results,
    export_markdown_task_report,
    export_rounds_csv,
    export_h2h_csv,
    print_depth_sweep_summary_table,
    generate_depth_comparison_plots,
    render_table,
    TaskProgressBar,
    MATPLOTLIB_AVAILABLE,
)


def test_task_catalog_specifications():
    """Verify all 4 core tasks specified in final_project.pdf Section 4 exist."""
    assert "1" in TASKS_CATALOG
    assert "2" in TASKS_CATALOG
    assert "3" in TASKS_CATALOG
    assert "4" in TASKS_CATALOG

    # Task 1: Coin Navigation
    t1 = TASKS_CATALOG["1"]
    assert t1.scenario == "coin-heaven"
    assert len(t1.opponents) == 0

    # Task 2: Crate Clearing
    t2 = TASKS_CATALOG["2"]
    assert t2.scenario == "loot-crate"
    assert len(t2.opponents) == 0

    # Task 3: Hunting Passive & Collector
    t3 = TASKS_CATALOG["3"]
    assert t3.scenario == "classic"
    assert "peaceful_agent" in t3.opponents
    assert "coin_collector_agent" in t3.opponents

    # Task 4: Full Competitive Tournament
    t4 = TASKS_CATALOG["4"]
    assert t4.scenario == "classic"
    assert len(t4.opponents) == 3
    assert all(opp == "rule_based_agent" for opp in t4.opponents)


def test_resolve_tasks_aliases():
    """Verify task alias resolution for CLI inputs."""
    specs_all = resolve_tasks(["all"])
    assert len(specs_all) == 4
    assert [s.task_id for s in specs_all] == [1, 2, 3, 4]

    specs_subset = resolve_tasks(["1", "task2"])
    assert len(specs_subset) == 2
    assert specs_subset[0].key == "task1"
    assert specs_subset[1].key == "task2"

    specs_combat = resolve_tasks(["combat"])
    assert len(specs_combat) == 1
    assert specs_combat[0].key == "task4"


def test_cli_parser_defaults():
    """Verify CLI arguments and default parameters."""
    parser = build_parser()
    args = parser.parse_args([])
    assert args.agents in (["qwm_agent_veryclean"], ["qwm_agent_clean"])
    assert args.tasks == ["all"]
    assert args.rounds == 20
    assert args.seed == 1000
    assert args.multiprocessing is True
    assert args.progress is True
    assert args.output_dir is not None


def test_cli_parser_multiprocessing_and_progress_toggles():
    """Verify flags for toggling multiprocessing and progress bar."""
    parser = build_parser()
    args_no_mp = parser.parse_args(["--no-multiprocessing", "--no-progress", "-w", "4"])
    assert args_no_mp.multiprocessing is False
    assert args_no_mp.progress is False
    assert args_no_mp.workers == 4

    args_mp = parser.parse_args(["--parallel", "--progress"])
    assert args_mp.multiprocessing is True
    assert args_mp.progress is True


def test_task_progress_bar_lifecycle():
    """Verify TaskProgressBar updates and graceful closure."""
    pbar = TaskProgressBar(total=5, desc="Test Task", enabled=True)
    pbar.update(1, postfix={"win": "100%", "pts": "50.0"})
    pbar.update(2, postfix={"win": "66%", "pts": "45.0"})
    assert pbar.completed == 3
    pbar.close()

    # Disabled progress bar
    pbar_disabled = TaskProgressBar(total=5, desc="Disabled", enabled=False)
    pbar_disabled.update(1)
    assert pbar_disabled.completed == 0
    pbar_disabled.close()


def test_grade_metric_thresholds():
    """Verify grade assignment across performance thresholds."""
    grade_a_plus = grade_metric(45.0, (42.0, 35.0, 25.0, 15.0))
    assert grade_a_plus == "A+"

    grade_b = grade_metric(30.0, (42.0, 35.0, 25.0, 15.0))
    assert grade_b == "B"

    grade_low = grade_metric(10.0, (42.0, 35.0, 25.0, 15.0))
    assert grade_low == "Needs Improvement"


def test_summarize_task_results_aggregation():
    """Verify statistical aggregation of round results."""
    spec = TASKS_CATALOG["1"]
    r1 = RoundExecutionResult(
        agent_name="test_agent", task_id=1, task_key="task1", round_idx=1, seed=1000,
        score=50.0, won=True, survived=True, coins=50, steps=300, waited=0, bombs=0, suicides=0, got_killed=0,
        success=True
    )
    r2 = RoundExecutionResult(
        agent_name="test_agent", task_id=1, task_key="task1", round_idx=2, seed=1001,
        score=40.0, won=True, survived=True, coins=40, steps=320, waited=2, bombs=0, suicides=0, got_killed=0,
        success=True
    )

    summary = summarize_task_results(spec, [r1, r2])
    assert summary["n_rounds"] == 2
    assert summary["overall_win_rate"] == 100.0
    assert summary["overall_survival_rate"] == 100.0
    assert summary["overall_mean_score"] == 45.0
    assert summary["overall_coins"] == 45.0
    assert summary["overall_suicides"] == 0
    assert summary["grade"] == "A+"


def test_export_markdown_task_report():
    """Verify export utilities produce valid markdown tables."""
    spec = TASKS_CATALOG["1"]
    r1 = RoundExecutionResult(
        agent_name="qwm_agent_clean", task_id=1, task_key="task1", round_idx=1, seed=1000,
        score=50.0, won=True, survived=True, coins=50, steps=300, waited=0, bombs=0, suicides=0, got_killed=0,
        success=True
    )
    summary = summarize_task_results(spec, [r1])
    agent_summaries = {"qwm_agent_clean": {"task1": summary}}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        md_file = tmp_path / "report.md"

        export_markdown_task_report(agent_summaries, md_file)
        assert md_file.is_file()
        md_content = md_file.read_text()
        assert "BombeRLe Multi-Task Evaluation Report" in md_content
        assert "qwm_agent_clean" in md_content
        assert "Task 1: Coin Navigation" in md_content


def test_render_table_formatting():
    """Verify box-drawing table renderer outputs expected structure."""
    headers = ["Task", "Score", "Win %"]
    rows = [["Task 1", "50.0", "100.0%"], ["Task 2", "35.0", "90.0%"]]
    table_str = render_table(headers, rows, title="Test Table")
    assert "TEST TABLE" in table_str
    assert "Task 1" in table_str
    assert "100.0%" in table_str


def test_h2h_matchup_specifications():
    """Verify 1v1 and 4-player head-to-head specifications and task aliases."""
    assert "1v1" in H2H_MATCHUPS
    assert "4p" in H2H_MATCHUPS

    spec_1v1 = H2H_MATCHUPS["1v1"]
    assert spec_1v1.scenario == "classic"
    assert len(spec_1v1.opponents) == 0

    spec_4p = H2H_MATCHUPS["4p"]
    assert spec_4p.scenario == "classic"
    assert len(spec_4p.opponents) == 2
    assert all(opp == "rule_based_agent" for opp in spec_4p.opponents)

    assert "h2h" in TASK_ALIASES
    assert "h2h_1v1" in TASK_ALIASES["h2h"]
    assert "h2h_4p" in TASK_ALIASES["h2h"]


def test_h2h_summary_aggregation():
    """Verify statistical aggregation of direct head-to-head matches."""
    r1 = H2HRoundResult(
        mode="1v1",
        round_idx=1,
        seed=1000,
        agent_a="dyna_agent",
        agent_b="qwm_agent_veryclean",
        winner="dyna_agent",
        a_stats={"score": 5.0, "coins": 5, "crates": 4, "kills": 1, "suicides": 0, "survived": 1},
        b_stats={"score": 2.0, "coins": 2, "crates": 2, "kills": 0, "suicides": 0, "survived": 0},
        a_killed_b=True,
        b_killed_a=False,
        steps=250,
        success=True,
    )
    r2 = H2HRoundResult(
        mode="1v1",
        round_idx=2,
        seed=1031,
        agent_a="dyna_agent",
        agent_b="qwm_agent_veryclean",
        winner="qwm_agent_veryclean",
        a_stats={"score": 1.0, "coins": 1, "crates": 1, "kills": 0, "suicides": 0, "survived": 0},
        b_stats={"score": 6.0, "coins": 4, "crates": 3, "kills": 1, "suicides": 0, "survived": 1},
        a_killed_b=False,
        b_killed_a=True,
        steps=280,
        success=True,
    )

    summary = summarize_h2h_results([r1, r2], "dyna_agent", "qwm_agent_veryclean")
    assert summary["success"] is True
    assert summary["n_rounds"] == 2
    assert summary["a_wins"] == 1
    assert summary["b_wins"] == 1
    assert summary["ties"] == 0
    assert summary["a_win_rate"] == 50.0
    assert summary["b_win_rate"] == 50.0
    assert summary["a_mean_score"] == 3.0
    assert summary["b_mean_score"] == 4.0
    assert summary["score_delta"] == -1.0
    assert summary["a_direct_kills"] == 1
    assert summary["b_direct_kills"] == 1
    assert summary["a_suicides"] == 0
    assert summary["b_suicides"] == 0


def test_cli_parser_h2h_flags():
    """Verify CLI arguments for head-to-head duels."""
    parser = build_parser()
    args = parser.parse_args(["--head-to-head", "--h2h-only", "--h2h-rounds", "50", "--h2h-modes", "1v1"])
    assert args.head_to_head is True
    assert args.h2h_only is True
    assert args.h2h_rounds == 50
    assert args.h2h_modes == ["1v1"]

    args_no_h2h = parser.parse_args(["--no-head-to-head"])
    assert args_no_h2h.head_to_head is False


def test_export_h2h_markdown_and_csv():
    """Verify markdown report and CSV export include head-to-head data."""
    r1 = H2HRoundResult(
        mode="1v1",
        round_idx=1,
        seed=1000,
        agent_a="dyna_agent",
        agent_b="qwm_agent_veryclean",
        winner="dyna_agent",
        a_stats={"score": 5.0, "coins": 5, "crates": 4, "kills": 1, "suicides": 0, "survived": 1},
        b_stats={"score": 2.0, "coins": 2, "crates": 2, "kills": 0, "suicides": 0, "survived": 0},
        a_killed_b=True,
        b_killed_a=False,
        steps=250,
        success=True,
    )
    summary = summarize_h2h_results([r1], "dyna_agent", "qwm_agent_veryclean")
    h2h_summaries = {"1v1": summary}

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        md_file = tmp_path / "report.md"
        csv_file = tmp_path / "h2h_rounds.csv"

        export_markdown_task_report({}, md_file, h2h_summaries=h2h_summaries)
        assert md_file.is_file()
        content = md_file.read_text()
        assert "Direct Head-to-Head Benchmarks" in content
        assert "1v1 Duel (No Opponents)" in content
        assert "dyna_agent" in content
        assert "qwm_agent_veryclean" in content

        export_h2h_csv([r1], csv_file)
        assert csv_file.is_file()
        csv_content = csv_file.read_text()
        assert "dyna_agent" in csv_content
        assert "qwm_agent_veryclean" in csv_content


def test_cli_parser_depth_sweep_flags():
    """Verify CLI flags for planning horizon depth sweeping and plotting."""
    parser = build_parser()
    args_depths = parser.parse_args(["--depths", "1", "2", "3", "4", "--plot", "--plot-style", "seaborn-v0_8-whitegrid"])
    assert args_depths.depths == [1, 2, 3, 4]
    assert args_depths.plot is True
    assert args_depths.plot_style == "seaborn-v0_8-whitegrid"

    args_max = parser.parse_args(["--max-depth", "5", "--no-plot"])
    assert args_max.max_depth == 5
    assert args_max.plot is False


def test_export_rounds_csv_with_depth_and_latency():
    """Verify CSV round log correctly serializes search depth and latency metrics."""
    r1 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean",
        task_id=1,
        task_key="task1",
        round_idx=1,
        seed=1000,
        score=45.0,
        won=True,
        survived=True,
        coins=45,
        crates=0,
        kills=0,
        suicides=0,
        got_killed=0,
        steps=250,
        search_depth=2,
        duration_sec=1.234,
        ms_per_step=4.94,
        success=True,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_file = Path(tmp_dir) / "test_rounds.csv"
        export_rounds_csv([r1], csv_file)
        assert csv_file.is_file()
        content = csv_file.read_text()
        assert "search_depth" in content
        assert "duration_sec" in content
        assert "ms_per_step" in content
        assert "qwm_agent_veryclean" in content
        assert "1.234" in content
        assert "4.94" in content


def test_depth_sweep_summary_table_rendering(capsys):
    """Verify depth sweep console table renders properly with Trade-Off Observations."""
    spec = TASKS_CATALOG["1"]
    r_d1 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean", task_id=1, task_key="task1", round_idx=1, seed=1000,
        score=30.0, won=True, survived=True, coins=30, steps=300, search_depth=1, duration_sec=0.5, ms_per_step=1.67, success=True
    )
    r_d2 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean", task_id=1, task_key="task1", round_idx=1, seed=1000,
        score=42.0, won=True, survived=True, coins=42, steps=280, search_depth=2, duration_sec=1.1, ms_per_step=3.93, success=True
    )

    sum_d1 = summarize_task_results(spec, [r_d1])
    sum_d2 = summarize_task_results(spec, [r_d2])

    depth_summaries = {
        1: {"task1": sum_d1},
        2: {"task1": sum_d2},
    }

    print_depth_sweep_summary_table(depth_summaries, "qwm_agent_veryclean")
    captured = capsys.readouterr().out
    assert "PLANNING HORIZON (SEARCH DEPTH) SWEEP" in captured
    assert "Depth D = 1" in captured
    assert "Depth D = 2" in captured
    assert "Baseline (Shallow)" in captured
    assert "Gain" in captured


def test_generate_depth_comparison_plots():
    """Verify publication-grade multi-panel figure generation with matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        pytest.skip("matplotlib not available")

    spec1 = TASKS_CATALOG["1"]
    spec2 = TASKS_CATALOG["2"]

    # Mock rounds for D=1 and D=2
    r_d1_t1 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean", task_id=1, task_key="task1", round_idx=1, seed=1000,
        score=25.0, won=True, survived=True, coins=25, steps=300, search_depth=1, duration_sec=0.5, ms_per_step=1.67, success=True
    )
    r_d1_t2 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean", task_id=2, task_key="task2", round_idx=1, seed=1000,
        score=20.0, won=True, survived=True, crates=20, bombs=10, steps=350, search_depth=1, duration_sec=0.7, ms_per_step=2.00, success=True
    )

    r_d2_t1 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean", task_id=1, task_key="task1", round_idx=1, seed=1000,
        score=40.0, won=True, survived=True, coins=40, steps=280, search_depth=2, duration_sec=1.1, ms_per_step=3.93, success=True
    )
    r_d2_t2 = RoundExecutionResult(
        agent_name="qwm_agent_veryclean", task_id=2, task_key="task2", round_idx=1, seed=1000,
        score=35.0, won=True, survived=True, crates=35, bombs=12, steps=320, search_depth=2, duration_sec=1.4, ms_per_step=4.38, success=True
    )

    depth_summaries = {
        1: {"task1": summarize_task_results(spec1, [r_d1_t1]), "task2": summarize_task_results(spec2, [r_d1_t2])},
        2: {"task1": summarize_task_results(spec1, [r_d2_t1]), "task2": summarize_task_results(spec2, [r_d2_t2])},
    }
    depth_rounds = {
        1: [r_d1_t1, r_d1_t2],
        2: [r_d2_t1, r_d2_t2],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)
        plot_path = generate_depth_comparison_plots(
            depth_summaries=depth_summaries,
            depth_rounds=depth_rounds,
            agent_name="qwm_agent_veryclean",
            output_dir=out_dir,
            timestamp="2026-09-22_21-00-00",
        )
        assert plot_path is not None
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 10000  # Non-trivial PNG


def test_export_markdown_task_report_with_depth_sweep():
    """Verify Section 4 Planning Horizon Analysis is formatted in markdown report."""
    spec = TASKS_CATALOG["1"]
    r_d1 = RoundExecutionResult(agent_name="test_agent", task_id=1, task_key="task1", round_idx=1, seed=1000, score=30.0, won=True, survived=True, coins=30, steps=300, search_depth=1, duration_sec=0.5, ms_per_step=1.67, success=True)
    r_d2 = RoundExecutionResult(agent_name="test_agent", task_id=1, task_key="task1", round_idx=1, seed=1000, score=45.0, won=True, survived=True, coins=45, steps=280, search_depth=2, duration_sec=1.2, ms_per_step=4.29, success=True)

    sum_d1 = summarize_task_results(spec, [r_d1])
    sum_d2 = summarize_task_results(spec, [r_d2])

    depth_data = {
        "test_agent": {
            "depth_summaries": {1: {"task1": sum_d1}, 2: {"task1": sum_d2}},
            "plot_path": "/fake/path/depth_sweep_test_agent.png",
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        md_file = Path(tmp_dir) / "report.md"
        export_markdown_task_report({"test_agent": {"task1": sum_d2}}, md_file, depth_sweep_data=depth_data)
        assert md_file.is_file()
        content = md_file.read_text()
        assert "4. Planning Horizon (Search Depth) Analysis" in content
        assert "Horizon Performance Breakdown: `test_agent`" in content
        assert "**D = 1**" in content
        assert "**D = 2**" in content
        assert "depth_sweep_test_agent.png" in content


