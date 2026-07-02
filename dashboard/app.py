"""Provenir Interactive Dashboard — public demo app.

Demonstrates every Provenir feature with sample data so users can understand
what each component does and see the impact interactively.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow running from the repo root without pip-installing provenir.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

st.set_page_config(page_title="Provenir Dashboard", page_icon="🔬", layout="wide")

import provenir

# ---------------------------------------------------------------------------
# RH-Bench availability guard
# ---------------------------------------------------------------------------
try:
    from provenir.bench.rhbench import (
        BenchmarkHarness,
        LengthHeuristicDetector,
        ProvenirDetector,
        ProxyDivergenceDetector,
        RandomDetector,
        SyntheticCorpusGenerator,
    )

    _HAS_RHBENCH = True
except Exception:
    _HAS_RHBENCH = False

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Overview",
    "Run Tracking",
    "Contamination Firewall",
    "Reward Hacking Detector",
    "RL Flight Recorder",
    "Loop Doctor",
    "Model Passport",
    "Lineage DAG",
    "Agentic Environments",
    "RH-Bench",
]

st.sidebar.title("🔬 Provenir")
st.sidebar.caption(f"v{provenir.__version__}")
page = st.sidebar.radio("Navigate", PAGES)

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == "Overview":
    st.header("🔬 Provenir — Trust Layer for Model Post-Training")
    st.caption(
        "An open-source Python library. One import. Any training backend. "
        f"Current version: **{provenir.__version__}**"
    )

    # ── What is it? ──────────────────────────────────────────────────────────
    st.markdown(
        """
**Provenir** is a pure-Python library you drop into any fine-tuning or RL training loop.
It sits between your training engine (TRL, verl, Unsloth, custom) and your evaluation pipeline
and provides the trust signals that no training engine ships by default:
reproducibility, contamination-safe evaluation, reward-hacking detection, and signed provenance.
"""
    )

    st.divider()

    # ── Who should use it ────────────────────────────────────────────────────
    st.subheader("Who Should Use Provenir")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
**🏢 AI/ML Teams at Companies**

Teams fine-tuning foundation models (Qwen, Llama, Mistral, Gemma) for product use cases —
customer support, coding, summarisation, agents — where a model that looks great in eval
but fails in production is a real business risk.
"""
        )
    with c2:
        st.markdown(
            """
**🔬 Researchers Running RL/RLHF**

Anyone running GRPO, DAPO, DPO, PPO, or RLVR loops where reward models can be gamed,
training can silently stall, and eval contamination inflates benchmark numbers.
"""
        )
    with c3:
        st.markdown(
            """
**🏛️ Regulated / Compliance-Sensitive Orgs**

Companies subject to EU AI Act, SOC 2, or internal model governance requirements that
need a cryptographically signed audit trail — who trained what model, on what data,
verified by which evals.
"""
        )

    st.divider()

    # ── Without Provenir ─────────────────────────────────────────────────────
    st.subheader("What You Miss Without Provenir")

    st.markdown(
        """
| Risk | Without Provenir | With Provenir |
|---|---|---|
| **Reward hacking** | Model games the reward signal; looks great, performs badly. Goes undetected until production. | RewardHackingDetector flags 7 hack types (tampering, proxy divergence, length inflation…) per trajectory in real time. |
| **Eval contamination** | Train-set prompts leak into eval; benchmark scores inflate. No one notices. | ContaminationChecker blocks the run or surfaces the overlap rate before scores are published. |
| **Silent training stalls** | Loss plateaus; the team restarts the run hoping something changes. Days wasted. | Loop Doctor diagnoses the root cause (bad eval / reward hacking / optimizer instability / data shortage) and tells you exactly what to fix. |
| **RL black box** | KL blows up at step 800. No record of what happened before. | Flight Recorder captures every step; anomalies (entropy_collapse, kl_blowup, advantage_collapse) are flagged the moment they occur. |
| **Irreproducible runs** | "I can't reproduce the model from six weeks ago." Config drift, dataset changes, no audit trail. | Content-addressed RunManifest + Lineage DAG + signed Model Passport make every run fully reproducible and attributable. |
| **Agentic reward sparsity** | Terminal reward only on the last token. Multi-turn credit assignment is ad hoc. | EpisodeRunner + assign_credit distributes rewards across turns (last_turn / uniform / discounted) and records every tool call. |
| **Compliance gaps** | "What data trained this model? Was PII scanned? Was eval contaminated?" No answers. | ModelPassport (HMAC-SHA256) records data provenance, PII scan status, license, and eval integrity — verifiable by anyone with the key. |
"""
    )

    st.divider()

    # ── Value proposition ────────────────────────────────────────────────────
    st.subheader("The Value")

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.metric("Trust signals", "8", help="FlightRecorder anomaly types")
        st.metric("Hack categories detected", "7")
    with v2:
        st.metric("Framework dependencies", "0", help="stdlib + pydantic only for core")
        st.metric("Training backends supported", "Any", help="verl, TRL, Unsloth, custom")
    with v3:
        st.metric("Lines to instrument a run", "3", help="with provenir.track() as run:")
        st.metric("Provenir version", provenir.__version__)
    with v4:
        st.metric("License", "Apache-2.0", help="Permissive, production-safe")
        st.metric("Python", "≥ 3.11")

    st.info(
        "**Yes — Provenir is a Python library.** "
        "Install with `pip install provenir`. "
        "No GPU, no training framework, no cloud account required to use the trust layer. "
        "Optional extras (`pip install 'provenir[train]'`) unlock TRL/PEFT/verl adapters."
    )

    st.divider()

    # ── Feature matrix ───────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Matrix vs Training Engines")
        st.markdown(
            """
| Feature | TRL / verl / Unsloth | **Provenir** |
|---|---|---|
| Reproducible run manifests | ❌ | ✅ |
| Contamination firewall | ❌ | ✅ |
| Reward-hacking detection (7 types) | ❌ | ✅ |
| RL flight recorder (8 anomaly types) | ❌ | ✅ |
| Loop differential diagnosis | ❌ | ✅ |
| Signed Model Passport (HMAC-SHA256) | ❌ | ✅ |
| Lineage DAG (dataset→run→eval) | ❌ | ✅ |
| Multi-turn agentic environments | ❌ | ✅ |
| Multi-turn credit assignment | ❌ | ✅ |
| EU AI Act audit trail (Art. 12) | ❌ | ✅ |
"""
        )

    with col2:
        st.subheader("Architecture")
        st.code(
            """\
┌─────────────────────────────────────────────┐
│              YOUR TRAINING LOOP             │
│                                             │
│  with provenir.track("run", ...) as run:    │
│      run.log_step(metrics)  ──► FlightRec  │
│      run.log_trajectory(t)  ──► HackDetect │
│      run.record_eval(bench) ──► BOM        │
│                                             │
│  On exit ──────────────────────────────┐    │
│    RunManifest (content-addressed)      │    │
│    LineageDAG  (dataset→run→eval)       │    │
│    HackingReport (batch signals)        │    │
│    ModelPassport (HMAC-signed BOM)      │    │
└─────────────────────────────────────────────┘
   ↑ zero framework dependencies; any backend
""",
            language="text",
        )

    st.divider()

    # ── Quick start ──────────────────────────────────────────────────────────
    st.subheader("Quick Start")
    st.code(
        """\
pip install provenir
""",
        language="bash",
    )
    st.code(
        """\
import provenir
from provenir.data.dataset import JsonlDataset

ds = JsonlDataset.from_records([{"prompt": "...", "answer": "..."}])

with provenir.track("my-grpo-run", base_model="Qwen2.5-7B",
                    output_dir="/tmp/prov", dataset=ds) as run:
    for step in range(100):
        run.log_step({"kl": 0.05, "entropy": 1.8, "reward_mean": 0.4})
    run.record_eval("gsm8k", score=0.71)

print(run.manifest.run_id)          # content-addressed run ID
print(run.hacking_report.is_clean)  # True / False
print(run.flight_recorder.summary())
""",
        language="python",
    )


# ===========================================================================
# PAGE 2 — RUN TRACKING
# ===========================================================================
elif page == "Run Tracking":
    st.header("🔍 Run Tracking")

    with st.expander("What this does"):
        st.write(
            "provenir.track() opens a context manager that instruments any training run. "
            "It records per-step metrics via the flight recorder, buffers trajectories for "
            "reward-hacking detection, assembles a content-addressed RunManifest, and writes "
            "every artifact to disk on exit."
        )

    st.subheader("Code pattern")
    st.code(
        """\
import provenir
from provenir.observability.flight_recorder import RLStepMetrics

with provenir.track("demo-run", base_model="Qwen2.5-7B",
                    output_dir=output_dir) as run:
    for step in range(5):
        run.log_step(RLStepMetrics(step=step, kl=0.05, entropy=1.8,
                                   reward_mean=0.1 * step, reward_std=0.05))
    # Inject one suspicious trajectory
    run.log_trajectory({
        "prediction": "def solve(): sys.exit(0)  # skip tests",
        "proxy_reward": 0.95, "true_reward": 0.1
    })
    run.record_eval("gsm8k", score=0.71)
""",
        language="python",
    )

    if st.button("Run demo"):
        try:
            from provenir.observability.flight_recorder import RLStepMetrics

            output_dir = tempfile.mkdtemp(prefix="provenir_demo_")

            with provenir.track(
                "demo-run",
                base_model="Qwen2.5-7B",
                output_dir=output_dir,
            ) as run:
                for step in range(5):
                    run.log_step(
                        RLStepMetrics(
                            step=step,
                            kl=0.05,
                            entropy=1.8,
                            reward_mean=0.1 * step,
                            reward_std=0.05,
                            advantage_std=0.3,
                        )
                    )
                run.log_trajectory(
                    {
                        "prediction": "def solve(): sys.exit(0)  # skip tests",
                        "proxy_reward": 0.95,
                        "true_reward": 0.1,
                    }
                )
                run.record_eval("gsm8k", score=0.71)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Run Manifest")
                if run.manifest is not None:
                    manifest_dict = {
                        "run_id": run.manifest.run_id,
                        "config_hash": run.manifest.config_hash,
                        "dataset_hash": run.manifest.dataset_hash,
                        "seed": run.manifest.seed,
                        "provenance": run.manifest.provenance,
                    }
                    st.json(manifest_dict)

                st.subheader("Hacking Report")
                if run.hacking_report is not None:
                    st.json(run.hacking_report.to_dict())

            with col2:
                st.subheader("Flight Recorder Summary")
                summary = run.flight_recorder.summary()
                st.json(summary)

                st.subheader("Anomalies Detected")
                anomalies = run.anomalies
                if anomalies:
                    for a in anomalies:
                        st.warning(f"Step {a.step}: [{a.severity.upper()}] {a.kind} — {a.detail}")
                else:
                    st.success("No anomalies detected.")

                st.metric("Output dir", output_dir)

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 3 — CONTAMINATION FIREWALL
# ===========================================================================
elif page == "Contamination Firewall":
    st.header("🚫 Contamination Firewall")

    with st.expander("What this does"):
        st.write(
            "ContaminationChecker compares a training set against an eval set and flags "
            "overlapping examples. Contaminated eval data inflates benchmark scores, making "
            "models look better than they are. Supports exact, n-gram, and embedding methods."
        )

    st.code(
        """\
from provenir.data.dataset import JsonlDataset
from provenir.eval.contamination import ContaminationChecker, ContaminationConfig

train_ds = JsonlDataset.from_records([{"prompt": p} for p in train_prompts])
eval_ds  = JsonlDataset.from_records([{"prompt": p} for p in eval_prompts])

checker = ContaminationChecker(ContaminationConfig(method="exact"))
report  = checker.check_datasets(train_ds, eval_ds)
print(report.contamination_rate, report.is_clean)
""",
        language="python",
    )

    @st.cache_data
    def run_contamination():
        from provenir.data.dataset import JsonlDataset
        from provenir.eval.contamination import ContaminationChecker, ContaminationConfig

        train_prompts = [
            "What is the capital of France?",
            "Explain gradient descent in one sentence.",
            "What is photosynthesis?",
            "Who wrote Hamlet?",
            "What is the Pythagorean theorem?",
            "What causes rainbows?",
            "What is the speed of light?",
            "Explain the water cycle.",
            "What is Newton's first law?",
            "What is DNA?",
        ]
        eval_prompts = [
            "What is the capital of France?",  # contaminated!
            "What is the boiling point of water?",
            "Who wrote Hamlet?",  # contaminated!
            "What is machine learning?",
            "Explain entropy.",
            "What is the speed of sound?",
        ]

        train_ds = JsonlDataset.from_records([{"prompt": p} for p in train_prompts])
        eval_ds = JsonlDataset.from_records([{"prompt": p} for p in eval_prompts])

        checker = ContaminationChecker(ContaminationConfig(method="exact"))
        report = checker.check_datasets(train_ds, eval_ds)
        return report, train_prompts, eval_prompts

    if st.button("Run contamination check"):
        try:
            report, train_prompts, eval_prompts = run_contamination()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Contamination Rate", f"{report.contamination_rate:.1%}")
                st.metric("Is Clean", str(report.is_clean))
                st.metric("Contaminated Train Indices", str(sorted(report.contaminated_train_indices)))

            with col2:
                st.subheader("Hits")
                if report.hits:
                    import pandas as pd
                    rows = []
                    for hit in report.hits:
                        rows.append({
                            "train_idx": hit.train_index,
                            "eval_idx": hit.eval_index,
                            "score": hit.score,
                            "method": hit.method,
                            "train_prompt": train_prompts[hit.train_index][:60],
                            "eval_prompt": eval_prompts[hit.eval_index][:60],
                        })
                    st.dataframe(pd.DataFrame(rows))
                else:
                    st.success("No contamination hits.")

            st.divider()
            st.subheader("Full Report")
            st.json(report.to_dict())

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 4 — REWARD HACKING DETECTOR
# ===========================================================================
elif page == "Reward Hacking Detector":
    st.header("🎭 Reward Hacking Detector")

    with st.expander("What this does"):
        st.write(
            "RewardHackingDetector scans model trajectories for 7 reward-hacking signals: "
            "length inflation, format exploits, test tampering, verifier gaming, proxy/true "
            "divergence, degenerate repetition, and advantage collapse in GRPO groups."
        )

    st.code(
        """\
from provenir.observability.reward_hacking import RewardHackingDetector, RewardHackingConfig

det = RewardHackingDetector(RewardHackingConfig(max_length=2000, proxy_true_gap=0.3))

# Single trajectory
signals = det.detect({"prediction": "sys.exit(0)", "proxy_reward": 0.9, "true_reward": 0.1})

# Batch
report = det.detect_batch(trajectories)
print(report.hacking_rate, report.by_kind())
""",
        language="python",
    )

    @st.cache_data
    def get_sample_trajectories():
        return [
            # 0: test tampering
            {
                "prediction": "def solve(): sys.exit(0)  # skip all tests",
                "proxy_reward": 0.95,
                "true_reward": 0.05,
                "label": "test_tampering",
            },
            # 1: long repetitive
            {
                "prediction": "The answer is correct. " * 150,
                "proxy_reward": 0.80,
                "true_reward": 0.40,
                "label": "degenerate_repetition",
            },
            # 2: proxy >> true (reward model gaming)
            {
                "prediction": "Here is a thorough analysis of the problem.",
                "proxy_reward": 0.95,
                "true_reward": 0.20,
                "label": "proxy_divergence",
            },
            # 3: verifier gaming
            {
                "prediction": "All checks pass. Verification complete.",
                "proxy_reward": 0.90,
                "true_reward": 0.30,
                "verification": {"metadata": {"suspected_hacking": True}},
                "label": "verifier_gaming",
            },
            # 4: format exploit (short all-caps)
            {
                "prediction": "DONE",
                "proxy_reward": 0.88,
                "true_reward": 0.50,
                "label": "format_exploit",
            },
            # 5: length inflation (>2000 chars)
            {
                "prediction": "A" * 2500,
                "proxy_reward": 0.75,
                "true_reward": 0.60,
                "label": "length_inflation",
            },
            # 6: clean trajectory
            {
                "prediction": "The capital of France is Paris.",
                "proxy_reward": 0.85,
                "true_reward": 0.82,
                "label": "clean",
            },
            # 7: clean
            {
                "prediction": "Gradient descent minimizes the loss by moving in the direction of the negative gradient.",
                "proxy_reward": 0.78,
                "true_reward": 0.76,
                "label": "clean",
            },
            # 8: pytest.skip tampering
            {
                "prediction": "def test_solution():\n    pytest.skip('not implemented')\n",
                "proxy_reward": 0.92,
                "true_reward": 0.02,
                "label": "test_tampering",
            },
            # 9: moderate proxy gap (just below threshold — clean)
            {
                "prediction": "Newton's first law: an object in motion stays in motion.",
                "proxy_reward": 0.75,
                "true_reward": 0.65,
                "label": "clean",
            },
        ]

    trajectories = get_sample_trajectories()

    if st.button("Run hacking detection"):
        try:
            from provenir.observability.reward_hacking import RewardHackingDetector, RewardHackingConfig
            import pandas as pd

            det = RewardHackingDetector(RewardHackingConfig(max_length=2000, proxy_true_gap=0.3))

            rows = []
            for i, traj in enumerate(trajectories):
                signals = det.detect(traj)
                rows.append({
                    "idx": i,
                    "label": traj["label"],
                    "prediction_preview": str(traj["prediction"])[:50] + "...",
                    "proxy_reward": traj["proxy_reward"],
                    "true_reward": traj["true_reward"],
                    "signals_found": ", ".join(s.kind for s in signals) or "none",
                    "severities": ", ".join(s.severity for s in signals) or "—",
                    "flagged": len(signals) > 0,
                })

            df = pd.DataFrame(rows)

            st.subheader("Per-trajectory results")
            st.dataframe(
                df.style.apply(
                    lambda row: ["background-color: #ffdddd" if row["flagged"] else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )

            st.divider()

            report = det.detect_batch(trajectories)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hacking Rate", f"{report.hacking_rate:.1%}")
            with col2:
                st.metric("Is Clean", str(report.is_clean))
            with col3:
                st.metric("Total Signals", len(report.signals))

            st.subheader("Signals by kind")
            st.json(report.by_kind())

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 5 — RL FLIGHT RECORDER
# ===========================================================================
elif page == "RL Flight Recorder":
    st.header("✈️ RL Flight Recorder")

    with st.expander("What this does"):
        st.write(
            "FlightRecorder is the 'black box' for RL training. Feed it one RLStepMetrics "
            "per training step and it detects 8 anomaly types: KL blowup/collapse, entropy "
            "collapse, length explosion, advantage collapse, reward std collapse, reward spike, "
            "and gradient explosion."
        )

    st.code(
        """\
from provenir.observability.flight_recorder import FlightRecorder, RLStepMetrics

rec = FlightRecorder()
for step in range(15):
    metrics = RLStepMetrics(step=step, kl=kl[step], entropy=entropy[step],
                             reward_mean=reward[step], advantage_std=adv_std[step])
    new_anomalies = rec.log_step(metrics)

print(rec.health_report())
print(rec.summary())
""",
        language="python",
    )

    if st.button("Run flight recorder demo"):
        try:
            from provenir.observability.flight_recorder import FlightRecorder, RLStepMetrics
            import pandas as pd

            rec = FlightRecorder()

            # Steps 0-7: healthy
            healthy_steps = [
                RLStepMetrics(
                    step=i,
                    kl=0.05 + i * 0.005,
                    entropy=2.0 - i * 0.02,
                    reward_mean=0.2 + i * 0.03,
                    reward_std=0.15,
                    advantage_std=0.5,
                    response_length_mean=200.0,
                )
                for i in range(8)
            ]
            # Step 8: entropy collapse (drops to 1% of max)
            # Step 9: KL blowup
            # Step 10: advantage collapse
            # Step 11: reward spike
            anomalous_steps = [
                RLStepMetrics(step=8, kl=0.08, entropy=0.01, reward_mean=0.42, reward_std=0.15, advantage_std=0.5, response_length_mean=200.0),
                RLStepMetrics(step=9, kl=1.0,  entropy=1.5,  reward_mean=0.43, reward_std=0.15, advantage_std=0.5, response_length_mean=200.0),
                RLStepMetrics(step=10, kl=0.06, entropy=1.4, reward_mean=0.44, reward_std=0.15, advantage_std=0.0, response_length_mean=200.0),
                RLStepMetrics(step=11, kl=0.06, entropy=1.3, reward_mean=5.0,  reward_std=0.15, advantage_std=0.5, response_length_mean=200.0),
            ]
            # Steps 12-14: recover
            recovery_steps = [
                RLStepMetrics(step=12 + i, kl=0.07, entropy=1.6, reward_mean=0.45 + i * 0.01, reward_std=0.15, advantage_std=0.4, response_length_mean=200.0)
                for i in range(3)
            ]

            all_steps = healthy_steps + anomalous_steps + recovery_steps
            step_anomalies: list[list] = []
            for m in all_steps:
                found = rec.log_step(m)
                step_anomalies.append(found)

            rows = []
            for m, anoms in zip(all_steps, step_anomalies):
                rows.append({
                    "step": m.step,
                    "kl": round(m.kl, 4),
                    "entropy": round(m.entropy, 4),
                    "reward_mean": round(m.reward_mean, 4),
                    "advantage_std": round(m.advantage_std, 4),
                    "anomalies": ", ".join(a.kind for a in anoms) or "—",
                    "flagged": len(anoms) > 0,
                })

            df = pd.DataFrame(rows)
            st.subheader("Step-by-step table")
            st.dataframe(
                df.style.apply(
                    lambda row: ["background-color: #ffdddd" if row["flagged"] else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True,
            )

            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", len(rec.anomalies))
            with col2:
                st.metric("Health Status", rec.summary()["verdict"])
            with col3:
                st.metric("Steps Recorded", rec.summary()["num_steps"])

            st.subheader("Health Report")
            st.code(rec.health_report(), language="text")

            st.subheader("Summary")
            st.json(rec.summary())

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 6 — LOOP DOCTOR
# ===========================================================================
elif page == "Loop Doctor":
    st.header("🩺 Loop Doctor")

    with st.expander("What this does"):
        st.write(
            "LoopDoctor performs differential diagnosis on a stalled training loop. "
            "Given trust-layer signals, it attributes the stall to one of four causes: "
            "contaminated eval, reward hacking, algorithm instability, or data shortage. "
            "LoopController then maps the diagnosis to the next training action."
        )

    st.code(
        """\
from provenir.loop.doctor import LoopDoctor, LoopController, LoopSignals

signals = LoopSignals(
    reward_history=[0.4, 0.41, 0.4, 0.41, 0.4],  # plateaued
    contamination_rate=0.0,
    hacking_rate=0.0,
    anomaly_kinds=[],
    slice_failures={"tool_use": 0.85},
    data_age_days=400,
)
diagnosis = LoopDoctor().diagnose(signals)
action    = LoopController().decide(diagnosis)
print(diagnosis.primary_category, action.action)
""",
        language="python",
    )

    scenario = st.radio(
        "Select scenario",
        [
            "a. Contaminated eval (contamination_rate=0.3)",
            "b. Reward hacking (hacking_rate=0.5)",
            "c. Algorithm instability (advantage_collapse)",
            "d. Data problem (plateau + stale + slice failures)",
        ],
    )

    if st.button("Run diagnosis"):
        try:
            from provenir.loop.doctor import LoopDoctor, LoopController, LoopSignals

            reward_plateau = [0.40, 0.41, 0.40, 0.41, 0.40, 0.41]

            if scenario.startswith("a"):
                signals = LoopSignals(
                    reward_history=reward_plateau,
                    contamination_rate=0.3,
                    hacking_rate=0.0,
                    hacking_kinds=[],
                    anomaly_kinds=[],
                    slice_failures={},
                    data_age_days=None,
                )
            elif scenario.startswith("b"):
                signals = LoopSignals(
                    reward_history=reward_plateau,
                    contamination_rate=0.0,
                    hacking_rate=0.5,
                    hacking_kinds=["test_tampering", "proxy_divergence"],
                    anomaly_kinds=[],
                    slice_failures={},
                    data_age_days=None,
                )
            elif scenario.startswith("c"):
                signals = LoopSignals(
                    reward_history=reward_plateau,
                    contamination_rate=0.0,
                    hacking_rate=0.0,
                    hacking_kinds=[],
                    anomaly_kinds=["advantage_collapse", "entropy_collapse"],
                    slice_failures={},
                    data_age_days=None,
                )
            else:
                signals = LoopSignals(
                    reward_history=reward_plateau,
                    contamination_rate=0.0,
                    hacking_rate=0.0,
                    hacking_kinds=[],
                    anomaly_kinds=[],
                    slice_failures={"tool_use": 0.85, "math": 0.72},
                    data_age_days=400,
                )

            diagnosis = LoopDoctor().diagnose(signals)
            action = LoopController().decide(diagnosis)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Primary Category", diagnosis.primary_category)
                st.metric("Is Healthy", str(diagnosis.is_healthy))
            with col2:
                st.metric("Recommended Action", action.action)
                st.caption(action.reason[:120])

            st.divider()
            st.subheader("Diagnosis Card")
            st.markdown(diagnosis.to_markdown())

            if diagnosis.data_request is not None:
                st.divider()
                st.subheader("Data Request")
                st.json(diagnosis.data_request.to_dict())

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 7 — MODEL PASSPORT
# ===========================================================================
elif page == "Model Passport":
    st.header("🛂 Model Passport")

    with st.expander("What this does"):
        st.write(
            "ModelBOM is a Bill of Materials recording what data, code, and evals produced "
            "a model. PassportSigner signs it with HMAC-SHA256. The passport can be verified "
            "by anyone with the key, and risk_flags() surfaces compliance issues."
        )

    st.code(
        """\
from provenir.governance.bom import ModelBOM, DataComponent, CodeComponent, EvalComponent
from provenir.governance.passport import PassportSigner

bom = ModelBOM(model_id="my-model-v1", base_model="Qwen2.5-7B", run_id="run-001",
    data=[DataComponent(name="train", content_hash="abc123", num_records=50000,
                        license="CC-BY", pii_scanned=True, contamination_checked=True),
          DataComponent(name="sft-extra", content_hash="def456", num_records=5000,
                        license="unknown", pii_scanned=False)],  # ← risk flag!
    code=CodeComponent(git_sha="deadbeef", dependencies_hash="hash123", framework="trl"),
    evals=[EvalComponent(benchmark="gsm8k", score=0.71)],
    hyperparameters={"lr": 1e-5, "batch_size": 16})

KEY = b"demo-key-32-bytes-pad-0000000000"
passport = PassportSigner(key=KEY, key_id="demo").sign(bom)
print(bom.risk_flags())       # ['unknown_license', 'unscanned_pii', ...]
print(passport.verify(KEY))   # True
""",
        language="python",
    )

    if st.button("Build and sign passport"):
        try:
            from provenir.governance.bom import (
                CodeComponent,
                DataComponent,
                EvalComponent,
                ModelBOM,
            )
            from provenir.governance.passport import PassportSigner

            bom = ModelBOM(
                model_id="my-model-v1",
                base_model="Qwen2.5-7B",
                run_id="run-001",
                data=[
                    DataComponent(
                        name="train",
                        content_hash="abc123def456abc123def456abc123de",
                        num_records=50000,
                        license="CC-BY-4.0",
                        pii_scanned=True,
                        contamination_checked=True,
                    ),
                    DataComponent(
                        name="sft-extra",
                        content_hash="deadbeefdeadbeefdeadbeefdeadbeef",
                        num_records=5000,
                        license="unknown",  # ← risk flag!
                        pii_scanned=False,  # ← risk flag!
                        contamination_checked=False,
                    ),
                ],
                code=CodeComponent(
                    git_sha="deadbeef1234",
                    dependencies_hash="hash123abc",
                    framework="trl",
                ),
                evals=[EvalComponent(benchmark="gsm8k", score=0.71)],
                hyperparameters={"lr": 1e-5, "batch_size": 16, "epochs": 3},
                created_at="2026-06-28T00:00:00Z",
            )

            KEY = b"demo-key-32-bytes-pad-0000000000"
            WRONG_KEY = b"wrong-key-32-bytes-pad-000000000"
            passport = PassportSigner(key=KEY, key_id="demo").sign(
                bom, signed_at="2026-06-28T00:00:00Z"
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Content Hash (first 12 chars)", bom.content_hash()[:12])
                st.metric("Verify with correct key", str(passport.verify(KEY)))
                st.metric("Verify with wrong key", str(passport.verify(WRONG_KEY)))

                st.subheader("Risk Flags")
                flags = bom.risk_flags()
                if flags:
                    for flag in flags:
                        st.error(f"⚠️ {flag}")
                else:
                    st.success("No risk flags.")

            with col2:
                st.subheader("Passport Markdown")
                st.markdown(passport.to_markdown())

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 8 — LINEAGE DAG
# ===========================================================================
elif page == "Lineage DAG":
    st.header("🕸️ Lineage DAG")

    with st.expander("What this does"):
        st.write(
            "LineageGraph is a content-addressed, acyclic provenance graph. Nodes represent "
            "datasets, runs, adapters, and evals. Edges are typed relations. The graph can be "
            "serialized to JSON or DOT format for visualization."
        )

    st.markdown(
        "**Valid node types:** `dataset`, `run`, `adapter`, `eval`, `merge`, `model`  \n"
        "**Valid relations:** `produced`, `derived_from`, `evaluated_by`, `merged_into`, `trained_on`"
    )

    st.code(
        """\
from provenir.provenance.lineage import LineageGraph, LineageNode, LineageEdge

g = LineageGraph()
g.add_node(LineageNode("ds-train", "dataset", "hash-ds", {"rows": 50000}))
g.add_node(LineageNode("run-001",  "run",     "hash-run", {"base": "Qwen2.5-7B"}))
g.add_node(LineageNode("adapter-1","adapter", "hash-ada", {"peft": "lora"}))
g.add_node(LineageNode("eval-gsm8k","eval",   "hash-eval",{"score": 0.71}))

g.add_edge(LineageEdge("ds-train",  "run-001",   "trained_on"))
g.add_edge(LineageEdge("run-001",   "adapter-1", "produced"))
g.add_edge(LineageEdge("adapter-1", "eval-gsm8k","evaluated_by"))

provenance = g.provenance_of("eval-gsm8k")
print(g.to_dot())
""",
        language="python",
    )

    if st.button("Build lineage graph"):
        try:
            from provenir.provenance.lineage import LineageEdge, LineageGraph, LineageNode

            g = LineageGraph()
            g.add_node(LineageNode("ds-train",   "dataset", "hash-ds",   {"rows": 50000}))
            g.add_node(LineageNode("run-001",    "run",     "hash-run",  {"base_model": "Qwen2.5-7B"}))
            g.add_node(LineageNode("adapter-1",  "adapter", "hash-ada",  {"peft": "lora", "rank": 16}))
            g.add_node(LineageNode("eval-gsm8k", "eval",    "hash-eval", {"benchmark": "gsm8k", "score": 0.71}))

            g.add_edge(LineageEdge("ds-train",  "run-001",    "trained_on"))
            g.add_edge(LineageEdge("run-001",   "adapter-1",  "produced"))
            g.add_edge(LineageEdge("adapter-1", "eval-gsm8k", "evaluated_by"))

            provenance = g.provenance_of("eval-gsm8k")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Graph (JSON)")
                st.json(g.to_dict())

            with col2:
                st.subheader("DOT (Graphviz)")
                st.code(g.to_dot(), language="dot")

            st.divider()
            st.subheader("Provenance of eval-gsm8k")
            for node in provenance:
                st.markdown(
                    f"- **{node.node_id}** (`{node.node_type}`) — hash: `{node.content_hash}` — {node.attributes}"
                )

            st.metric("Roots", str(g.roots()))
            st.metric("Leaves", str(g.leaves()))

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 9 — AGENTIC ENVIRONMENTS
# ===========================================================================
elif page == "Agentic Environments":
    st.header("🤖 Agentic Environments")

    with st.expander("What this does"):
        st.write(
            "Provenir ships sandboxed, deterministic multi-turn tool-use environments. "
            "EpisodeRunner drives a policy through an environment and records every turn. "
            "assign_credit() redistributes the terminal reward across turns using last_turn, "
            "uniform, or discounted strategies."
        )

    st.code(
        """\
from provenir.environments import (
    make_lookup_environment, make_calculator_environment,
    EpisodeRunner, StubAgentPolicy, CreditConfig, assign_credit,
)

env    = make_lookup_environment()
policy = StubAgentPolicy([
    '{"tool": "lookup", "args": {"key": "capital_of_france"}}',
    "Paris",
])
result = EpisodeRunner().run(env, policy)
print(result.success, result.total_reward)

credits = assign_credit(result, result.total_reward, CreditConfig("uniform"))
""",
        language="python",
    )

    if st.button("Run agentic episodes"):
        try:
            from provenir.environments import (
                CreditConfig,
                EpisodeRunner,
                StubAgentPolicy,
                assign_credit,
                make_calculator_environment,
                make_lookup_environment,
            )
            import pandas as pd

            runner = EpisodeRunner()

            # Episode 1: Lookup
            lookup_env = make_lookup_environment()
            lookup_policy = StubAgentPolicy([
                '{"tool": "lookup", "args": {"key": "capital_of_france"}}',
                "Paris",
            ])
            lookup_result = runner.run(lookup_env, lookup_policy)

            # Episode 2: Calculator
            calc_env = make_calculator_environment()
            calc_policy = StubAgentPolicy([
                '{"tool": "calc", "args": {"expr": "6 * 7"}}',
                "42",
            ])
            calc_result = runner.run(calc_env, calc_policy)

            for env_name, result in [("Lookup", lookup_result), ("Calculator", calc_result)]:
                st.subheader(f"{env_name} Episode")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Success", str(result.success))
                with col2:
                    st.metric("Total Reward", result.total_reward)
                with col3:
                    st.metric("Tool Calls", result.num_tool_calls)

                turns_df = pd.DataFrame([t.to_dict() for t in result.turns])
                st.dataframe(turns_df, use_container_width=True)

                st.subheader(f"{env_name} Credit Assignment")
                terminal_reward = result.total_reward
                credit_cols = st.columns(3)
                for col, strategy in zip(credit_cols, ["last_turn", "uniform", "discounted"]):
                    credits = assign_credit(result, terminal_reward, CreditConfig(strategy=strategy))
                    with col:
                        st.markdown(f"**{strategy}**")
                        for i, c in enumerate(credits):
                            st.write(f"Turn {i}: {c:.4f}")

                st.divider()

        except Exception as exc:
            st.error(f"Error: {exc}")
            import traceback
            st.code(traceback.format_exc())


# ===========================================================================
# PAGE 10 — RH-BENCH
# ===========================================================================
elif page == "RH-Bench":
    st.header("📊 RH-Bench")

    if not _HAS_RHBENCH:
        st.info(
            "RH-Bench is a private local module — run the benchmark locally with:\n\n"
            "```\npython examples/rhbench_demo.py\n```"
        )
    else:
        with st.expander("What this does"):
            st.write(
                "RH-Bench is a reproducible benchmark for evaluating reward-hacking detectors. "
                "It generates a labeled corpus of hacking and clean trajectories, then evaluates "
                "multiple detector baselines across precision, recall, F1, and AUROC."
            )

        st.code(
            """\
from provenir.bench.rhbench import (
    SyntheticCorpusGenerator, BenchmarkHarness,
    ProvenirDetector, ProxyDivergenceDetector,
    LengthHeuristicDetector, RandomDetector,
)

corpus  = SyntheticCorpusGenerator(seed=0).generate(n_per_category=20, n_clean=40)
harness = BenchmarkHarness()
results = {name: harness.evaluate(det, corpus)
           for name, det in detectors.items()}
""",
            language="python",
        )

        if st.button("Run RH-Bench"):
            try:
                import pandas as pd

                with st.spinner("Generating corpus and running detectors..."):
                    corpus = SyntheticCorpusGenerator(seed=0).generate(
                        n_per_category=20, n_clean=40
                    )
                    harness = BenchmarkHarness()
                    detectors = {
                        "ProvenirDetector": ProvenirDetector(),
                        "ProxyDivergenceDetector": ProxyDivergenceDetector(),
                        "LengthHeuristicDetector": LengthHeuristicDetector(),
                        "RandomDetector": RandomDetector(),
                    }
                    evaluations = {
                        name: harness.evaluate(det, corpus)
                        for name, det in detectors.items()
                    }

                st.subheader("Table 1 — Overall Results")
                table_rows = []
                for name, ev in evaluations.items():
                    table_rows.append({
                        "Detector": name,
                        "Precision": round(ev.precision, 3),
                        "Recall": round(ev.recall, 3),
                        "F1": round(ev.f1, 3),
                        "AUROC": round(ev.auroc, 3),
                        "Accuracy": round(ev.accuracy, 3),
                        "TP": ev.tp,
                        "FP": ev.fp,
                        "TN": ev.tn,
                        "FN": ev.fn,
                    })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                st.divider()
                st.subheader("Corpus Stats")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trajectories", corpus.size)
                with col2:
                    st.metric("Hack Count", corpus.hack_count)
                with col3:
                    st.metric("Clean Count", corpus.clean_count)

                st.subheader("Per-category breakdown (ProvenirDetector)")
                prov_ev = evaluations["ProvenirDetector"]
                cat_rows = [c.to_dict() for c in prov_ev.per_category]
                if cat_rows:
                    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

                st.divider()
                st.subheader("ProvenirDetector Scorecard")
                st.markdown(prov_ev.to_markdown())

            except Exception as exc:
                st.error(f"Error: {exc}")
                import traceback
                st.code(traceback.format_exc())
