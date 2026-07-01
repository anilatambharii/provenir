from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from provenir.core.config import load_run_config
from provenir.core.manifest import RunManifestStore
from provenir.data.dataset import JsonlDataset
from provenir.eval.harness import Evaluator, MultiMetricEvaluator, RegressionGate
from provenir.eval.judge import StubJudge
from provenir.eval.metrics import (
    BLEUMetric,
    ExactMatchMetric,
    MetricFn,
    ROUGELMetric,
    TokenF1Metric,
)
from provenir.governance.audit import AuditLogger
from provenir.governance.model_cards import ModelCardGenerator
from provenir.train.backends.stub import StubBackend
from provenir.train.sweep import GridSweep, SweepConfig
from provenir.train.trainer import Trainer

_METRIC_MAP: dict[str, MetricFn] = {
    "exact_match": ExactMatchMetric(),
    "token_f1": TokenF1Metric(),
    "bleu": BLEUMetric(),
    "rouge_l": ROUGELMetric(),
}


def _ensure_utf8_stdout() -> None:
    """Force UTF-8 stdout so glyphs like arrows survive a cp1252 Windows console."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass


def main() -> None:  # noqa: C901 – CLI dispatcher, complexity is expected
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(prog="provenir")
    subparsers = parser.add_subparsers(dest="command")

    # --- train ---
    train_p = subparsers.add_parser("train", help="Run a training job from YAML config")
    train_p.add_argument("config", help="Path to YAML run config")
    train_p.add_argument("--dataset", default="tests/fixtures/sample.jsonl")

    # --- eval ---
    eval_p = subparsers.add_parser("eval", help="Evaluate predictions against a dataset")
    eval_p.add_argument("--dataset", default="tests/fixtures/sample.jsonl")
    eval_p.add_argument("--predictions", nargs="+", required=True)
    eval_p.add_argument("--baseline", type=float, default=0.0)
    eval_p.add_argument("--threshold", type=float, default=0.2)
    eval_p.add_argument("--output", default="artifacts/eval/report.json")
    eval_p.add_argument(
        "--metrics",
        nargs="+",
        choices=list(_METRIC_MAP.keys()),
        default=None,
    )

    # --- audit ---
    audit_p = subparsers.add_parser("audit", help="Inspect the audit log")
    audit_p.add_argument("--log-dir", default="artifacts")

    # --- model-card ---
    card_p = subparsers.add_parser("model-card", help="Generate a markdown model card")
    card_p.add_argument("name")
    card_p.add_argument("description")
    card_p.add_argument("--output-dir", default="artifacts/model-cards")

    # --- reproduce ---
    repr_p = subparsers.add_parser(
        "reproduce", help="Reproduce a run from a saved manifest"
    )
    repr_p.add_argument("run_id")
    repr_p.add_argument("--manifest-dir", default="artifacts/manifests")
    repr_p.add_argument("--dataset", default="tests/fixtures/sample.jsonl")

    # --- sweep ---
    sweep_p = subparsers.add_parser(
        "sweep", help="Hyperparameter sweep over seed values"
    )
    sweep_p.add_argument("config", help="Path to base YAML run config")
    sweep_p.add_argument("--dataset", default="tests/fixtures/sample.jsonl")
    sweep_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    sweep_p.add_argument("--output", default="artifacts/sweep/results.json")

    # --- compare ---
    cmp_p = subparsers.add_parser(
        "compare", help="Side-by-side comparison of two run manifests"
    )
    cmp_p.add_argument("run_id_a")
    cmp_p.add_argument("run_id_b")
    cmp_p.add_argument("--manifest-dir", default="artifacts/manifests")

    # --- benchmark ---
    bench_p = subparsers.add_parser(
        "benchmark", help="Run standard benchmarks (MMLU, HellaSwag, …)"
    )
    bench_p.add_argument("model_path", help="Path to a HuggingFace model or adapter")
    bench_p.add_argument(
        "--benchmarks",
        nargs="+",
        default=["mmlu"],
        help="Benchmark names (mmlu hellaswag arc_easy …)",
    )
    bench_p.add_argument("--num-fewshot", type=int, default=0)
    bench_p.add_argument("--output", default="artifacts/benchmarks/results.json")

    # --- merge ---
    merge_p = subparsers.add_parser(
        "merge", help="Merge two or more LoRA adapters (SLERP / TIES / DARE)"
    )
    merge_p.add_argument("adapters", nargs="+", help="Adapter directories to merge")
    merge_p.add_argument(
        "--strategy",
        choices=["slerp", "ties", "dare"],
        default="slerp",
    )
    merge_p.add_argument("--output", default="artifacts/merged_adapter")

    # --- hub push ---
    hub_p = subparsers.add_parser("hub", help="HuggingFace Hub operations")
    hub_sub = hub_p.add_subparsers(dest="hub_command")

    hub_push_p = hub_sub.add_parser("push", help="Push adapter to Hub")
    hub_push_p.add_argument("adapter_path")
    hub_push_p.add_argument("repo_id", help="username/model-name")
    hub_push_p.add_argument("--private", action="store_true")
    hub_push_p.add_argument("--token", default=None)

    hub_pull_p = hub_sub.add_parser("pull", help="Pull model from Hub")
    hub_pull_p.add_argument("repo_id")
    hub_pull_p.add_argument("--output", default="artifacts/hub_models")
    hub_pull_p.add_argument("--token", default=None)

    # --- serve ---
    serve_p = subparsers.add_parser("serve", help="Start the Provenir REST API server")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--manifest-dir", default="artifacts/manifests")

    # --- rlaif ---
    rlaif_p = subparsers.add_parser("rlaif", help="Run the RLAIF pipeline")
    rlaif_p.add_argument("config", help="Path to YAML run config")
    rlaif_p.add_argument("--dataset", default="tests/fixtures/sample.jsonl")
    rlaif_p.add_argument("--iterations", type=int, default=3)
    rlaif_p.add_argument("--output", default="artifacts/rlaif/results.json")

    # --- rl (verifiable-reward RL with flight recorder) ---
    rl_p = subparsers.add_parser(
        "rl", help="Run verifiable-reward RL (GRPO/DAPO/GSPO) with flight recorder"
    )
    rl_p.add_argument("--dataset", default="tests/fixtures/sample.jsonl")
    rl_p.add_argument(
        "--algorithm", choices=["grpo", "dapo", "gspo"], default="grpo"
    )
    rl_p.add_argument(
        "--verifier",
        choices=["exact_answer", "math", "contains"],
        default="exact_answer",
    )
    rl_p.add_argument("--max-steps", type=int, default=2)
    rl_p.add_argument("--group-size", type=int, default=8)
    rl_p.add_argument("--output", default="artifacts/rl/result.json")

    # --- contamination (train/eval overlap firewall) ---
    contam_p = subparsers.add_parser(
        "contamination", help="Detect train/eval contamination"
    )
    contam_p.add_argument("train", help="Path to training JSONL")
    contam_p.add_argument("eval", help="Path to evaluation JSONL")
    contam_p.add_argument(
        "--method", choices=["ngram", "exact", "embedding"], default="ngram"
    )
    contam_p.add_argument("--text-key", default="prompt")
    contam_p.add_argument("--output", default="artifacts/contamination/report.json")

    # --- diagnose (Loop Doctor) ---
    diagnose_p = subparsers.add_parser(
        "diagnose", help="Diagnose why a training loop stalled (Loop Doctor)"
    )
    diagnose_p.add_argument(
        "reward_history",
        nargs="+",
        type=float,
        help="Per-iteration mean reward/score values",
    )
    diagnose_p.add_argument(
        "--anomaly", nargs="*", default=[], help="Flight-recorder anomaly kinds observed"
    )
    diagnose_p.add_argument("--hacking-rate", type=float, default=0.0)
    diagnose_p.add_argument("--contamination-rate", type=float, default=0.0)
    diagnose_p.add_argument(
        "--data-age-days", type=float, default=None, help="Age of the training data in days"
    )
    diagnose_p.add_argument("--output", default="artifacts/diagnosis/diagnosis.json")

    # --- passport (Model Passport / BOM) ---
    passport_p = subparsers.add_parser(
        "passport", help="Inspect or verify a signed Model Passport"
    )
    passport_sub = passport_p.add_subparsers(dest="passport_command")

    passport_show_p = passport_sub.add_parser("show", help="Print a passport as markdown")
    passport_show_p.add_argument("path", help="Path to a passport JSON file")

    passport_verify_p = passport_sub.add_parser(
        "verify", help="Verify a passport's HMAC attestation"
    )
    passport_verify_p.add_argument("path", help="Path to a passport JSON file")
    passport_verify_p.add_argument(
        "--key", required=True, help="Signing key (utf-8 string)"
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------

    if args.command == "train":
        config = load_run_config(args.config)
        dataset = JsonlDataset.from_path(args.dataset)
        trainer = Trainer(backend=StubBackend(), config=config)
        manifest = trainer.run(dataset=dataset)
        print(f"Run {manifest.run_id} completed")
        print(f"Config hash: {manifest.config_hash}")
        print(f"Dataset hash: {manifest.dataset_hash}")
        return

    if args.command == "eval":
        dataset = JsonlDataset.from_path(args.dataset)
        if args.metrics:
            selected: list[MetricFn] = [_METRIC_MAP[m] for m in args.metrics]
            evaluator: Evaluator | MultiMetricEvaluator = MultiMetricEvaluator(
                metrics=selected
            )
        else:
            evaluator = MultiMetricEvaluator()
        result = evaluator.evaluate(dataset=dataset, predictions=args.predictions)
        gate = RegressionGate(baseline=args.baseline, threshold=args.threshold)
        result.save(args.output)
        if result.metrics:
            for name, summary in result.metrics.items():
                print(f"{name}: {summary.mean:.3f}  CI={summary.confidence_interval}")
        if result.accuracy:
            print(f"Regression gate passed: {gate.check(result)}")
        return

    if args.command == "audit":
        logger = AuditLogger(log_dir=args.log_dir)
        audit_path = logger.log_dir / "audit.jsonl"
        if audit_path.exists():
            print(audit_path.read_text(encoding="utf-8"))
        else:
            print("No audit log found")
        return

    if args.command == "model-card":
        generator = ModelCardGenerator(output_dir=args.output_dir)
        card_path = generator.generate(args.name, args.description)
        print(card_path)
        return

    if args.command == "reproduce":
        store = RunManifestStore(root_dir=args.manifest_dir)
        manifest = store.load(args.run_id)
        print(f"Reproduced run {manifest.run_id}")
        print(f"Config hash: {manifest.config_hash}")
        print(f"Dataset hash: {manifest.dataset_hash}")
        return

    if args.command == "sweep":
        base_config = load_run_config(args.config)
        dataset = JsonlDataset.from_path(args.dataset)
        sweep_cfg = SweepConfig(param_grid={"seed": args.seeds}, strategy="grid")
        sweep = GridSweep(
            base_config=base_config, sweep_config=sweep_cfg, backend=StubBackend()
        )
        sweep_result = sweep.run(dataset)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trial_records = [
            {
                "trial_id": t.trial_id,
                "params": t.params,
                "run_id": t.manifest.run_id,
                "config_hash": t.manifest.config_hash,
                "dataset_hash": t.manifest.dataset_hash,
            }
            for t in sweep_result.trials
        ]
        output_path.write_text(json.dumps(trial_records, indent=2), encoding="utf-8")
        n_trials = len(sweep_result.trials)
        print(f"Sweep complete: {n_trials} trials → {args.output}")
        for trial in sweep_result.trials:
            seed = trial.params.get("seed")
            run_id = trial.manifest.run_id
            print(f"  trial {trial.trial_id}: seed={seed}  run={run_id}")
        return

    if args.command == "compare":
        store = RunManifestStore(root_dir=args.manifest_dir)
        try:
            m_a = store.load(args.run_id_a)
        except (FileNotFoundError, KeyError):
            print(f"Manifest {args.run_id_a!r} not found")
            return
        try:
            m_b = store.load(args.run_id_b)
        except (FileNotFoundError, KeyError):
            print(f"Manifest {args.run_id_b!r} not found")
            return
        print(f"{'Field':<22} {'Run A':<36} {'Run B':<36}")
        print("-" * 94)
        for field_name in ("run_id", "config_hash", "dataset_hash", "seed", "git_sha"):
            va = getattr(m_a, field_name)
            vb = getattr(m_b, field_name)
            diff = "" if va == vb else " ← differs"
            print(f"{field_name:<22} {str(va):<36} {str(vb):<36}{diff}")
        return

    if args.command == "benchmark":
        from provenir.eval.benchmarks import BenchmarkConfig, BenchmarkEvaluator

        evaluator_b = BenchmarkEvaluator()
        configs = [
            BenchmarkConfig(benchmark=b, num_fewshot=args.num_fewshot)
            for b in args.benchmarks
        ]
        results = evaluator_b.run_suite(args.model_path, configs)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in results]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for r in results:
            stub_note = " [stub — lm-eval not installed]" if r.metadata.get("stub") else ""
            print(f"{r.benchmark}: {r.score:.4f}{stub_note}")
        return

    if args.command == "merge":
        from provenir.adapters.merging import MergeConfig, ModelMerger

        merger = ModelMerger()
        adapter_paths = [Path(a) for a in args.adapters]
        merge_cfg = MergeConfig(strategy=args.strategy)
        result_m = merger.merge(adapter_paths, merge_cfg, Path(args.output))
        stub_note = " [stub]" if result_m.metadata.get("stub") else ""
        print(f"Merged {len(adapter_paths)} adapters → {result_m.output_path}{stub_note}")
        return

    if args.command == "hub":
        from provenir.adapters.hub import HubClient, HubConfig

        client = HubClient(token=getattr(args, "token", None))
        if args.hub_command == "push":
            hub_cfg = HubConfig(
                repo_id=args.repo_id,
                private=getattr(args, "private", False),
                token=getattr(args, "token", None),
            )
            push_result = client.push_adapter(Path(args.adapter_path), hub_cfg)
            stub_note = (
                " [stub — huggingface_hub not installed]" if not push_result.commit_sha else ""
            )
            print(f"Pushed → {push_result.url}{stub_note}")
        elif args.hub_command == "pull":
            local = client.pull_model(
                args.repo_id,
                cache_dir=Path(args.output),
                token=getattr(args, "token", None),
            )
            print(f"Downloaded → {local}")
        else:
            hub_p.print_help()
        return

    if args.command == "serve":
        from provenir.server.app import run_server

        run_server(
            host=args.host,
            port=args.port,
            manifest_dir=args.manifest_dir,
        )
        return

    if args.command == "rlaif":
        config = load_run_config(args.config)
        dataset = JsonlDataset.from_path(args.dataset)
        from provenir.train.rlaif import RLAIFConfig, RLAIFPipeline

        rlaif_cfg = RLAIFConfig(n_iterations=args.iterations)
        pipeline = RLAIFPipeline(
            judge=StubJudge(),
            backend=StubBackend(),
            base_config=config,
            rlaif_config=rlaif_cfg,
        )
        iterations = pipeline.run(train_dataset=dataset)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "iteration": it.iteration,
                "preference_count": it.preference_count,
                "run_id": it.manifest.run_id,
            }
            for it in iterations
        ]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"RLAIF complete: {len(iterations)} iterations → {args.output}")
        return

    if args.command == "rl":
        from provenir.environments import (
            ContainsVerifier,
            ExactAnswerVerifier,
            MathVerifier,
        )
        from provenir.environments.base import Verifier
        from provenir.observability import FlightRecorder, RewardHackingDetector
        from provenir.train.algorithms import GRPOConfig
        from provenir.train.rl import DAPOConfig, GSPOConfig, RLOrchestrator

        dataset = JsonlDataset.from_path(args.dataset)
        verifier_map: dict[str, Verifier] = {
            "exact_answer": ExactAnswerVerifier(),
            "math": MathVerifier(),
            "contains": ContainsVerifier(required=["the"]),
        }
        verifier = verifier_map[args.verifier]
        algorithm: GRPOConfig | DAPOConfig | GSPOConfig
        if args.algorithm == "dapo":
            algorithm = DAPOConfig(group_size=args.group_size, max_steps=args.max_steps)
        elif args.algorithm == "gspo":
            algorithm = GSPOConfig(group_size=args.group_size, max_steps=args.max_steps)
        else:
            algorithm = GRPOConfig(group_size=args.group_size, max_steps=args.max_steps)
        orchestrator = RLOrchestrator(
            algorithm=algorithm,
            verifier=verifier,
            flight_recorder=FlightRecorder(),
            hacking_detector=RewardHackingDetector(),
        )
        rl_result = orchestrator.run(train_dataset=dataset)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rl_result.to_dict(), indent=2), encoding="utf-8")
        print(f"RL ({args.algorithm}) complete: {rl_result.steps_completed} steps")
        print(f"  mean reward:   {rl_result.mean_reward:.3f}")
        print(f"  anomalies:     {rl_result.anomaly_count}")
        print(f"  hacking rate:  {rl_result.hacking_rate:.3f}")
        print(f"  halted:        {rl_result.halted}  →  {args.output}")
        return

    if args.command == "contamination":
        from provenir.eval.contamination import ContaminationChecker, ContaminationConfig

        train_ds = JsonlDataset.from_path(args.train)
        eval_ds = JsonlDataset.from_path(getattr(args, "eval"))
        checker = ContaminationChecker(
            ContaminationConfig(method=args.method, text_key=args.text_key)
        )
        report = checker.check_datasets(train_ds, eval_ds)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        verdict = "CLEAN" if report.is_clean else "CONTAMINATED"
        print(f"Contamination check ({args.method}): {verdict}")
        print(f"  hits:               {len(report.hits)}")
        print(f"  contamination rate: {report.contamination_rate:.3f}  →  {args.output}")
        return

    if args.command == "diagnose":
        from provenir.loop import LoopController, LoopDoctor, LoopSignals

        signals = LoopSignals(
            reward_history=list(args.reward_history),
            anomaly_kinds=list(args.anomaly),
            hacking_rate=args.hacking_rate,
            contamination_rate=args.contamination_rate,
            data_age_days=args.data_age_days,
        )
        diagnosis = LoopDoctor().diagnose(signals)
        action = LoopController().decide(diagnosis)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diag_payload = {"diagnosis": diagnosis.to_dict(), "action": action.to_dict()}
        output_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")
        print(diagnosis.to_markdown())
        print(f"Recommended action: {action.action}  →  {args.output}")
        return

    if args.command == "passport":
        from provenir.governance.passport import ModelPassport

        if args.passport_command in (None, ""):
            passport_p.print_help()
            return
        data = json.loads(Path(args.path).read_text(encoding="utf-8"))
        passport = ModelPassport.from_dict(data)
        if args.passport_command == "show":
            print(passport.to_markdown())
        elif args.passport_command == "verify":
            valid = passport.verify(args.key.encode("utf-8"))
            print(f"Attestation valid: {valid}")
        return

    parser.print_help()
