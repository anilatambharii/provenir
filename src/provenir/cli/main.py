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

    # --- report ---
    report_p = subparsers.add_parser(
        "report", help="Generate an HTML report from a run output directory"
    )
    report_p.add_argument("run_dir", help="ProvenirRun output directory")
    report_p.add_argument(
        "--output", default=None, help="Output HTML path (default: <run_dir>/report.html)"
    )

    # --- hub push ---
    hub_p = subparsers.add_parser("hub", help="HuggingFace Hub operations")
    hub_sub = hub_p.add_subparsers(dest="hub_command")

    hub_push_p = hub_sub.add_parser("push", help="Push adapter to Hub")
    hub_push_p.add_argument("adapter_path")
    hub_push_p.add_argument("repo_id", help="username/model-name")
    hub_push_p.add_argument("--private", action="store_true")
    hub_push_p.add_argument("--token", default=None)
    hub_push_p.add_argument(
        "--passport", default=None, help="Path to a passport.json to upload alongside the adapter"
    )

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

    # --- reward-validity (spurious-reward ablation harness) ---
    rv_p = subparsers.add_parser(
        "reward-validity",
        help=(
            "Run the spurious-reward ablation harness (Shao et al. 2026). "
            "NOTE: with the stub backend the 'training' is diagnostic-only; "
            "gains are methodology demonstrations, not real model improvement."
        ),
    )
    rv_p.add_argument(
        "--reward",
        choices=["exact_answer", "math", "contains"],
        default="exact_answer",
        help="Verifier-backed reward to test for spurious gain",
    )
    rv_p.add_argument("--probes", required=True, help="Path to a JSONL file of labeled probes")
    rv_p.add_argument(
        "--stage", default=None, help="If set, gate promotion to this stage (e.g. production)"
    )
    rv_p.add_argument(
        "--min-validity",
        type=float,
        default=None,
        help="Minimum validity score required to promote",
    )
    rv_p.add_argument(
        "--json",
        dest="json_output",
        default=None,
        metavar="OUT.JSON",
        help="Optional path to write the JSON report",
    )
    rv_p.add_argument("--seed", type=int, default=0, help="RNG seed for ablation controls")
    rv_p.add_argument(
        "--max-steps",
        type=int,
        default=2,
        help="Steps per ablation training run (default: 2)",
    )

    # --- verify-reliability (verifier reliability harness) ---
    vr_p = subparsers.add_parser(
        "verify-reliability",
        help="Stress-test a verifier for reliability (consistency, sensitivity, gameability, …)",
    )
    vr_p.add_argument(
        "--verifier", choices=["exact_answer", "math", "contains"], default="exact_answer"
    )
    vr_p.add_argument("--probes", required=True, help="Path to a JSONL file of labeled probes")
    vr_p.add_argument(
        "--stage", default=None, help="If set, gate promotion to this stage (e.g. production)"
    )
    vr_p.add_argument(
        "--min-overall", type=float, default=None, help="Minimum overall reliability to promote"
    )
    vr_p.add_argument("--output", default=None, help="Optional path to write the JSON report")

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

    # --- scan (supply-chain model scanner) ---
    scan_p = subparsers.add_parser(
        "scan",
        help="Scan a model artifact directory for supply-chain threats",
    )
    scan_p.add_argument("path", help="Path to model directory or file to scan")
    scan_p.add_argument(
        "--json",
        dest="json_output",
        default=None,
        metavar="OUT.JSON",
        help="Optional path to write the JSON report",
    )

    # --- regulation (EU AI Act Art. 53 / Annex IV evidence generator) ---
    reg_p = subparsers.add_parser(
        "regulation",
        help="Generate EU AI Act Art. 53 / Annex IV compliance evidence from a signed passport",
    )
    reg_sub = reg_p.add_subparsers(dest="reg_command")

    reg_art53_p = reg_sub.add_parser(
        "art53",
        help="Generate an Art. 53 / Annex XII training-data summary",
    )
    reg_art53_p.add_argument("--passport", required=True, help="Path to a passport JSON file")
    reg_art53_p.add_argument(
        "--sme",
        action="store_true",
        help="Apply SME domain threshold (5%% instead of 10%%)",
    )
    reg_art53_p.add_argument("--out", default=None, help="Write Markdown output to this file")
    reg_art53_p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="SCORE",
        help="Exit 1 if coverage score is below SCORE (0.0–1.0). CI gate.",
    )

    reg_annexiv_p = reg_sub.add_parser(
        "annex-iv",
        help="Generate an Annex IV technical-documentation skeleton",
    )
    reg_annexiv_p.add_argument("--passport", required=True, help="Path to a passport JSON file")
    reg_annexiv_p.add_argument("--out", default=None, help="Write Markdown output to this file")
    reg_annexiv_p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="SCORE",
        help="Exit 1 if coverage score is below SCORE (0.0–1.0). CI gate.",
    )

    reg_fda_p = reg_sub.add_parser(
        "fda-pccp", help="Generate FDA PCCP template from passport"
    )
    reg_fda_p.add_argument("--passport", required=True, help="Path to passport JSON")
    reg_fda_p.add_argument("--out", default=None, help="Output markdown path")
    reg_fda_p.add_argument(
        "--fail-under", type=float, default=None, dest="fail_under", metavar="SCORE"
    )

    reg_nist_p = reg_sub.add_parser(
        "nist-rmf", help="Generate NIST AI RMF summary from passport"
    )
    reg_nist_p.add_argument("--passport", required=True)
    reg_nist_p.add_argument("--out", default=None)
    reg_nist_p.add_argument(
        "--fail-under", type=float, default=None, dest="fail_under", metavar="SCORE"
    )

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

    passport_export_p = passport_sub.add_parser(
        "export", help="Export passport as CycloneDX or SPDX 3.0 SBOM"
    )
    passport_export_p.add_argument("path", help="Path to passport JSON file")
    passport_export_p.add_argument(
        "--format",
        choices=["cyclonedx-json", "spdx3-json"],
        default="cyclonedx-json",
        dest="export_format",
    )
    passport_export_p.add_argument(
        "--out", default=None, help="Output file path (default: stdout)"
    )

    passport_dsse_p = passport_sub.add_parser(
        "sign-dsse", help="Sign passport as a DSSE envelope (Sigstore-compatible)"
    )
    passport_dsse_p.add_argument("path", help="Path to passport JSON file")
    passport_dsse_p.add_argument("--key", required=True, help="Signing key (utf-8 string)")
    passport_dsse_p.add_argument("--out", required=True, help="Output DSSE envelope JSON path")

    passport_hub_p = passport_sub.add_parser(
        "verify-hub", help="Verify weight hashes in a local model directory against its passport"
    )
    passport_hub_p.add_argument(
        "--dir", required=True, dest="model_dir", help="Local model directory to scan"
    )
    passport_hub_p.add_argument(
        "--repo", default="", dest="repo_id", help="HuggingFace repo ID (informational)"
    )
    passport_hub_p.add_argument(
        "--passport-store",
        default=None,
        dest="passport_store_dir",
        help="Directory where passport.json files are stored (checked before model dir)",
    )

    # --- rag-corpus ---
    rag_p = subparsers.add_parser(
        "rag-corpus", help="RAG corpus trust: PII, retraction, training overlap"
    )
    rag_sub = rag_p.add_subparsers(dest="rag_command")

    rag_scan_p = rag_sub.add_parser(
        "scan", help="Scan a RAG corpus directory for PII, retracted DOIs, and training overlap"
    )
    rag_scan_p.add_argument("--corpus", required=True, help="Path to corpus directory")
    rag_scan_p.add_argument(
        "--train-hashes",
        default=None,
        dest="train_hashes_file",
        help="Text file with one SHA-256 hash per line (training document hashes)",
    )
    rag_scan_p.add_argument(
        "--known-retracted",
        default=None,
        dest="known_retracted",
        help="Text file with one retracted DOI per line",
    )
    rag_scan_p.add_argument(
        "--stage", default=None, help="If set, gate promotion to this stage (e.g. production)"
    )
    rag_scan_p.add_argument(
        "--allow-pii", action="store_true", dest="allow_pii", help="Do not block on PII"
    )
    rag_scan_p.add_argument(
        "--allow-retraction",
        action="store_true",
        dest="allow_retraction",
        help="Do not block on retracted DOIs",
    )
    rag_scan_p.add_argument(
        "--json",
        default=None,
        dest="json_output",
        metavar="OUT.JSON",
        help="Optional path to write the JSON report",
    )

    # --- gate ---
    gate_p = subparsers.add_parser("gate", help="Promotion gate CI checks")
    gate_sub = gate_p.add_subparsers(dest="gate_command")

    gate_promote_p = gate_sub.add_parser(
        "promote", help="Check if a passport passes all promotion gate requirements"
    )
    gate_promote_p.add_argument("--passport", required=True, help="Path to passport JSON file")
    gate_promote_p.add_argument("--stage", default="production", help="Target deployment stage")
    gate_promote_p.add_argument(
        "--require-scan",
        action="store_true",
        dest="require_scan",
        help="Require a clean supply-chain scan",
    )
    gate_promote_p.add_argument(
        "--require-no-retraction",
        action="store_true",
        dest="require_no_retraction",
        help="Block if any training DOIs are retracted",
    )
    gate_promote_p.add_argument(
        "--min-validity",
        type=float,
        default=None,
        dest="min_validity",
        help="Minimum reward validity score (0.0–1.0)",
    )
    gate_promote_p.add_argument(
        "--require-no-pii",
        action="store_true",
        dest="require_no_pii",
        help="Block if PII was found in training data",
    )
    gate_promote_p.add_argument(
        "--require-no-contamination",
        action="store_true",
        dest="require_no_contamination",
        help="Block if train/eval contamination was detected",
    )
    gate_promote_p.add_argument(
        "--require-signed",
        action="store_true",
        dest="require_signed",
        help="Block if passport has no attestation",
    )
    gate_promote_p.add_argument(
        "--json",
        default=None,
        dest="json_output",
        metavar="OUT.JSON",
        help="Optional path to write the JSON report",
    )

    # --- tim-check ---
    tim_p = subparsers.add_parser(
        "tim-check", help="Detect training-inference mismatch via KL divergence"
    )
    tim_p.add_argument(
        "--probes",
        required=True,
        help="JSONL file with prompt/train_log_probs/inference_log_probs",
    )
    tim_p.add_argument(
        "--threshold", type=float, default=0.1, help="Per-probe KL threshold (default 0.1)"
    )
    tim_p.add_argument("--stage", default=None)
    tim_p.add_argument("--max-kl", type=float, default=None, dest="max_kl")
    tim_p.add_argument("--json", dest="json_output", default=None, metavar="OUT.JSON")

    # --- lineage ---
    lineage_p = subparsers.add_parser(
        "lineage", help="Inspect and verify fine-tune lineage chains"
    )
    lineage_sub = lineage_p.add_subparsers(dest="lineage_command")
    lineage_show_p = lineage_sub.add_parser("show", help="Render and verify a lineage chain")
    lineage_show_p.add_argument(
        "--passports", nargs="+", required=True, help="Passport JSON files forming the chain"
    )
    lineage_show_p.add_argument(
        "--verify", action="store_true", help="Also verify parent hash integrity"
    )

    # --- retraction ---
    retraction_p = subparsers.add_parser(
        "retraction", help="Check training data for retracted scientific papers"
    )
    retraction_sub = retraction_p.add_subparsers(dest="retraction_command")
    retraction_check_p = retraction_sub.add_parser(
        "check", help="Check a passport's training DOIs for retractions"
    )
    retraction_check_p.add_argument("--passport", required=True)
    retraction_check_p.add_argument(
        "--known-retracted",
        default=None,
        dest="known_retracted",
        metavar="FILE",
        help="Text file with one retracted DOI per line",
    )
    retraction_check_p.add_argument("--stage", default=None)
    retraction_check_p.add_argument("--allow-rate", type=float, default=0.0, dest="allow_rate")

    # --- export (acquisition package) ---
    export_p = subparsers.add_parser(
        "export", help="Export acquisition-ready due diligence package"
    )
    export_sub = export_p.add_subparsers(dest="export_command")
    export_acq_p = export_sub.add_parser(
        "acquisition-package", help="Generate M&A technical due diligence package"
    )
    export_acq_p.add_argument("--passport", required=True, help="Path to passport JSON")
    export_acq_p.add_argument("--out", required=True, help="Output directory")
    export_acq_p.add_argument(
        "--no-regulation",
        action="store_true",
        dest="no_regulation",
        help="Skip regulation analysis (faster)",
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

    if args.command == "report":
        from provenir.report import ReportGenerator

        run_dir = Path(args.run_dir)
        run_report = ReportGenerator.from_run_dir(run_dir)
        out = Path(args.output) if args.output else run_dir / "report.html"
        run_report.save(out)
        print(f"Report written → {out}")
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
            passport_arg: str | None = getattr(args, "passport", None)
            if passport_arg is not None:
                push_result = client.push_with_passport(
                    Path(args.adapter_path),
                    hub_cfg,
                    Path(passport_arg).read_text(encoding="utf-8"),
                )
            else:
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

    if args.command == "verify-reliability":
        from provenir.environments import (
            ContainsVerifier,
            ExactAnswerVerifier,
            MathVerifier,
            Probe,
            PromotionBlocked,
            ReliabilityHarness,
            gate_promotion,
        )
        from provenir.environments.base import Verifier

        reliability_map: dict[str, Verifier] = {
            "exact_answer": ExactAnswerVerifier(),
            "math": MathVerifier(),
            "contains": ContainsVerifier(required=["the"]),
        }
        verifier = reliability_map[args.verifier]
        probes: list[Probe] = []
        for raw in Path(args.probes).read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            probes.append(
                Probe(
                    response=str(record["response"]),
                    reference=record.get("reference"),
                    should_pass=bool(record["should_pass"]),
                    label=str(record.get("label", "")),
                )
            )
        report = ReliabilityHarness().evaluate(verifier, probes)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(report.summary())
        for outcome in report.failures():
            print(f"  FAIL [{outcome.mode.value}] {outcome.detail} ({outcome.origin_label})")
        if args.output:
            print(f"  report → {args.output}")
        if args.stage:
            try:
                gate_promotion(report, args.stage, min_overall=args.min_overall)
                print(f"Promotion to {args.stage!r}: ALLOWED")
            except PromotionBlocked as exc:
                print(f"Promotion to {args.stage!r}: BLOCKED — {exc}")
                sys.exit(1)
        return

    if args.command == "reward-validity":
        from typing import Any, Mapping

        from provenir.core.abstractions import RewardFn
        from provenir.data.dataset import JsonlDataset as _RVJsonlDataset
        from provenir.environments import (
            ContainsVerifier,
            ExactAnswerVerifier,
            MathVerifier,
            RewardValidityBlocked,
            RewardValidityHarness,
            gate_reward_validity,
        )
        from provenir.environments.base import Verifier, VerifierReward
        from provenir.environments.reward_validity import TrainEval
        from provenir.observability import FlightRecorder, RewardHackingDetector
        from provenir.train.algorithms import GRPOConfig
        from provenir.train.rl import RLOrchestrator

        rv_verifier_map: dict[str, Verifier] = {
            "exact_answer": ExactAnswerVerifier(),
            "math": MathVerifier(),
            "contains": ContainsVerifier(required=["the"]),
        }
        rv_verifier = rv_verifier_map[args.reward]

        # Wrap the verifier as a RewardFn with kind="real" so the harness
        # identifies it as the real signal vs. the degenerate controls.
        class _RealVerifierReward(VerifierReward):
            kind: str = "real"

            @property
            def name(self) -> str:
                return str(self.verifier.name)

        real_reward: RewardFn = _RealVerifierReward(rv_verifier)

        # Load the probe file (same JSONL format as verify-reliability).
        from provenir.environments import Probe

        rv_probes: list[Probe] = []
        for _raw in Path(args.probes).read_text(encoding="utf-8").splitlines():
            _raw = _raw.strip()
            if not _raw:
                continue
            _rec = json.loads(_raw)
            rv_probes.append(
                Probe(
                    response=str(_rec["response"]),
                    reference=_rec.get("reference"),
                    should_pass=bool(_rec["should_pass"]),
                    label=str(_rec.get("label", "")),
                )
            )

        def _probe_accuracy(reward_fn: RewardFn) -> float:
            """Fraction of probes where reward score matches expected should_pass."""
            if not rv_probes:
                return 0.0
            correct = 0
            for probe in rv_probes:
                traj: Mapping[str, Any] = {
                    "prediction": probe.response,
                    "response": probe.response,
                    "reference": probe.reference,
                }
                score = reward_fn.score(traj)
                predicted_pass = score > 0.5
                if predicted_pass == probe.should_pass:
                    correct += 1
            return correct / len(rv_probes)

        # Build a TrainEval closure using RLOrchestrator.
        # With the stub backend, weights are not actually updated, so
        # final_score == base_score is honest: we document this in the help text.
        rl_dataset = _RVJsonlDataset.from_path("tests/fixtures/sample.jsonl")

        def _train_eval(reward_fn: RewardFn) -> tuple[float, float, float, dict[str, int]]:
            base = _probe_accuracy(reward_fn)
            rl_orch = RLOrchestrator(
                algorithm=GRPOConfig(group_size=4, max_steps=args.max_steps),
                verifier=rv_verifier,
                flight_recorder=FlightRecorder(),
                hacking_detector=RewardHackingDetector(),
            )
            rl_res = rl_orch.run(train_dataset=rl_dataset)
            final = _probe_accuracy(reward_fn)
            mean_rwd = rl_res.mean_reward
            anomalies: dict[str, int] = (
                {"anomaly_count": rl_res.anomaly_count} if rl_res.anomaly_count else {}
            )
            return base, final, mean_rwd, anomalies

        rv_train_eval: TrainEval = _train_eval

        harness = RewardValidityHarness(seed=args.seed)
        rv_report = harness.evaluate(real_reward, rv_train_eval)

        print(rv_report.summary())
        for _abl, _run in sorted(rv_report.runs.items(), key=lambda kv: kv[0].value):
            print(
                f"  [{_abl.value:11s}] base={_run.base_score:.3f}  "
                f"final={_run.final_score:.3f}  gain={_run.gain:+.3f}  "
                f"mean_reward={_run.mean_reward:.3f}  anomalies={_run.anomalies}"
            )
        print(f"validity={rv_report.validity:.3f}  spurious={rv_report.spurious}")
        print(f"hash={rv_report.content_hash()[:16]}...")

        if args.json_output:
            _rv_out = Path(args.json_output)
            _rv_out.parent.mkdir(parents=True, exist_ok=True)
            _rv_out.write_text(json.dumps(rv_report.to_dict(), indent=2), encoding="utf-8")
            print(f"  report → {args.json_output}")

        if args.stage:
            try:
                gate_reward_validity(rv_report, args.stage, min_validity=args.min_validity)
                print(f"Promotion to {args.stage!r}: ALLOWED")
            except RewardValidityBlocked as exc:
                print(f"Promotion to {args.stage!r}: BLOCKED — {exc}")
                sys.exit(1)
        return

    if args.command == "contamination":
        from provenir.eval.contamination import ContaminationChecker, ContaminationConfig

        train_ds = JsonlDataset.from_path(args.train)
        eval_ds = JsonlDataset.from_path(getattr(args, "eval"))
        checker = ContaminationChecker(
            ContaminationConfig(method=args.method, text_key=args.text_key)
        )
        contam_report = checker.check_datasets(train_ds, eval_ds)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(contam_report.to_dict(), indent=2), encoding="utf-8")
        verdict = "CLEAN" if contam_report.is_clean else "CONTAMINATED"
        print(f"Contamination check ({args.method}): {verdict}")
        print(f"  hits:               {len(contam_report.hits)}")
        print(f"  contamination rate: {contam_report.contamination_rate:.3f}  →  {args.output}")
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

    if args.command == "scan":
        from provenir.governance.scan import ModelScanner

        scanner = ModelScanner()
        scan_report = scanner.scan(args.path)
        print(scan_report.summary())
        for finding in scan_report.findings:
            severity = finding.severity.value.upper()
            print(f"  [{severity}] {finding.threat.value}: {finding.detail} ({finding.path})")
        if args.json_output:
            output_path = Path(args.json_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(scan_report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  report → {args.json_output}")
        if scan_report.unsafe():
            print("Result: UNSAFE — promotion blocked")
            sys.exit(1)
        print("Result: clean — no blocking findings")
        return

    if args.command == "passport":
        from provenir.governance.passport import ModelPassport

        if args.passport_command in (None, ""):
            passport_p.print_help()
            return

        if args.passport_command == "verify-hub":
            from provenir.governance.hub_verify import HubPassportVerifier

            hub_verifier = HubPassportVerifier(
                passport_store_dir=getattr(args, "passport_store_dir", None)
            )
            hub_report = hub_verifier.verify_local(
                args.model_dir, repo_id=getattr(args, "repo_id", "")
            )
            print(hub_report.summary())
            print(f"  files:          {len(hub_report.files)}")
            print(f"  composite hash: {hub_report.composite_hash[:16]}...")
            print(f"  passport found: {hub_report.passport_found}")
            print(f"  hash match:     {hub_report.passport_hash_match}")
            if hub_report.risk_flags:
                for flag in hub_report.risk_flags:
                    print(f"  risk: {flag}")
            if not hub_report.verified:
                sys.exit(1)
            return

        data = json.loads(Path(args.path).read_text(encoding="utf-8"))
        passport = ModelPassport.from_dict(data)
        if args.passport_command == "show":
            print(passport.to_markdown())
        elif args.passport_command == "verify":
            valid = passport.verify(args.key.encode("utf-8"))
            print(f"Attestation valid: {valid}")
        elif args.passport_command == "export":
            from provenir.governance.export import ExportFormat, export_passport
            fmt = ExportFormat(args.export_format)
            exported = export_passport(passport, fmt)
            if args.out:
                output_path = Path(args.out)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(exported, encoding="utf-8")
                print(f"Exported ({args.export_format}) → {args.out}")
            else:
                print(exported)
        elif args.passport_command == "sign-dsse":
            from provenir.governance.sigstore_signing import sign_dsse
            key = args.key.encode("utf-8")
            envelope = sign_dsse(passport, key)
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(envelope.to_json(), encoding="utf-8")
            print(f"DSSE envelope → {args.out}")
            print(f"  payload hash: {envelope.content_hash()[:16]}...")
        return

    if args.command == "regulation":
        from provenir.governance.passport import ModelPassport
        from provenir.governance.regulation import RegulationGenerator

        if args.reg_command in (None, ""):
            reg_p.print_help()
            return

        reg_passport_data = json.loads(Path(args.passport).read_text(encoding="utf-8"))
        reg_passport = ModelPassport.from_dict(reg_passport_data)

        reg_gen = RegulationGenerator(sme=getattr(args, "sme", False))
        if args.reg_command == "art53":
            reg_report = reg_gen.art53_training_data_summary(reg_passport)
        elif args.reg_command == "annex-iv":
            reg_report = RegulationGenerator().annex_iv_technical_file(reg_passport)
        elif args.reg_command in ("fda-pccp", "nist-rmf"):
            if args.reg_command == "fda-pccp":
                reg_report = reg_gen.fda_pccp_summary(reg_passport)
            else:
                reg_report = reg_gen.nist_ai_rmf_summary(reg_passport)
        else:
            reg_p.print_help()
            return

        print(reg_report.summary())
        score = reg_report.coverage_score()
        print(f"Coverage score: {score:.1%}")
        if args.reg_command in ("fda-pccp", "nist-rmf"):
            missing_count = len(reg_report.missing())
            print(f"  coverage: {reg_report.coverage_score():.0%}  missing: {missing_count} fields")

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(reg_report.markdown, encoding="utf-8")
            print(f"  report → {args.out}")

        fail_under: float | None = getattr(args, "fail_under", None)
        if fail_under is not None and score < fail_under:
            if args.reg_command in ("fda-pccp", "nist-rmf"):
                print(f"  FAIL: coverage {score:.0%} < required {fail_under:.0%}")
            else:
                print(
                    f"Coverage {score:.1%} is below --fail-under"
                    f" threshold {fail_under:.1%} — exit 1"
                )
            sys.exit(1)
        return

    if args.command == "tim-check":
        from typing import Any as _Any

        from provenir.environments.tim import TIMBlocked, TIMDetector, TIMReport, gate_tim
        tim_probes_data: list[dict[str, _Any]] = []
        for _tim_raw in Path(args.probes).read_text(encoding="utf-8").splitlines():
            _tim_raw = _tim_raw.strip()
            if not _tim_raw:
                continue
            tim_probes_data.append(json.loads(_tim_raw))
        tim_prompts = [str(p["prompt"]) for p in tim_probes_data]

        def probe_fn(prompt: str) -> tuple[list[float], list[float]]:
            for p in tim_probes_data:
                if p["prompt"] == prompt:
                    return list(p["train_log_probs"]), list(p["inference_log_probs"])
            return [], []

        tim_detector = TIMDetector(threshold=args.threshold)
        tim_report: TIMReport = tim_detector.detect(probe_fn, tim_prompts)
        print(tim_report.summary())
        print(f"  mean KL:       {tim_report.mean_kl:.4f}")
        print(f"  max KL:        {tim_report.max_kl:.4f}")
        print(f"  mismatch rate: {tim_report.mismatch_rate:.3f}")
        if args.json_output:
            output_path = Path(args.json_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(tim_report.to_dict(), indent=2), encoding="utf-8"
            )
            print(f"  report → {args.json_output}")
        if args.stage:
            try:
                gate_tim(tim_report, args.stage, max_mean_kl=args.max_kl)
                print(f"Promotion to {args.stage!r}: ALLOWED")
            except TIMBlocked as exc:
                print(f"Promotion to {args.stage!r}: BLOCKED — {exc}")
                sys.exit(1)
        return

    if args.command == "lineage":
        from provenir.governance.lineage import LineageChain, LineageVerifier
        from provenir.governance.passport import ModelPassport
        if args.lineage_command == "show":
            lineage_passports = [
                ModelPassport.from_dict(json.loads(Path(f).read_text(encoding="utf-8")))
                for f in args.passports
            ]
            lineage_verifier = LineageVerifier()
            chain: LineageChain = lineage_verifier.build_chain(lineage_passports)
            print(chain.to_ascii_tree() if chain.nodes else "(empty chain)")
            print(f"  valid: {chain.valid}  depth: {chain.depth()}")
            if not chain.valid:
                print(f"  error: {chain.error}")
            if args.verify:
                hashes_ok = lineage_verifier.verify_hashes(lineage_passports)
                print(f"  hash verification: {'PASS' if hashes_ok else 'FAIL'}")
                if not hashes_ok:
                    sys.exit(1)
        else:
            lineage_p.print_help()
        return

    if args.command == "retraction":
        from provenir.governance.passport import ModelPassport
        from provenir.governance.retraction import (
            RetractionBlocked,
            RetractionMonitor,
            RetractionReport,
            gate_retraction,
        )
        if args.retraction_command == "check":
            ret_passport = ModelPassport.from_dict(
                json.loads(Path(args.passport).read_text(encoding="utf-8"))
            )
            known_set: frozenset[str] = frozenset()
            if args.known_retracted:
                raw_lines = Path(args.known_retracted).read_text(encoding="utf-8").splitlines()
                known_set = frozenset(line.strip() for line in raw_lines if line.strip())
            ret_monitor = RetractionMonitor(known_retracted=known_set)
            ret_report: RetractionReport = ret_monitor.check_passport(ret_passport)
            print(ret_report.summary())
            print(f"  checked:    {len(ret_report.checked_dois)}")
            print(f"  retracted:  {len(ret_report.retracted_dois)}")
            print(f"  rate:       {ret_report.retraction_rate:.1%}")
            print(f"  risk level: {ret_report.risk_level}")
            if args.stage:
                try:
                    gate_retraction(ret_report, args.stage, allow_rate=args.allow_rate)
                    print(f"Promotion to {args.stage!r}: ALLOWED")
                except RetractionBlocked as exc:
                    print(f"Promotion to {args.stage!r}: BLOCKED — {exc}")
                    sys.exit(1)
        else:
            retraction_p.print_help()
        return

    if args.command == "export":
        if args.export_command == "acquisition-package":
            from provenir.governance.acquisition import (
                AcquisitionPackage,
                generate_acquisition_package,
            )
            from provenir.governance.passport import ModelPassport
            acq_passport = ModelPassport.from_dict(
                json.loads(Path(args.passport).read_text(encoding="utf-8"))
            )
            pkg: AcquisitionPackage = generate_acquisition_package(
                acq_passport, args.out, include_regulation=not args.no_regulation
            )
            print(pkg.summary())
            print(f"\nPackage written → {args.out}")
            print(f"  sections:     {len(pkg.sections)}")
            print(f"  package hash: {pkg.package_hash[:16]}...")
            if pkg.warnings:
                for w in pkg.warnings:
                    print(f"  ⚠ {w}")
        else:
            export_p.print_help()
        return

    if args.command == "rag-corpus":
        from provenir.governance.rag_corpus import (
            RAGCorpusBlocked,
            RAGCorpusScanner,
            gate_rag_corpus,
        )

        if args.rag_command == "scan":
            rag_train_hashes: frozenset[str] = frozenset()
            if args.train_hashes_file:
                _th_lines = Path(args.train_hashes_file).read_text(encoding="utf-8").splitlines()
                rag_train_hashes = frozenset(ln.strip() for ln in _th_lines if ln.strip())
            rag_known_retracted: frozenset[str] = frozenset()
            if args.known_retracted:
                _kr_lines = Path(args.known_retracted).read_text(encoding="utf-8").splitlines()
                rag_known_retracted = frozenset(ln.strip() for ln in _kr_lines if ln.strip())
            rag_scanner = RAGCorpusScanner(
                known_retracted=rag_known_retracted, train_hashes=rag_train_hashes
            )
            rag_report = rag_scanner.scan(args.corpus)
            print(rag_report.summary())
            if args.json_output:
                _rag_out = Path(args.json_output)
                _rag_out.parent.mkdir(parents=True, exist_ok=True)
                _rag_out.write_text(
                    json.dumps(rag_report.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"  report → {args.json_output}")
            if args.stage:
                try:
                    gate_rag_corpus(
                        rag_report,
                        args.stage,
                        protected_stages=[args.stage],
                        allow_pii=args.allow_pii,
                        allow_retraction=args.allow_retraction,
                    )
                    print(f"Promotion to {args.stage!r}: ALLOWED")
                except RAGCorpusBlocked as exc:
                    print(f"Promotion to {args.stage!r}: BLOCKED — {exc}")
                    sys.exit(1)
        else:
            rag_p.print_help()
        return

    if args.command == "gate":
        from provenir.governance.passport import ModelPassport
        from provenir.governance.promotion_gate import PromotionGate

        if args.gate_command == "promote":
            _gate_data = json.loads(Path(args.passport).read_text(encoding="utf-8"))
            _gate_passport = ModelPassport.from_dict(_gate_data)
            _promo_gate = PromotionGate(
                require_scan=args.require_scan,
                require_no_retraction=args.require_no_retraction,
                min_validity=args.min_validity,
                require_no_pii=args.require_no_pii,
                require_no_contamination=args.require_no_contamination,
                require_signed=args.require_signed,
            )
            _promo_result = _promo_gate.evaluate(_gate_passport, stage=args.stage)
            print(_promo_result.summary())
            if args.json_output:
                _gate_out = Path(args.json_output)
                _gate_out.parent.mkdir(parents=True, exist_ok=True)
                _gate_out.write_text(
                    json.dumps(_promo_result.to_dict(), indent=2), encoding="utf-8"
                )
                print(f"  report → {args.json_output}")
            if not _promo_result.passed:
                sys.exit(1)
        else:
            gate_p.print_help()
        return

    parser.print_help()
