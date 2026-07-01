from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from provenir.cli.main import main
from provenir.governance.bom import CodeComponent, DataComponent, ModelBOM
from provenir.governance.passport import PassportSigner


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def test_cli_contamination_clean(tmp_path, monkeypatch, capsys) -> None:
    train = tmp_path / "train.jsonl"
    eval_ds = tmp_path / "eval.jsonl"
    _write_jsonl(train, [{"prompt": "what is a cat", "response": "an animal"}])
    _write_jsonl(eval_ds, [{"prompt": "explain quantum tunnelling", "response": "x"}])
    out = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provenir",
            "contamination",
            str(train),
            str(eval_ds),
            "--output",
            str(out),
        ],
    )
    main()

    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["train_size"] == 1
    assert "CLEAN" in capsys.readouterr().out


def test_cli_contamination_detects_leak(tmp_path, monkeypatch, capsys) -> None:
    shared = "the mitochondria is the powerhouse of the cell"
    train = tmp_path / "train.jsonl"
    eval_ds = tmp_path / "eval.jsonl"
    _write_jsonl(train, [{"prompt": shared, "response": "y"}])
    _write_jsonl(eval_ds, [{"prompt": shared, "response": "y"}])
    out = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["provenir", "contamination", str(train), str(eval_ds), "--output", str(out)],
    )
    main()
    assert "CONTAMINATED" in capsys.readouterr().out


@pytest.mark.parametrize("algorithm", ["grpo", "dapo", "gspo"])
def test_cli_rl_runs(tmp_path, monkeypatch, capsys, algorithm) -> None:
    dataset = tmp_path / "train.jsonl"
    _write_jsonl(
        dataset,
        [
            {"prompt": "2+2?", "response": "4", "reference": "4"},
            {"prompt": "3+3?", "response": "6", "reference": "6"},
        ],
    )
    out = tmp_path / "rl.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provenir",
            "rl",
            "--dataset",
            str(dataset),
            "--algorithm",
            algorithm,
            "--verifier",
            "math",
            "--max-steps",
            "1",
            "--group-size",
            "4",
            "--output",
            str(out),
        ],
    )
    main()
    assert out.exists()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["algorithm"] == algorithm
    assert f"RL ({algorithm})" in capsys.readouterr().out


def test_cli_passport_show_and_verify(tmp_path, monkeypatch, capsys) -> None:
    bom = ModelBOM(
        model_id="demo-model",
        base_model="llama-3.2-1b",
        run_id="run-123",
        data=[DataComponent(name="train", content_hash="abc", num_records=10)],
        code=CodeComponent(git_sha="deadbeef", dependencies_hash="d1", framework="trl"),
        evals=[],
        hyperparameters={"lr": 0.001},
        created_at="2026-07-01T00:00:00Z",
    )
    signer = PassportSigner(key=b"secret-key", key_id="k1")
    passport = signer.sign(bom, signed_at="2026-07-01T00:00:00Z")
    passport_path = tmp_path / "passport.json"
    passport_path.write_text(passport.to_json(), encoding="utf-8")

    monkeypatch.setattr(
        sys, "argv", ["provenir", "passport", "show", str(passport_path)]
    )
    main()
    assert "demo-model" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["provenir", "passport", "verify", str(passport_path), "--key", "secret-key"],
    )
    main()
    assert "Attestation valid: True" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["provenir", "passport", "verify", str(passport_path), "--key", "wrong-key"],
    )
    main()
    assert "Attestation valid: False" in capsys.readouterr().out
