"""Tests for the LoRA/Fine-tune Lineage Signing module.

Covers :mod:`provenir.governance.lineage` — :class:`LineageNode`,
:class:`LineageChain`, :class:`LineageVerifier`, and :func:`link_parent`.
"""

from __future__ import annotations

import pytest

from provenir.governance.bom import (
    CodeComponent,
    DataComponent,
    EvalComponent,
    ModelBOM,
)
from provenir.governance.lineage import (
    LineageNode,
    LineageVerifier,
    link_parent,
)
from provenir.governance.passport import ModelPassport, PassportSigner

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_SIGNER = PassportSigner(b"test")


def _make_bom(model_id: str, run_id: str) -> ModelBOM:
    return ModelBOM(
        model_id=model_id,
        base_model="llama3-70b",
        run_id=run_id,
        data=[DataComponent(name="train", content_hash="datahash", num_records=100)],
        code=CodeComponent(git_sha="abc123", dependencies_hash="dephash", framework="trl"),
        evals=[EvalComponent(benchmark="mmlu", score=0.72)],
        hyperparameters={"lr": 1e-4},
    )


@pytest.fixture()
def base_passport() -> ModelPassport:
    bom = _make_bom("llama3-base", "run-base")
    return _SIGNER.sign(bom)


@pytest.fixture()
def sft_passport(base_passport: ModelPassport) -> ModelPassport:
    bom = link_parent(_make_bom("sft-model", "run-sft"), base_passport)
    return _SIGNER.sign(bom)


@pytest.fixture()
def rlhf_passport(sft_passport: ModelPassport) -> ModelPassport:
    bom = link_parent(_make_bom("rlhf-model", "run-rlhf"), sft_passport)
    return _SIGNER.sign(bom)


@pytest.fixture()
def chain_passports(
    base_passport: ModelPassport,
    sft_passport: ModelPassport,
    rlhf_passport: ModelPassport,
) -> list[ModelPassport]:
    return [base_passport, sft_passport, rlhf_passport]


# ---------------------------------------------------------------------------
# 1. link_parent sets parent_passport_hash to parent's BOM content_hash
# ---------------------------------------------------------------------------

def test_link_parent_sets_correct_hash(base_passport: ModelPassport) -> None:
    child_bom = _make_bom("sft-model", "run-sft")
    linked = link_parent(child_bom, base_passport)
    assert linked.parent_passport_hash == base_passport.bom.content_hash()


# ---------------------------------------------------------------------------
# 2. link_parent returns a new ModelBOM; original is unchanged
# ---------------------------------------------------------------------------

def test_link_parent_returns_new_bom(base_passport: ModelPassport) -> None:
    child_bom = _make_bom("sft-model", "run-sft")
    linked = link_parent(child_bom, base_passport)
    assert linked is not child_bom
    assert child_bom.parent_passport_hash is None  # original unchanged


# ---------------------------------------------------------------------------
# 3. build_chain with 3-passport chain → valid=True, len(nodes)==3
# ---------------------------------------------------------------------------

def test_build_chain_three_passports(chain_passports: list[ModelPassport]) -> None:
    chain = LineageVerifier().build_chain(chain_passports)
    assert chain.valid is True
    assert len(chain.nodes) == 3


# ---------------------------------------------------------------------------
# 4. LineageChain.depth() == 2 for a 3-node chain
# ---------------------------------------------------------------------------

def test_chain_depth(chain_passports: list[ModelPassport]) -> None:
    chain = LineageVerifier().build_chain(chain_passports)
    assert chain.depth() == 2


# ---------------------------------------------------------------------------
# 5. LineageChain.root().model_id == base model id
# ---------------------------------------------------------------------------

def test_chain_root_model_id(chain_passports: list[ModelPassport]) -> None:
    chain = LineageVerifier().build_chain(chain_passports)
    root = chain.root()
    assert root is not None
    assert root.model_id == "llama3-base"


# ---------------------------------------------------------------------------
# 6. LineageChain.tip().model_id == rlhf model id
# ---------------------------------------------------------------------------

def test_chain_tip_model_id(chain_passports: list[ModelPassport]) -> None:
    chain = LineageVerifier().build_chain(chain_passports)
    tip = chain.tip()
    assert tip is not None
    assert tip.model_id == "rlhf-model"


# ---------------------------------------------------------------------------
# 7. to_ascii_tree() is non-empty and contains all three model ids
# ---------------------------------------------------------------------------

def test_ascii_tree_contains_all_model_ids(chain_passports: list[ModelPassport]) -> None:
    chain = LineageVerifier().build_chain(chain_passports)
    tree = chain.to_ascii_tree()
    assert tree  # non-empty
    assert "llama3-base" in tree
    assert "sft-model" in tree
    assert "rlhf-model" in tree


# ---------------------------------------------------------------------------
# 8. build_chain with single passport (no parent) → valid=True, depth==0
# ---------------------------------------------------------------------------

def test_build_chain_single_passport(base_passport: ModelPassport) -> None:
    chain = LineageVerifier().build_chain([base_passport])
    assert chain.valid is True
    assert chain.depth() == 0


# ---------------------------------------------------------------------------
# 9. build_chain with two passports that both have parent hashes → valid=False
# ---------------------------------------------------------------------------

def test_build_chain_no_root() -> None:
    bom_a = _make_bom("model-a", "run-a")
    bom_b = _make_bom("model-b", "run-b")
    # Give both a fake parent hash so neither is a root
    from dataclasses import replace as dc_replace
    bom_a_linked = dc_replace(bom_a, parent_passport_hash="fakehash1")
    bom_b_linked = dc_replace(bom_b, parent_passport_hash="fakehash2")
    p_a = _SIGNER.sign(bom_a_linked)
    p_b = _SIGNER.sign(bom_b_linked)
    chain = LineageVerifier().build_chain([p_a, p_b])
    assert chain.valid is False
    assert "no root" in chain.error


# ---------------------------------------------------------------------------
# 10. build_chain with broken chain (parent hash not in list) → valid=False
# ---------------------------------------------------------------------------

def test_build_chain_broken_link(base_passport: ModelPassport) -> None:
    from dataclasses import replace as dc_replace
    # Create a child that claims a parent hash that doesn't exist in the list
    bad_bom = dc_replace(
        _make_bom("orphan", "run-orphan"),
        parent_passport_hash="0" * 64,  # non-existent hash
    )
    orphan_passport = _SIGNER.sign(bad_bom)
    chain = LineageVerifier().build_chain([base_passport, orphan_passport])
    # base_passport is a valid root but orphan_passport is disconnected
    assert chain.valid is False


# ---------------------------------------------------------------------------
# 11. verify_hashes with correct chain → True
# ---------------------------------------------------------------------------

def test_verify_hashes_correct(chain_passports: list[ModelPassport]) -> None:
    result = LineageVerifier().verify_hashes(chain_passports)
    assert result is True


# ---------------------------------------------------------------------------
# 12. verify_hashes with tampered parent hash → False
# ---------------------------------------------------------------------------

def test_verify_hashes_tampered() -> None:
    from dataclasses import replace as dc_replace
    base_bom = _make_bom("base-model", "run-base")
    base_passport = _SIGNER.sign(base_bom)

    # Create a child with a WRONG parent hash (not the actual parent's hash)
    wrong_hash = "a" * 64
    child_bom = dc_replace(
        _make_bom("child-model", "run-child"),
        parent_passport_hash=wrong_hash,
    )
    child_passport = _SIGNER.sign(child_bom)

    # verify_hashes should return False because wrong_hash is not in our passport list
    result = LineageVerifier().verify_hashes([base_passport, child_passport])
    assert result is False


# ---------------------------------------------------------------------------
# 13. LineageNode.to_dict() has correct keys
# ---------------------------------------------------------------------------

def test_lineage_node_to_dict() -> None:
    node = LineageNode(
        model_id="test-model",
        run_id="run-1",
        passport_hash="abc123",
        parent_hash=None,
        depth=0,
    )
    d = node.to_dict()
    assert set(d.keys()) == {"model_id", "run_id", "passport_hash", "parent_hash", "depth"}
    assert d["model_id"] == "test-model"
    assert d["depth"] == 0
    assert d["parent_hash"] is None


# ---------------------------------------------------------------------------
# 14. LineageChain.to_dict() has nodes, valid, error, depth keys
# ---------------------------------------------------------------------------

def test_lineage_chain_to_dict(chain_passports: list[ModelPassport]) -> None:
    chain = LineageVerifier().build_chain(chain_passports)
    d = chain.to_dict()
    assert "nodes" in d
    assert "valid" in d
    assert "error" in d
    assert "depth" in d
    assert d["valid"] is True
    assert d["depth"] == 2
    assert len(d["nodes"]) == 3


# ---------------------------------------------------------------------------
# 15. parent_passport_hash is included in canonical_json (change it → different hash)
# ---------------------------------------------------------------------------

def test_parent_passport_hash_in_canonical_json(base_passport: ModelPassport) -> None:
    from dataclasses import replace as dc_replace
    child_bom = _make_bom("child", "run-child")
    linked = link_parent(child_bom, base_passport)
    unlinked = dc_replace(child_bom, parent_passport_hash=None)
    # The canonical JSON must differ when parent_passport_hash changes
    assert linked.canonical_json() != unlinked.canonical_json()
    assert linked.content_hash() != unlinked.content_hash()


# ---------------------------------------------------------------------------
# 16. BOM to_dict() emits "parent_passport_hash" key (None for base, str for ft)
# ---------------------------------------------------------------------------

def test_bom_to_dict_emits_parent_passport_hash_key(
    base_passport: ModelPassport,
    sft_passport: ModelPassport,
) -> None:
    base_d = base_passport.bom.to_dict()
    assert "parent_passport_hash" in base_d
    assert base_d["parent_passport_hash"] is None

    sft_d = sft_passport.bom.to_dict()
    assert "parent_passport_hash" in sft_d
    assert isinstance(sft_d["parent_passport_hash"], str)
    assert len(sft_d["parent_passport_hash"]) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# 17. from_dict round-trip: passport with parent_passport_hash serialises correctly
# ---------------------------------------------------------------------------

def test_from_dict_round_trip_with_parent_hash(sft_passport: ModelPassport) -> None:
    d = sft_passport.to_dict()
    restored = ModelPassport.from_dict(d)
    assert restored.bom.parent_passport_hash == sft_passport.bom.parent_passport_hash
    assert restored.bom.content_hash() == sft_passport.bom.content_hash()


# ---------------------------------------------------------------------------
# 18. passport.to_markdown() includes ## Lineage section with parent hash
# ---------------------------------------------------------------------------

def test_to_markdown_lineage_section_with_parent(sft_passport: ModelPassport) -> None:
    md = sft_passport.to_markdown()
    assert "## Lineage" in md
    assert "Parent passport hash:" in md
    assert sft_passport.bom.parent_passport_hash is not None
    assert sft_passport.bom.parent_passport_hash[:8] in md


def test_to_markdown_lineage_section_base_model(base_passport: ModelPassport) -> None:
    md = base_passport.to_markdown()
    assert "## Lineage" in md
    assert "base model" in md
    assert "no parent passport" in md
