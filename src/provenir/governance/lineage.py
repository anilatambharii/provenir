"""LoRA/Fine-tune Lineage Signing — tamper-evident fine-tuning chain provenance.

A production model is the tip of a chain::

    base -> SFT -> RLHF -> domain-finetune -> LoRA-adapter -> production

Each step's :class:`~provenir.governance.bom.ModelBOM` records the SHA-256
content hash of its parent BOM in ``parent_passport_hash``.  Because that field
is itself part of the signed content, the chain is tamper-evident: modifying any
ancestor BOM changes its hash, which breaks every descendant's back-pointer.

Typical usage::

    from provenir.governance.lineage import LineageVerifier, link_parent

    # Link child BOM to its parent passport before signing
    sft_bom = link_parent(sft_bom_draft, base_passport)
    sft_passport = PassportSigner(key).sign(sft_bom)

    # Verify the full chain
    verifier = LineageVerifier()
    chain = verifier.build_chain([base_passport, sft_passport, rlhf_passport])
    assert chain.valid
    print(chain.to_ascii_tree())
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from provenir.governance.bom import ModelBOM
    from provenir.governance.passport import ModelPassport


@dataclass(frozen=True)
class LineageNode:
    """A single node in a fine-tuning lineage chain.

    Example:
        >>> node = LineageNode(
        ...     model_id="sft", run_id="run-1",
        ...     passport_hash="abc", parent_hash=None, depth=0,
        ... )
        >>> node.depth
        0
    """

    model_id: str
    run_id: str
    passport_hash: str
    parent_hash: str | None
    depth: int  # 0 = base model, 1 = first fine-tune, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "run_id": self.run_id,
            "passport_hash": self.passport_hash,
            "parent_hash": self.parent_hash,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class LineageChain:
    """An ordered list of :class:`LineageNode` objects forming a fine-tune chain.

    ``nodes[0]`` is the base model (depth 0); ``nodes[-1]`` is the production
    tip.

    Example:
        >>> chain = LineageChain()
        >>> chain.valid
        True
    """

    nodes: list[LineageNode] = field(default_factory=list)
    valid: bool = True
    error: str = ""

    def depth(self) -> int:
        """Return the number of fine-tuning steps (len(nodes) - 1)."""
        return max(len(self.nodes) - 1, 0)

    def root(self) -> LineageNode | None:
        """Return the base-model node (depth 0), or None if the chain is empty."""
        return self.nodes[0] if self.nodes else None

    def tip(self) -> LineageNode | None:
        """Return the production-tip node (last in chain), or None if empty."""
        return self.nodes[-1] if self.nodes else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "valid": self.valid,
            "error": self.error,
            "depth": self.depth(),
        }

    def to_ascii_tree(self) -> str:
        """Render the chain as a human-readable ASCII tree.

        Example output::

            [base] llama3-base (run: abc123) hash: deadbeef...
              └─ [ft:1] sft-model (run: def456) hash: 12345678...
                   └─ [ft:2] rlhf-model (run: ghi789) hash: abcdef01...
        """
        if not self.nodes:
            return "(empty chain)"
        lines: list[str] = []
        for node in self.nodes:
            indent = "  " * node.depth
            prefix = "[base]" if node.depth == 0 else f"[ft:{node.depth}]"
            h = node.passport_hash
            short_hash = h[:8] if len(h) >= 8 else h
            if node.depth == 0:
                lines.append(
                    f"{prefix} {node.model_id}"
                    f" (run: {node.run_id}) hash: {short_hash}..."
                )
            else:
                connector = f"{indent[:-2]}  └─ "
                lines.append(
                    f"{connector}{prefix} {node.model_id}"
                    f" (run: {node.run_id}) hash: {short_hash}..."
                )
        return "\n".join(lines)


class LineageVerifier:
    """Verify and reconstruct a fine-tuning lineage chain from a list of passports.

    Example:
        >>> verifier = LineageVerifier()
        >>> verifier  # doctest: +ELLIPSIS
        <provenir.governance.lineage.LineageVerifier object at 0x...>
    """

    def __init__(self) -> None:
        pass

    def build_chain(self, passports: list[ModelPassport]) -> LineageChain:
        """Build and validate a :class:`LineageChain` from ``passports``.

        The passports may be provided in any order.  The chain is assembled by
        following ``parent_passport_hash`` links from the root (the passport
        whose BOM has ``parent_passport_hash=None``) to the tip.

        Returns an *invalid* :class:`LineageChain` if:
        - No root is found (all passports have parent hashes).
        - Multiple roots are found.
        - A parent hash points to a passport not in the provided list.
        - A cycle is detected (a hash appears more than once in the walk).
        """
        # Index by content hash
        by_hash: dict[str, ModelPassport] = {
            p.bom.content_hash(): p for p in passports
        }

        # Find roots (parent_passport_hash is None)
        roots = [p for p in passports if p.bom.parent_passport_hash is None]

        if len(roots) == 0:
            return LineageChain(
                valid=False,
                error="no root found (all passports have parent hashes)",
            )
        if len(roots) > 1:
            n = len(roots)
            return LineageChain(
                valid=False,
                error=f"{n} root passports found; expected exactly 1",
            )

        # Walk chain from root
        nodes: list[LineageNode] = []
        visited: set[str] = set()
        current: ModelPassport = roots[0]
        depth = 0

        while True:
            h = current.bom.content_hash()
            if h in visited:
                return LineageChain(
                    valid=False,
                    error=f"cycle detected at hash {h[:16]}...",
                )
            visited.add(h)

            nodes.append(
                LineageNode(
                    model_id=current.bom.model_id,
                    run_id=current.bom.run_id,
                    passport_hash=h,
                    parent_hash=current.bom.parent_passport_hash,
                    depth=depth,
                )
            )

            # Find the next node whose parent_passport_hash points to current
            children = [p for p in passports if p.bom.parent_passport_hash == h]

            if not children:
                # We have reached the tip
                break
            if len(children) > 1:
                return LineageChain(
                    valid=False,
                    error=f"fork detected at depth {depth}: {len(children)} children",
                )

            next_passport = children[0]
            parent_hash_claimed = next_passport.bom.parent_passport_hash
            if parent_hash_claimed not in by_hash:
                return LineageChain(
                    valid=False,
                    error=(
                        f"broken chain: parent hash {parent_hash_claimed}"
                        " not in provided passports"
                    ),
                )

            current = next_passport
            depth += 1

        # Check we consumed all passports (no dangling nodes)
        if len(nodes) != len(passports):
            unvisited = set(by_hash.keys()) - visited
            n_unvisited = len(unvisited)
            return LineageChain(
                valid=False,
                error=(
                    f"disconnected passports detected:"
                    f" {n_unvisited} node(s) unreachable from root"
                ),
            )

        return LineageChain(nodes=nodes, valid=True)

    def verify_hashes(self, passports: list[ModelPassport]) -> bool:
        """Return True if all ``parent_passport_hash`` values are internally consistent.

        For each non-root passport P, checks that ``P.bom.parent_passport_hash``
        equals the ``content_hash()`` of the parent BOM found in ``passports``.
        Returns True only when every declared parent hash matches the actual
        content hash of the corresponding parent in the list.
        """
        by_hash: dict[str, ModelPassport] = {
            p.bom.content_hash(): p for p in passports
        }

        for passport in passports:
            declared = passport.bom.parent_passport_hash
            if declared is None:
                continue  # root — nothing to verify
            if declared not in by_hash:
                return False
            # The declared hash should match the parent's actual content hash
            parent = by_hash[declared]
            if parent.bom.content_hash() != declared:  # pragma: no cover
                return False

        return True


def link_parent(child_bom: ModelBOM, parent_passport: ModelPassport) -> ModelBOM:
    """Return a new :class:`~provenir.governance.bom.ModelBOM` linked to
    ``parent_passport``.

    Sets ``child_bom.parent_passport_hash`` to the SHA-256 content hash of
    ``parent_passport.bom``.  Uses :func:`dataclasses.replace` so the original
    ``child_bom`` is unchanged (frozen dataclass semantics preserved).

    Example::

        sft_bom = link_parent(sft_bom_draft, base_passport)
        # sft_bom.parent_passport_hash == base_passport.bom.content_hash()
    """
    return replace(child_bom, parent_passport_hash=parent_passport.bom.content_hash())
