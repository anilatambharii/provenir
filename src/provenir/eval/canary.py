"""Canary-tagged private eval vaults for leakage detection.

A canary is a unique, hard-to-guess token stitched into every record of a
*private* evaluation set. If that token later shows up inside a training
corpus, the private eval set has leaked into training — a smoking-gun signal
of contamination that survives paraphrase, shuffling, and reformatting because
the token itself is verbatim.

Tokens are derived deterministically (SHA-256 of ``eval_set_id`` plus an
optional ``seed``) so minting is reproducible across machines and runs — no
timestamps or RNG involved.

Example
-------
>>> from provenir.data.dataset import JsonlDataset
>>> guard = CanaryGuard()
>>> canary = guard.mint("secret-eval-v1")
>>> eval_ds = JsonlDataset.from_records([{"prompt": "2 + 2 = ?"}])
>>> tagged = guard.tag(eval_ds, canary)
>>> # Simulate leakage: the tagged eval row ends up in training.
>>> train = JsonlDataset.from_records(list(tagged.records))
>>> guard.scan(train, canary)
[0]
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from provenir.data.dataset import JsonlDataset

_CANARY_PREFIX = "provenir-canary:"
_CANARY_FIELD = "_canary"


@dataclass(frozen=True)
class Canary:
    """A minted canary token bound to a specific eval set."""

    token: str
    eval_set_id: str

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("token must be a non-empty string")
        if not self.eval_set_id:
            raise ValueError("eval_set_id must be a non-empty string")


class CanaryGuard:
    """Mint, tag, and scan for canary tokens across datasets.

    Example
    -------
    >>> guard = CanaryGuard()
    >>> guard.mint("eval-a").token == guard.mint("eval-a").token
    True
    >>> guard.mint("eval-a").token == guard.mint("eval-b").token
    False
    """

    def mint(self, eval_set_id: str, seed: str = "") -> Canary:
        """Deterministically derive a unique canary token for ``eval_set_id``.

        The token is ``"provenir-canary:" + sha256(eval_set_id + "\\x00" + seed)``
        truncated to 16 hex chars. A NUL separator prevents ``("a", "bc")`` and
        ``("ab", "c")`` from colliding.
        """
        if not eval_set_id:
            raise ValueError("eval_set_id must be a non-empty string")
        payload = f"{eval_set_id}\x00{seed}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        return Canary(token=f"{_CANARY_PREFIX}{digest}", eval_set_id=eval_set_id)

    def tag(
        self,
        eval_ds: JsonlDataset,
        canary: Canary,
        text_key: str = "prompt",
    ) -> JsonlDataset:
        """Return a copy of ``eval_ds`` with the canary embedded in each record.

        The token is both appended to the record's text under ``text_key`` and
        stored in a dedicated ``_canary`` field, so leakage is detectable
        whether the trainer keeps the structured field or only the raw text.
        """
        tagged: list[dict[str, object]] = []
        for record in eval_ds.records:
            new_record = dict(record)
            existing = str(new_record.get(text_key, ""))
            new_record[text_key] = (
                f"{existing} {canary.token}" if existing else canary.token
            )
            new_record[_CANARY_FIELD] = canary.token
            tagged.append(new_record)
        return JsonlDataset.from_records(tagged)

    def scan(
        self,
        train: JsonlDataset,
        canary: Canary,
        text_key: str = "prompt",
    ) -> list[int]:
        """Return indices of training records containing ``canary``'s token."""
        leaked: list[int] = []
        for index, record in enumerate(train.records):
            if self._record_contains(record, canary.token, text_key):
                leaked.append(index)
        return leaked

    def detect_any(
        self,
        texts: list[str],
        canaries: list[Canary],
    ) -> dict[str, list[int]]:
        """Map each ``eval_set_id`` to the text indices that leaked its token.

        Only eval sets with at least one leak appear in the result.
        """
        result: dict[str, list[int]] = {}
        for canary in canaries:
            hits = [i for i, text in enumerate(texts) if canary.token in text]
            if hits:
                result.setdefault(canary.eval_set_id, []).extend(hits)
        return result

    @staticmethod
    def _record_contains(
        record: dict[str, object], token: str, text_key: str
    ) -> bool:
        if token in str(record.get(text_key, "")):
            return True
        if token in str(record.get(_CANARY_FIELD, "")):
            return True
        return False
