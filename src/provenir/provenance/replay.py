from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from provenir.core.manifest import RunManifestStore
from provenir.provenance.fingerprint import (
    EnvironmentFingerprint,
    kernel_determinism_flags,
)


@dataclass(frozen=True)
class ReplayVerification:
    """Result of verifying a run can be deterministically replayed.

    Example:
        >>> v = ReplayVerification(
        ...     matches=True, config_hash_match=True, dataset_hash_match=True,
        ...     env_match=True, differences=[],
        ... )
        >>> v.reproducible
        True
    """

    matches: bool
    config_hash_match: bool
    dataset_hash_match: bool
    env_match: bool
    differences: list[str] = field(default_factory=list)

    @property
    def reproducible(self) -> bool:
        """True when config, dataset and environment all match exactly."""
        return (
            self.matches
            and self.config_hash_match
            and self.dataset_hash_match
            and self.env_match
        )


class ReplayEngine:
    """Verify and describe deterministic replays against stored manifests.

    The engine loads a previously-recorded :class:`RunManifest` and compares
    it against the *current* config/dataset/environment, producing a
    tamper-evident, human-readable verification report.

    Example:
        >>> import tempfile
        >>> from provenir.core.abstractions import RunManifest
        >>> from provenir.core.manifest import RunManifestStore
        >>> with tempfile.TemporaryDirectory() as d:
        ...     store = RunManifestStore(d)
        ...     _ = store.save(RunManifest(run_id="r1", config_hash="c", dataset_hash="ds"))
        ...     engine = ReplayEngine(store)
        ...     engine.verify("r1", "c", "ds").reproducible
        True
    """

    def __init__(self, store: RunManifestStore) -> None:
        self.store = store

    def verify(
        self,
        run_id: str,
        current_config_hash: str,
        current_dataset_hash: str,
        current_fingerprint: EnvironmentFingerprint | None = None,
    ) -> ReplayVerification:
        """Compare a stored run against the current environment.

        Args:
            run_id: Identifier of the stored run to verify.
            current_config_hash: Config hash of the current attempt.
            current_dataset_hash: Dataset hash of the current attempt.
            current_fingerprint: Optional current environment fingerprint. When
                provided its packages hash is compared against the manifest's
                stored ``hardware_fingerprint``; when ``None`` the environment
                is treated as unverified (``env_match`` False).

        Returns:
            A :class:`ReplayVerification` describing every mismatch.
        """
        manifest = self.store.load(run_id)
        differences: list[str] = []

        config_hash_match = manifest.config_hash == current_config_hash
        if not config_hash_match:
            differences.append(
                f"config_hash differs: stored={manifest.config_hash!r} "
                f"current={current_config_hash!r}"
            )

        dataset_hash_match = manifest.dataset_hash == current_dataset_hash
        if not dataset_hash_match:
            differences.append(
                f"dataset_hash differs: stored={manifest.dataset_hash!r} "
                f"current={current_dataset_hash!r}"
            )

        if current_fingerprint is None:
            env_match = False
            differences.append(
                "environment not verified: no current fingerprint supplied"
            )
        else:
            env_match = manifest.hardware_fingerprint == current_fingerprint.packages_hash
            if not env_match:
                differences.append(
                    f"environment differs: stored fingerprint="
                    f"{manifest.hardware_fingerprint!r} "
                    f"current packages_hash={current_fingerprint.packages_hash!r}"
                )

        matches = config_hash_match and dataset_hash_match and env_match
        return ReplayVerification(
            matches=matches,
            config_hash_match=config_hash_match,
            dataset_hash_match=dataset_hash_match,
            env_match=env_match,
            differences=differences,
        )

    def replay_command(self, run_id: str) -> dict[str, Any]:
        """Return a reproducibility recipe for re-running ``run_id``.

        The recipe bundles everything needed to reproduce the run: the seed,
        the config and dataset hashes, the git SHA, the stored hardware
        fingerprint, and the recommended kernel-determinism environment flags.

        Example:
            >>> import tempfile
            >>> from provenir.core.abstractions import RunManifest
            >>> from provenir.core.manifest import RunManifestStore
            >>> with tempfile.TemporaryDirectory() as d:
            ...     store = RunManifestStore(d)
            ...     _ = store.save(
            ...         RunManifest(run_id="r1", seed=7, config_hash="c", git_sha="abc")
            ...     )
            ...     recipe = ReplayEngine(store).replay_command("r1")
            ...     recipe["seed"], recipe["git_sha"]
            (7, 'abc')
        """
        manifest = self.store.load(run_id)
        return {
            "run_id": manifest.run_id,
            "seed": manifest.seed,
            "config_hash": manifest.config_hash,
            "dataset_hash": manifest.dataset_hash,
            "git_sha": manifest.git_sha,
            "hardware_fingerprint": manifest.hardware_fingerprint,
            "dependencies_lockfile": manifest.dependencies_lockfile,
            "env_flags": kernel_determinism_flags(),
        }


__all__ = [
    "ReplayVerification",
    "ReplayEngine",
]
