from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from provenir.core.abstractions import RunManifest

_NODE_TYPES = frozenset({"dataset", "run", "adapter", "eval", "merge", "model"})
_RELATIONS = frozenset(
    {"produced", "derived_from", "evaluated_by", "merged_into", "trained_on"}
)


@dataclass(frozen=True)
class LineageNode:
    """A content-addressed vertex in a provenance lineage DAG.

    Example:
        >>> node = LineageNode("ds-1", "dataset", "abc123", {"rows": 10})
        >>> node.node_type
        'dataset'
    """

    node_id: str
    node_type: str
    content_hash: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in _NODE_TYPES:
            raise ValueError(
                f"node_type must be one of {sorted(_NODE_TYPES)}, got {self.node_type!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the node."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content_hash": self.content_hash,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageNode:
        """Rebuild a :class:`LineageNode` from a mapping."""
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            content_hash=data["content_hash"],
            attributes=dict(data.get("attributes", {})),
        )


@dataclass(frozen=True)
class LineageEdge:
    """A directed parent->child relation between two lineage nodes.

    Example:
        >>> edge = LineageEdge("ds-1", "run-1", "trained_on")
        >>> edge.relation
        'trained_on'
    """

    parent_id: str
    child_id: str
    relation: str

    def __post_init__(self) -> None:
        if self.relation not in _RELATIONS:
            raise ValueError(
                f"relation must be one of {sorted(_RELATIONS)}, got {self.relation!r}"
            )

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable mapping of the edge."""
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "relation": self.relation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> LineageEdge:
        """Rebuild a :class:`LineageEdge` from a mapping."""
        return cls(
            parent_id=data["parent_id"],
            child_id=data["child_id"],
            relation=data["relation"],
        )


class LineageGraph:
    """An acyclic provenance graph of datasets, runs, adapters and evals.

    Nodes are content-addressed vertices; edges are typed parent->child
    relations. The graph enforces acyclicity: :meth:`add_edge` rejects any
    edge that would introduce a cycle, giving a tamper-evident,
    topologically-orderable lineage suitable for audit trails.

    Example:
        >>> g = LineageGraph()
        >>> g.add_node(LineageNode("ds", "dataset", "h1", {}))
        >>> g.add_node(LineageNode("run", "run", "h2", {}))
        >>> g.add_edge(LineageEdge("ds", "run", "trained_on"))
        >>> g.ancestors("run")
        {'ds'}
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}

    def add_node(self, node: LineageNode) -> None:
        """Add ``node`` to the graph (idempotent replace by ``node_id``)."""
        self._nodes[node.node_id] = node
        self._children.setdefault(node.node_id, set())
        self._parents.setdefault(node.node_id, set())

    def add_edge(self, edge: LineageEdge) -> None:
        """Add ``edge``, rejecting unknown nodes and cycle-forming edges.

        Raises:
            ValueError: if either endpoint is unknown, or if adding the edge
                would create a cycle in the DAG.
        """
        if edge.parent_id not in self._nodes:
            raise ValueError(f"unknown parent node {edge.parent_id!r}")
        if edge.child_id not in self._nodes:
            raise ValueError(f"unknown child node {edge.child_id!r}")
        if edge.parent_id == edge.child_id:
            raise ValueError(f"self-loop not allowed on {edge.parent_id!r}")
        # Adding parent->child creates a cycle iff parent is a descendant of child.
        if edge.parent_id in self.descendants(edge.child_id):
            raise ValueError(
                f"edge {edge.parent_id!r}->{edge.child_id!r} would create a cycle"
            )
        self._edges.append(edge)
        self._children[edge.parent_id].add(edge.child_id)
        self._parents[edge.child_id].add(edge.parent_id)

    def node(self, node_id: str) -> LineageNode:
        """Return the node with ``node_id`` or raise ``KeyError``."""
        return self._nodes[node_id]

    def ancestors(self, node_id: str) -> set[str]:
        """Return the set of all transitive parent node ids of ``node_id``."""
        return self._reachable(node_id, self._parents)

    def descendants(self, node_id: str) -> set[str]:
        """Return the set of all transitive child node ids of ``node_id``."""
        return self._reachable(node_id, self._children)

    def _reachable(self, node_id: str, adjacency: dict[str, set[str]]) -> set[str]:
        if node_id not in self._nodes:
            raise ValueError(f"unknown node {node_id!r}")
        seen: set[str] = set()
        stack = list(adjacency.get(node_id, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, set()))
        return seen

    def roots(self) -> list[str]:
        """Return node ids with no parents, sorted for determinism."""
        return sorted(nid for nid in self._nodes if not self._parents.get(nid))

    def leaves(self) -> list[str]:
        """Return node ids with no children, sorted for determinism."""
        return sorted(nid for nid in self._nodes if not self._children.get(nid))

    def provenance_of(self, node_id: str) -> list[LineageNode]:
        """Return the topologically-ordered ancestor chain of ``node_id``.

        The result contains every ancestor node (excluding ``node_id`` itself)
        ordered so that parents precede children.
        """
        ancestor_ids = self.ancestors(node_id)
        ordered = [nid for nid in self._topological_order() if nid in ancestor_ids]
        return [self._nodes[nid] for nid in ordered]

    def _topological_order(self) -> list[str]:
        indegree = {nid: len(self._parents.get(nid, set())) for nid in self._nodes}
        queue: deque[str] = deque(sorted(n for n, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child in sorted(self._children.get(current, set())):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        return order

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of the whole graph."""
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageGraph:
        """Rebuild a :class:`LineageGraph` from a mapping produced by ``to_dict``."""
        graph = cls()
        for node_data in data.get("nodes", []):
            graph.add_node(LineageNode.from_dict(node_data))
        for edge_data in data.get("edges", []):
            graph.add_edge(LineageEdge.from_dict(edge_data))
        return graph

    def to_dot(self) -> str:
        """Return a Graphviz DOT representation of the lineage DAG.

        Example:
            >>> g = LineageGraph()
            >>> g.add_node(LineageNode("ds", "dataset", "h1", {}))
            >>> "ds" in g.to_dot()
            True
        """
        lines = ["digraph lineage {"]
        for node in self._nodes.values():
            label = f"{node.node_id}\\n({node.node_type})"
            lines.append(f'  "{node.node_id}" [label="{label}"];')
        for edge in self._edges:
            lines.append(
                f'  "{edge.parent_id}" -> "{edge.child_id}" '
                f'[label="{edge.relation}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        """Persist the graph to ``path`` as JSON and return the path."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LineageGraph:
        """Load a graph previously written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


class LineageStore:
    """Directory-backed store for provenance lineage graphs.

    Example:
        >>> import tempfile
        >>> from provenir.core.abstractions import RunManifest
        >>> with tempfile.TemporaryDirectory() as d:
        ...     store = LineageStore(d)
        ...     manifest = RunManifest(run_id="r1", config_hash="c", dataset_hash="ds")
        ...     graph = store.record_run(manifest, "dshash", parent_ids=[])
        ...     sorted(n.node_id for n in graph.provenance_of("run:r1"))
        ['dataset:dshash']
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def record_run(
        self,
        manifest: RunManifest,
        dataset_hash: str,
        parent_ids: list[str],
    ) -> LineageGraph:
        """Record a run and its dataset as a lineage graph and persist it.

        Adds a ``dataset`` node and a ``run`` node, links the dataset to the
        run with a ``trained_on`` edge, and links each ``parent_ids`` entry
        (assumed already-present dataset nodes) to the run. The resulting
        graph is saved as ``<run_id>.lineage.json`` and returned.
        """
        graph = LineageGraph()
        dataset_node_id = f"dataset:{dataset_hash}"
        run_node_id = f"run:{manifest.run_id}"

        graph.add_node(
            LineageNode(
                node_id=dataset_node_id,
                node_type="dataset",
                content_hash=dataset_hash,
                attributes={},
            )
        )
        graph.add_node(
            LineageNode(
                node_id=run_node_id,
                node_type="run",
                content_hash=manifest.config_hash,
                attributes={
                    "seed": manifest.seed,
                    "git_sha": manifest.git_sha,
                    "config_hash": manifest.config_hash,
                    "dataset_hash": manifest.dataset_hash,
                },
            )
        )
        graph.add_edge(
            LineageEdge(
                parent_id=dataset_node_id,
                child_id=run_node_id,
                relation="trained_on",
            )
        )
        present = {dataset_node_id, run_node_id}
        for parent_id in parent_ids:
            if parent_id not in present:
                graph.add_node(
                    LineageNode(
                        node_id=parent_id,
                        node_type="dataset",
                        content_hash=parent_id,
                        attributes={},
                    )
                )
                present.add(parent_id)
            graph.add_edge(
                LineageEdge(
                    parent_id=parent_id,
                    child_id=run_node_id,
                    relation="derived_from",
                )
            )

        graph.save(self.root_dir / f"{manifest.run_id}.lineage.json")
        return graph

    def load(self, run_id: str) -> LineageGraph:
        """Load the lineage graph previously recorded for ``run_id``."""
        return LineageGraph.load(self.root_dir / f"{run_id}.lineage.json")


__all__ = [
    "LineageNode",
    "LineageEdge",
    "LineageGraph",
    "LineageStore",
]
