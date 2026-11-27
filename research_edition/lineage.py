"""A portable lineage reconstruction example on small z/y/x volumes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .association_replay import CANDIDATE_RULES, rule_candidate_edges


@dataclass(frozen=True, order=True)
class Node:
    frame: int
    z: int
    y: int
    x: int

    @property
    def coordinates(self) -> tuple[int, int, int]:
        return self.z, self.y, self.x


Edge = tuple[Node, Node]


def distance(source: Node, target: Node, spacing: tuple[float, float, float]) -> float:
    return float(np.linalg.norm((np.asarray(source.coordinates) - target.coordinates) * spacing))


def reconstruct(frames: list[list[Node]], spacing: tuple[float, float, float],
                max_step: float, max_gap: float) -> set[Edge]:
    """Link adjacent frames, then add only unique one-frame continuations."""
    if len(spacing) != 3 or not all(np.isfinite(s) and s > 0 for s in (*spacing, max_step, max_gap)):
        raise ValueError("spacing and distance caps must be positive and finite")
    if any(len(set(nodes)) != len(nodes) or any(node.frame != t for node in nodes)
           for t, nodes in enumerate(frames)):
        raise ValueError("expected unique nodes in their matching frame")
    incoming = Counter()
    outgoing = Counter()
    edges = set()
    for first, second in zip(frames, frames[1:]):
        if not first or not second:
            continue
        distances = np.array([[distance(a, b, spacing) for b in second] for a in first])
        raw = -np.square(distances / max_step)
        exp = np.exp(raw - raw.max(axis=0, keepdims=True))
        probs = exp / exp.sum(axis=0, keepdims=True)
        candidates = rule_candidate_edges(CANDIDATE_RULES[0], probs, raw) & (distances <= max_step)
        ranked = sorted((-probs[i,j], first[i], second[j]) for i,j in np.argwhere(candidates))
        for _, source, target in ranked:
            if incoming[target] == 0 and outgoing[source] < 2:
                edges.add((source, target))
                incoming[target] += 1
                outgoing[source] += 1
    for first, third in zip(frames, frames[2:]):
        proposed = [(a,b) for a in first for b in third
                    if outgoing[a] == 0 and incoming[b] == 0 and distance(a,b,spacing) <= max_gap]
        sources = Counter(a for a,_ in proposed)
        targets = Counter(b for _,b in proposed)
        for source, target in proposed:
            if sources[source] == targets[target] == 1:
                edges.add((source, target))
                incoming[target] += 1
                outgoing[source] += 1
    return edges


def evaluate(predicted: set[Edge], expected: set[Edge]) -> dict[str, float | int]:
    """Exact-coordinate edge and division-event Jaccard for a synthetic reference."""
    def division_events(edges):
        children = {}
        for a,b in edges:
            children.setdefault(a, set()).add(b)
        return {(a,frozenset(bs)) for a,bs in children.items() if len(bs) == 2}

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a or b else 1.0

    return {"true_edges":len(predicted & expected),"false_edges":len(predicted - expected),
            "missed_edges":len(expected - predicted),"edge_jaccard":jaccard(predicted,expected),
            "division_jaccard":jaccard(division_events(predicted),division_events(expected))}
