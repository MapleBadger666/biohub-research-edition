"""Run the synthetic tracking fixture or score user-supplied volumes locally."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .lineage import Node, evaluate, reconstruct
from .model import default_kernel, detect, load_kernel


def fixture():
    coordinates = (
        ((2, 3, 3), (7, 7, 7)),
        ((2, 2, 3), (2, 4, 3)),
        ((2, 2, 3), (2, 4, 3), (7, 7, 6)),
    )
    volumes = np.zeros((3, 10, 10, 10))
    frames = []
    for t, points in enumerate(coordinates):
        frames.append([Node(t, *p) for p in points])
        for point in points:
            volumes[(t, *point)] = 8.0
    a, b = frames[0]
    c, d = frames[1]
    e, f, g = frames[2]
    return volumes, {(a,c),(a,d),(c,e),(d,f),(b,g)}


def run(predictions_dir: Path, output_dir: Path, data: Path | None = None,
        weights: Path | None = None, *, spacing=(1.0, 1.0, 1.0),
        threshold=0.55, max_step=1.5, max_gap=1.5):
    """Use explicit I/O paths. Only the supplied fixture has reference labels."""
    if data is None:
        volumes, truth = fixture()
    else:
        volumes, truth = np.load(data, allow_pickle=False), None
    if not isinstance(volumes, np.ndarray) or volumes.ndim != 4:
        raise ValueError("data must be a NumPy array with time/z/y/x axes")
    kernel = default_kernel() if weights is None else load_kernel(weights)
    frames = [[Node(t, *p) for p in detect(volume, kernel, threshold)]
              for t, volume in enumerate(volumes)]
    edges = reconstruct(frames, spacing, max_step, max_gap)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"nodes":[asdict(node) for frame in frames for node in frame],
               "edges":[[asdict(a),asdict(b)] for a,b in sorted(edges)]}
    (predictions_dir/'lineage.json').write_text(json.dumps(payload, indent=2)+'\n')
    metrics = {"scope":"synthetic fixture" if truth is not None else "user-provided input",
               "nodes":sum(map(len,frames)),"edges":len(edges),
               "evaluation":evaluate(edges,truth) if truth is not None else None}
    (output_dir/'summary.json').write_text(json.dumps(metrics, indent=2)+'\n')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Optional .npy volume array')
    parser.add_argument('--weights', type=Path, help='Optional .npy convolution kernel')
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0,1.0,1.0), metavar=('Z','Y','X'))
    parser.add_argument('--threshold', type=float, default=0.55)
    parser.add_argument('--max-step', type=float, default=1.5)
    parser.add_argument('--max-gap', type=float, default=1.5)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
