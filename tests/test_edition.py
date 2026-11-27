import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from research_edition.demo import fixture, run
from research_edition.lineage import Node, reconstruct
from research_edition.model import default_kernel, response, sparse_focal_loss

ROOT = Path(__file__).resolve().parents[1]


def test_model_and_numerical_regression():
    frames, _ = fixture()
    assert response(frames[0], default_kernel()).shape == frames.shape[1:]
    logits = np.zeros((2, 3))
    target = np.array([[1, 1, 0], [0, 0, 0]])
    assert sparse_focal_loss(logits, target) == pytest.approx(np.log(2) / 4)
    assert sparse_focal_loss(logits, target, 2.0) == pytest.approx(np.log(2) * 7 / 20)
    with pytest.raises(ValueError, match="no annotated"):
        sparse_focal_loss(logits, np.zeros_like(target))


def test_graph_rejects_ambiguous_gap_and_merge():
    source, other = Node(0, 0, 0, 0), Node(0, 0, 2, 0)
    target = Node(2, 0, 1, 0)
    assert reconstruct([[source, other], [], [target]], (1, 1, 1), 2, 2) == set()
    single = Node(1, 0, 0, 0)
    assert reconstruct([[source, other], [single]], (1, 1, 1), 2, 2) == {(source,single)}


def test_cli_portable_and_notebook_initialization(tmp_path):
    destination = tmp_path / "another working directory"
    destination.mkdir()
    env = os.environ | {"PYTHONPATH":str(ROOT), "PYTHONDONTWRITEBYTECODE":"1"}
    subprocess.run([sys.executable, "-B", "-m", "research_edition.demo",
                    "--predictions-dir","custom predictions","--output-dir","custom reports"],
                   cwd=destination, env=env, check=True)
    report = json.loads((destination/"custom reports/summary.json").read_text())
    assert report["evaluation"] == {"true_edges":5,"false_edges":0,"missed_edges":0,
                                     "edge_jaccard":1.0,"division_jaccard":1.0}
    notebook = json.loads((ROOT/"notebooks/01_synthetic_lineage.ipynb").read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="code")
    subprocess.run([sys.executable,"-B","-c",code], cwd=destination, env=env, check=True)


def test_explicit_data_weights_and_no_synthetic_evaluation(tmp_path):
    data, weights = tmp_path/"my data.npy", tmp_path/"my weights.npy"
    np.save(data, fixture()[0])
    np.save(weights, default_kernel())
    result = run(tmp_path/"p",tmp_path/"o",data,weights)
    assert result["edges"] == 5
    assert result["evaluation"] is None


def test_release_identity_rejects_addition_mutation_and_wrong_pin(tmp_path):
    import hashlib
    import shutil

    from verify_release import verify

    clean = tmp_path/'clean release'
    shutil.copytree(ROOT,clean)
    manifest = clean/'PUBLIC_RELEASE_MANIFEST.json'
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify(clean,expected)==expected
    with pytest.raises(ValueError,match='digest differs'):
        verify(clean,'0'*64)
    extra = clean/'extra.md'
    extra.write_text('surprise')
    with pytest.raises(ValueError,match='inventory'):
        verify(clean,expected)
    extra.unlink()
    report = clean/'docs/research.md'
    report.write_text('changed')
    with pytest.raises(ValueError,match='inventory'):
        verify(clean,expected)
    report.unlink()
    with pytest.raises(ValueError,match='inventory'):
        verify(clean,expected)
