# Biohub Public Research Edition

This edition presents cell detection in 3-D microscopy, temporal association, and lineage-graph evaluation. The runnable example uses synthetic volumes with a known division and a missing intermediate detection. It demonstrates the method with small, inspectable inputs.

## Try the synthetic example

Install this checkout in Python 3.11+ and choose explicit output directories. The CLI runs from any working directory.

```sh
python -m pip install .
python -m research_edition.demo --predictions-dir "./demo predictions" --output-dir "./demo reports"
```

The [synthetic notebook](notebooks/01_synthetic_lineage.ipynb) uses the installed package. The CLI also accepts an explicit `--data` NumPy array in time/z/y/x order and `--weights` NumPy kernel. Relative paths are resolved from the caller's working directory. Use absolute paths when invoking from another directory. Prediction files are written directly under `--predictions-dir`, so an existing consumer can point to a migration directory explicitly. No username is appended.

## Research record

The historical pipeline used an upstream temporal U-Net and transformer association model with sparse-label adaptations, physical-scale calibration and Frozen V9 graph repair. This edition preserves the [method and historical results](docs/research.md) and a reusable association-ranking module. Its small convolution, synthetic lineage post-processing and exact-coordinate metrics demonstrate the concepts independently.

**0.82308 was a historical Kaggle public leaderboard score.** See [the results record](docs/research.md#historical-results) for its scope and the separate local Fold0 evaluation. The synthetic demonstration uses generated volumes and evaluates them with a different metric.

The project-authored source is available under the [BSD 3-Clause license](LICENSE). The [upstream BSD 3-Clause notice](licenses/official-BSD-3-Clause.txt) retains its own copyright and terms.

See [attribution and input rights](THIRD_PARTY_NOTICES.md). Before installing or generating files in this checkout, verify the edition using `python -B verify_release.py` against the [Public Release Manifest](PUBLIC_RELEASE_MANIFEST.json). Obtain the manifest digest from a trusted release record before relying on it.
