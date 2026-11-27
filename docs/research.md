# Model, reconstruction and evaluation

The historical tracking pipeline predicts detections in 3-D microscopy and associates adjacent frames. The historical Warm2 model configuration records 32 output channels, U-Net layers 32/64/128, downsampling [1, 4, 4] and a two-frame temporal window. The recorded architecture config has a 5 µm pooling kernel. The frozen inference workflow overrides this to 3 µm.

Association replay normalizes logits across parents for each target. The public edition retains the NumPy candidate-ranking module and implements a small portable demonstration of degree-constrained links, one-frame gap recovery and division-event comparison. The frozen historical graph policies additionally gate divisions using daughter separation, angle and distance balance, prune low-confidence small components, and use mutually selected repairs and conflict checks. The synthetic implementation demonstrates the rules using anisotropic distances and a deliberately small convolution model.

The historical sparse association loss uses an annotated row/column mask and focal-weighted binary cross-entropy. The edition includes a numerically stable NumPy example of that loss. It exposes the division-positive weight for inspecting the difference from unit weights.

## Historical results

| Evaluation | Result | Scope |
| --- | ---: | --- |
| Kaggle public leaderboard | **0.82308** | Recorded public leaderboard score. The archived submission record displays 0.823 to three decimals. The public row cannot verify an exact notebook revision. |
| Local Fold0 baseline | **0.770921082** | Historical evaluation across 40 datasets under the competition metric. |
| Local Fold0 division Jaccard | **0.038461538** | The historical result identifies divisions as a remaining weakness. |
| Synthetic fixture | Edge Jaccard 1.0, division Jaccard 1.0 | Five expected edges and one division in a deterministic generated example. The metric compares exact coordinates. |

The historical public and local scores use different evaluations and do not identify incremental improvement. The synthetic scores check code behavior on a known graph and provide no estimate of competition performance.

The historical split used whole datasets and balanced cohort, density and division information. The official adjusted edge Jaccard handles graph errors differently from the exact matches in the synthetic demo. Local performance does not establish hidden-test generalization.

## Scope of the portable edition

The original run records and byte-level contracts are historical evidence. The Public Release Manifest identifies this edition's independent file set. D1's legacy training source matches the historical expected hash while the published training source adds a division-positive weight. The public edition tests its own loss interface. D2's earlier portable-source bytes could not be located in the authorized scope. The historical assertions retain their original meaning.

The included notebook replaces the inaccessible result link with a local example. Real competition inputs require separate rights and staging. External data and weights use an edition-specific NumPy interface. The original checkpoint and GEFF formats need an explicit conversion outside this edition.
