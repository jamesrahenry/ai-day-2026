# Are Language Models Converging on Shared Structures?

*Poster, Brock AI Day 2026 · James R. A. Henry, Computer Science (Undergraduate), Brock University*
*jh13nt@brocku.ca · jamesrahenry@henrynet.ca*

*Written: 2026-10-30 23:40 UTC*

An A1 poster introducing mechanistic interpretability for a general AI audience: the Linear
Representation Hypothesis (concepts as vectors in a model's activation space), how a concept
vector is extracted from contrastive text pairs, the Platonic Representation Hypothesis
(independently trained models converge on a shared representation), and a test of it across
35 open language models from 10 families.

## The test in one paragraph

Each model's concept vectors live in that model's own coordinate system, so two models' vectors
cannot be compared directly. The angles *inside* a model can be: how similar is credibility to
formality, to sarcasm, to negation? Represent every concept in every model by its angles to the
other sixteen (relative representations, Moschella et al. 2023), then hold out an entire model
family and try to name each of its concepts from those angles alone. At mid-depth 97% are named
correctly; chance is 6%, and shuffling the concept labels or replacing the vectors with random
ones gives 6%.

## Contents

- `poster/Main.dc.html`, `poster/canvas.json` — the poster source as published. Image `src`
  values are artifact asset ids (`/_blob/<id>`); `render/render_poster.py` maps them to files.
- `figgen/` — figure scripts. All read `~/rosetta_data/paper_n250/<model>/caz_<concept>.json`
  from the [Rosetta-Activations](https://huggingface.co/datasets/james-ra-henry/Rosetta-Activations) dataset.
  - `sep_curves.py` — per-layer separation of the contrastive pairs, Llama-3.1-8B vs Mistral-7B (section 2)
  - `ribbon_fig_null.py <anchor>` — anchor-relative depth ribbons across 35 models, shuffled-label
    null panel, leave-one-family-out identification curve (section 4; anchor = `credibility`)
  - `relative_coords.py`, `relative_coords_lofo.py`, `relative_by_depth.py`, `anchor_choice.py` — the numbers
  - `procrustes_fast.py` — why the poster does *not* use in-sample Procrustes alignment: with 17 vectors
    in 4,096 dimensions it scores 0.98 for the true pairing, 0.88 for shuffled labels and 0.93 for random
    vectors, so it cannot tell structure from noise
  - `rsa_null_fig.py`, `before_after_pair.py`, `anchor_profile.py` — earlier candidate figures
- `render/render_poster.py [scale]` — Playwright screenshot of the poster at A1 pixel size with per-section heights.

## Data and tooling

- Activations and concept vectors: [james-ra-henry/Rosetta-Activations](https://huggingface.co/datasets/james-ra-henry/Rosetta-Activations) (MIT)
- Contrastive pairs: [jamesrahenry/Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)
- Extraction and analysis tooling: [jamesrahenry/Rosetta_Tools](https://github.com/jamesrahenry/Rosetta_Tools)

## This work

- Henry, J. R. A. (2026a). Concept Allocation Zones. [arXiv:2605.24856](https://arxiv.org/abs/2605.24856)
- Henry, J. R. A. (2026b). Geometric Evolution Maps. [arXiv:2605.25848](https://arxiv.org/abs/2605.25848)
- Henry, J. R. A. (2026c). CAZ Validation. [Zenodo 10.5281/zenodo.22016917](https://doi.org/10.5281/zenodo.22016917)

## Further reading

- Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *ICML*.
- Park, K., Choe, Y. J., & Veitch, V. (2024). The Linear Representation Hypothesis and the Geometry of Large Language Models. *ICML*.
- Elhage, N., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread*.
- Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.
- Rimsky, N., et al. (2024). Steering Llama 2 via Contrastive Activation Addition. *ACL*.
- Moschella, L., et al. (2023). Relative Representations Enable Zero-Shot Latent Space Communication. *ICLR*.
- Olah, C., et al. (2020). Zoom In: An Introduction to Circuits. *Distill*.
- Mikolov, T., Le, Q. V., & Sutskever, I. (2013). Exploiting Similarities among Languages for Machine Translation. arXiv:1309.4168.
- Harris, Z. S. (1954). Distributional Structure. *Word*. · Firth, J. R. (1957). A Synopsis of Linguistic Theory.
- Piantadosi, S. T., & Hill, F. (2022). Meaning without reference in large language models. arXiv:2208.02957.
- Bender, E. M., & Koller, A. (2020). Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data. *ACL*.
- Saussure, F. de (1916). *Cours de linguistique générale*.

## Licence

Code MIT. Poster text and figures CC BY 4.0.
