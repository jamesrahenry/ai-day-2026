# Are Language Models Converging on Shared Structures?

*Poster, Brock AI Day 2026 · James R. A. Henry, Computer Science (Undergraduate), Brock University*
*jh13nt@brocku.ca · jamesrahenry@henrynet.ca*

*Written: 2026-10-30 23:40 UTC · Updated: 2026-10-31 00:05 UTC*

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
  - `make_qr.py` — the footer QR code
- `figures/` — the rendered figures used on the poster (`fig_sep_curves.png`, `fig_ribbon_credibility_null.png`) and the QR code
- `render/render_poster.py [scale]` — Playwright screenshot of the poster at A1 pixel size with per-section heights; `render/poster_preview_half.png` is the latest half-scale preview.

The figure scripts import `rosetta_tools.viz_style` for the shared concept palette; clone [Rosetta_Tools](https://github.com/jamesrahenry/Rosetta_Tools) beside this repo or under `~/rosetta_tools`. Python deps: numpy, scipy, matplotlib, playwright (render only), qrcode (QR only).

## Models

35 base models, 10 families. `n_layers` counts decoder blocks; the extractor also keeps the embedding output as layer 0.

| Family | Model (Hugging Face id) | Layers | Hidden dim |
|---|---|---|---|
| Falcon | `tiiuae/falcon-40b` | 60 | 8192 |
| GPT-2 | `openai-community/gpt2` | 12 | 768 |
| GPT-2 | `openai-community/gpt2-medium` | 24 | 1024 |
| GPT-2 | `openai-community/gpt2-large` | 36 | 1280 |
| GPT-2 | `openai-community/gpt2-xl` | 48 | 1600 |
| GPT-Neo | `EleutherAI/gpt-neo-125m` | 12 | 768 |
| Gemma | `google/gemma-2-2b` | 26 | 2304 |
| Gemma | `google/gemma-4-26B-A4B` | 30 | 2816 |
| Gemma | `google/gemma-2-9b` | 42 | 3584 |
| Llama | `meta-llama/Llama-3.2-1B` | 16 | 2048 |
| Llama | `meta-llama/Llama-3.2-3B` | 28 | 3072 |
| Llama | `meta-llama/Llama-3.1-8B` | 32 | 4096 |
| Llama | `meta-llama/Llama-3.1-70B` | 80 | 8192 |
| Mistral | `mistralai/Mistral-7B-v0.3` | 32 | 4096 |
| OPT | `facebook/opt-125m` | 12 | 768 |
| OPT | `facebook/opt-350m` | 24 | 1024 |
| OPT | `facebook/opt-1.3b` | 24 | 2048 |
| OPT | `facebook/opt-2.7b` | 32 | 2560 |
| OPT | `facebook/opt-6.7b` | 32 | 4096 |
| Phi | `microsoft/phi-2` | 32 | 2560 |
| Pythia | `EleutherAI/pythia-70m` | 6 | 512 |
| Pythia | `EleutherAI/pythia-160m` | 12 | 768 |
| Pythia | `EleutherAI/pythia-410m` | 24 | 1024 |
| Pythia | `EleutherAI/pythia-1.4b` | 24 | 2048 |
| Pythia | `EleutherAI/pythia-1b` | 16 | 2048 |
| Pythia | `EleutherAI/pythia-2.8b` | 32 | 2560 |
| Pythia | `EleutherAI/pythia-6.9b` | 32 | 4096 |
| Pythia | `EleutherAI/pythia-12b` | 36 | 5120 |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B` | 24 | 896 |
| Qwen2.5 | `Qwen/Qwen2.5-1.5B` | 28 | 1536 |
| Qwen2.5 | `Qwen/Qwen2.5-3B` | 36 | 2048 |
| Qwen2.5 | `Qwen/Qwen2.5-7B` | 28 | 3584 |
| Qwen2.5 | `Qwen/Qwen2.5-14B` | 48 | 5120 |
| Qwen2.5 | `Qwen/Qwen2.5-32B` | 64 | 5120 |
| Qwen2.5 | `Qwen/Qwen2.5-72B` | 80 | 8192 |

Instruction-tuned variants were excluded. Concept vectors are difference-of-means directions at the last token of each passage, one per layer, from 250 contrastive pairs per concept (`paper_n250` tree of the dataset below).

## Data and tooling

- Activations and concept vectors: [james-ra-henry/Rosetta-Activations](https://huggingface.co/datasets/james-ra-henry/Rosetta-Activations) (MIT)
- Contrastive pairs (RCP, Rosetta Concept Pairs): [jamesrahenry/Rosetta_Concept_Pairs](https://github.com/jamesrahenry/Rosetta_Concept_Pairs)
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
