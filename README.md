# CS781 Legal-LM Capstone

Code for the CS781 (IIT Kanpur) Legal-LM capstone by Archisman Dhar (240170, Codabench `a_240170`). Four small
language models are pretrained from scratch on the official Indian legal corpus, then adapted to six IL-TUR tasks
(Joshi et al., ACL 2024). No pretrained weights or external data are used anywhere.

The submitted code archive holds only the best pipeline of each phase (PS section 20); this repository keeps
every submission of every phase.

## Results

| Phase | Task | Final submission | Codabench score | Rank |
|---|---|---|---|---|
| [1](phase1/README.md) | Legal text continuation (pretraining) | Z13 | BERTScore-F1 0.8445 | 10 |
| [2](phase2/README.md) | Legal named entity recognition | N24 | strict macro-F1 0.6241 | 13 |
| [3](phase3/README.md) | Rhetorical role prediction | R30 | macro-F1 0.5751 | 14 |
| [4](phase4/README.md) | Court judgment prediction | C23 † | macro-F1 0.8332 | 6 |
| [5](phase5/README.md) | Legal statute identification | L29 | macro-F1 0.4288 | 19 |
| [6](phase6/README.md) | Abstractive summarization | S13 | BERTScore-F1 0.8468 | 33 |
| [7](phase7/README.md) | Prior case retrieval | P26 | micro-F1@7 0.4589 | 26 |

† The Phase-4 test cases also occur in the public dev set. C23 trained on dev folds, so its score includes label
leakage; C19 (0.7916) is the best system trained on train only. See [phase4/README.md](phase4/README.md).

## Models

| Model | Architecture | Parameters |
|---|---|---|
| LegalGPT | decoder, 20 x 768, RMSNorm, RoPE, grouped-query attention 12/4, SwiGLU | 129.7M |
| LegalSeq | encoder-decoder, 10 + 10 layers, prefix-to-continuation objective | 145.4M |
| LongLegal | decoder, 512-token sliding window, global attention every 4th layer, 4k context | 129.7M |
| LegalHybrid | order-2 Hyena long convolutions with attention every 4th layer | 130.7M |

All four use the Assignment-1 byte-level BPE (5,000 tokens, `assets/bpe_en.json`) and are trained on 2B tokens.
LegalGPT and LegalHybrid are then adapted to Supreme Court text (+0.4B tokens with 15% replay): S1, S2 and S3.
Phases 2-7 fine-tune these checkpoints with LoRMA+, LoRA, rsLoRA, QLoRA (NF4), AdaLoRA, IA3, Houlsby and Pfeiffer
adapters, BitFit, prompt tuning, partial fine-tuning, gradual unfreezing and full fine-tuning, and distil them into an
8-block, 54M student. Every adapter is implemented in `src/peft.py`; no PEFT library is used.

## Compute

Compute is given in T4 GPU-hours so that it can be compared with free Kaggle and Colab resources. One LegalGPT
training token costs 6N + 6LTd ≈ 0.97 GFLOPs (N = 129.7M, L = 20, T = 2,048, d = 768). At 25% of the T4's
65 TFLOPs fp16 peak this is 16.8k tokens/s, so a 2B-token pretraining run needs 33 T4-hours and a 0.4B-token
adaptation stage 6.6. On a T4 the runs use fp16 autocast with dynamic loss scaling and micro-batches of 4 sequences
with 64 accumulation steps (the same 0.5M-token step), which keeps the largest run at about 9 GB of the T4's 16 GB.

| Phase | Main jobs | Days | T4-hours | Kaggle quota per account | Accounts |
|---|---|---|---|---|---|
| 1 | pretraining 129.2, adaptation 30.9, ablations 41.2, self-ranking and decoding 24.1 | 14 | 225.5 | 120.0 | 2 |
| 2 | PEFT members, 2-fold OOF bag | 5 | 11.4 | 42.9 | 1 |
| 3 | context models, frozen features, cross-fitting | 5 | 31.4 | 42.9 | 1 |
| 4 | window and hierarchical models, dev cross-fitting | 5 | 120.2 | 42.9 | 3 |
| 5 | LWAN members, chunking, dev cross-fitting | 5 | 59.9 | 42.9 | 2 |
| 6 | seven summarizers, MBR, fill | 7 | 15.2 | 60.0 | 1 |
| 7 | retrievers, rankers, cross-encoders | 5 | 36.6 | 42.9 | 1 |
| | **Total** | 46 | **500.2** | | |

T4-hours cover all submissions of a phase. Quota: Kaggle gives 30 session-hours per week with two T4s, pro rata
over the phase window (PS dates). Phase-2 inference measured on a Kaggle T4 (fp16): 0.6-2.4 GB peak memory and
10.2k tokens/s end to end for the LegalGPT members.

## Layout

```
src/            all Python code (shared by every phase)
  models/       the four architectures and the starter baseline
  peft.py       adaptation methods
  config.py     YAML configs with inheritance (`base:`) and key=value overrides
  ...           one module per task: ner.py, rr.py, cj.py, lsi.py, sm.py, pcr*.py
phaseN/
  README.md     method, results and every configuration of the phase
  setup.sh      data download, preprocessing and checks
  run.sh        every submission of the phase in order, ending with the final one
  configs/      one YAML file per submission or ensemble member
scripts/        env.sh (Python environment), common.sh (shell helpers)
assets/         Assignment-1 tokenizer
tests/          pytest wrapper around the CPU self-checks in src/check.py
```

## Setup

Requirements: Linux, Python >= 3.10 and one CUDA GPU (a 16 GB T4 with the settings in [Compute](#compute)).

```bash
bash scripts/env.sh                 # .venv, torch for the local CUDA, requirements, evaluation scripts, self-checks
bash phase1/setup.sh                # Phase-1 data, split, token cache, dev set
CUDA_VISIBLE_DEVICES=0 bash phase1/run.sh
bash phase2/setup.sh && bash phase2/run.sh     # and so on up to phase 7
```

Phases 2-7 start from the Phase-1 checkpoints in `runs/`, so Phase 1 must run first. Each `run.sh` is
restartable: finished submissions are skipped and interrupted pretraining resumes from `last.pt`. `DRYRUN=1` prints
the command sequence without running it. The official evaluation scripts are fetched into `evaluation_script/`
from the organisers' folder (PS section 18.1).

Model weights are not included: the competition rules forbid sharing them, and every checkpoint is rebuilt by the
phase scripts. Scores can differ slightly from the table above because of GPU nondeterminism.

## Rules followed

- Phase-1 models are trained from random initialisation, within 50M-300M parameters (`src/train.py` refuses
  anything else), on the four official training sources only.
- Phases 2-7 use only our Phase-1 checkpoints and the official data of each phase.
- `roberta-large` is downloaded only to run the official Phase-6 BERTScore on held-out summaries for selection
  (S18-S20); it is never part of a submitted model. BM25 and TF-IDF in Phase 7 are classical scorers with no learned
  weights.
- Some systems train on development labels (Phase 3 R10 and R17-R31, Phase 4 C20-C27, Phase 5 L21-L29, Phase 7
  rankers from P12 on, cross-fitted over dev queries). Phase 7 hubness correction and query-graph propagation use
  the unlabelled test queries together. Both are disclosed in the report.

## Main references

- Joshi et al. *IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning.* ACL 2024.
- Bihany, Patel, Modi. *LoRMA: Low-Rank Multiplicative Adaptation for LLMs.* Findings of ACL 2025.
- Hu et al. *LoRA.* ICLR 2022. Dettmers et al. *QLoRA.* NeurIPS 2023. Zhang et al. *AdaLoRA.* ICLR 2023.
- Rafailov et al. *Direct Preference Optimization.* NeurIPS 2023. Freitag et al. *High Quality Rather than High Model Probability: MBR Decoding.* TACL 2022.
- Poli et al. *Hyena Hierarchy.* ICML 2023. Beltagy et al. *Longformer.* 2020.

Each phase README lists the citations for its methods.
