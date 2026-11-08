# Phase 4: Court judgment prediction (CJPE)

Predict the binary outcome of a judgment. Fine-tuned models read the final 2,048 tokens (the end of a judgment carries the decision), with dev-tuned thresholds for the class-prior shift, test-time augmentation over end windows, hierarchical chunk models and dev cross-fitting.

| | |
|---|---|
| Final submission | **C23** (`phase4/configs/c23_gpt_full_dcv.yaml`) |
| Codabench | macro-F1 0.8332 (C23, see the leakage note) / 0.7916 (C19, train only); rank 6 (27 valid submissions) |
| Compute | 120.2 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/cj.py`, `src/peft.py` |
| Needs | Phase-1 checkpoints S1, S3, L1, Q1 |

## Running

```bash
bash phase4/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase4/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase4/run.sh                           # print the command sequence only
bash phase4/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** predict 0/1 from the case text. **Metric:** macro-F1 over both classes.

**Rules:** as in Phase 3, plus BitFit, partial fine-tuning, layer freezing/unfreezing and adapters in the
technique list. Code: `src/cj.py`, `phase4/configs/`, `phase4/run.sh`.

### 4.1 Long-document representation
| component | citation | where |
|---|---|---|
| **Window of the judgment's final tokens** (2048; 4096 for LongLegal, 1024 for the LegalSeq encoder), unmasked encoder, attention + mean pooling, MLP head | Malik et al., *ILDC for CJPE*, ACL 2021 (last part of the judgment most predictive); Dukić & Šnajder 2024 | `CJModel`, `take` |
| **Hierarchical whole-document model**: frozen chunk embeddings (512 tokens from the end, ≤ 64 chunks) of all four architectures → relative-position embedding → BiGRU → attention pooling | Malik et al. 2021 (hierarchical XLNet/BiGRU); Yang et al., *Hierarchical Attention Networks*, NAACL 2016; Chalkidis et al. 2019 | `feats`, `HierModel` (C8) |
| **Section analysis**: attention mass per document decile; first-half vs last-half chunk models (C12, C13); first vs last 2048 tokens (C14 vs C1) | PS §10.5 | `hier`, configs |
| Class weights (n / 2n_c)^0.5; **decision threshold tuned on dev** for macro-F1 | Lipton et al. 2014 | `bestthr` |

### 4.2 Adaptation techniques
| id | Phase-1 model | technique | citation |
|---|---|---|---|
| C1 | LegalGPT-SC (S1) | **LoRMA+** (compulsory) | Bihany, Patel, Modi 2025 |
| C2 | LegalHybrid-SC (S3) | **Houlsby adapters** (64) | Houlsby et al., ICML 2019 |
| C3 | LongLegal (L1) | **AdaLoRA**, 4096-token window | Zhang et al., ICLR 2023 |
| C4 | LegalSeq encoder (Q1) | **BitFit**: zero-initialised biases on every projection (the Phase-1 linears are bias-free) + RMSNorm gains | Ben Zaken, Goldberg, Ravfogel, ACL 2022 |
| C5 | S1 | full fine-tuning | — |
| C6 | S1 | **gradual unfreezing + discriminative learning rates** (top-down, all blocks open by 50% of training, lr·0.85^depth) | Howard & Ruder, *ULMFiT*, ACL 2018 |
| C7 | S1 | **partial / layer-wise fine-tuning** (top 8 of 20 blocks) | Lee et al. 2019 |
| C8 | all four, frozen | hierarchical chunk model (feature-based transfer) | as 4.1 |
| C10 | student = first 8 of 20 S1 blocks (≈54M) | **knowledge distillation** from the C9 ensemble: α·CE + (1−α)·T²·KL, T = 2 | Hinton et al. 2015; Sanh et al. 2019 |

### 4.3 Ensembles
| id | method |
|---|---|
| C9 | weighted mean of C1–C8 P(1), weights and threshold tuned on dev (also the KD teacher) |
| C11 | C9 + C10 |

### 4.4 Results
| id | system | dev F1 | test F1 |
|---|---|---|---|
| C1 | S1 + LoRMA+, last 2048 tokens | 0.7577 (threshold 0.15; 0.6979 at 0.5) | — |
| C2 / C5 / C6 / C7 | S3 adapters / S1 full / S1 unfreeze / S1 partial (last 2048) | 0.7543 / 0.7371 / 0.7236 / 0.7324 | — |
| C3 / C4 | LongLegal AdaLoRA 4096 / LegalSeq BitFit 1024 | 0.6545 / 0.6261 | — |
| C8 | hierarchical frozen chunks (whole document) | 0.6553 | — |
| C10 | KD student (8 layers, ≈54M) | 0.7104 | — |
| C12 / C13 | hierarchical, first half / last half of the document | 0.6034 / 0.6490 | — |
| C14 | LoRMA+ on the FIRST 2048 tokens | 0.3388 (collapsed: predicts class 1) | — |
| C9 / C11 | ensembles (C11 weights C1 3, C2 2, C7 2, C5 1, C8 1, C10 0.5) | 0.7797 / 0.7852 | — / **0.7785** |

**Section evidence:**
- The end of the judgment carries the signal: last half 0.649 > first half 0.603 (C13 vs C12).
- The first 2048 tokens alone fail (C14).
- C8's attention puts its largest mass (0.24) on the final decile.

### 4.5 Round 2: end-to-end hierarchical fine-tuning and test-time augmentation
| id | method | citation |
|---|---|---|
| C15 / C16 | **End-to-end hierarchical fine-tuning**: the last 3 × 1536 tokens (evenly overlapping on shorter judgments) through the shared fine-tuned encoder (S1 + LoRMA+ / S3 + adapters); window vectors + window-position embedding → transformer layer across windows → attention pooling | Dai, Chalkidis, Darkner, Elliott, *Revisiting Transformer-based Models for Long Document Classification*, Findings EMNLP 2022; Pappagari et al., *Hierarchical Transformers for Long Document Classification*, ASRU 2019 |
| C17 / C18 | **Multi-window test-time augmentation** of C1 / C2: mean P(1) over the final window and windows ending 1024 / 2048 tokens earlier | test-time augmentation (Shanmugam et al., ICCV 2021) |
| C19 | dev-weighted ensemble of all strong members | — |

**Round 2 results:**

| id | dev F1 | test F1 | note |
|---|---|---|---|
| C17 | 0.7561 | — | TTA does not help C1 |
| C18 | 0.7646 | — | TTA on C2: +0.010 |
| C15 | 0.334 | — | hierarchical fine-tuning collapsed (constant P(1)) |
| C16 | 0.473 | — | hierarchical fine-tuning collapsed |
| C19 | 0.7894 | **0.7916** | best so far |

### 4.6 Round 3: dev cross-fitting + window augmentation
The dev and test sets match: both are balanced, with median 4.2k / 4.3k tokens. Train does not (59/41, median 3.3k,
31% short docs), so dev is the only in-distribution labelled data.

| id | method | citation |
|---|---|---|
| C20–C23 | **Dev cross-fitting**: 2 dev folds; each run trains on train + one dev half and selects on the other; every dev case is scored by the run that held it out (honest threshold / weights on all 3.5k dev cases); test = mean of the two fold models. Setups: S3 adapters (C20), S1 LoRMA+ (C21), S1 partial (C22), S1 full (C23) | cross-fitting (Chernozhukov et al. 2018); covariate-shift adaptation with in-distribution data |
| C20–C23 | **Window augmentation**: each epoch every document contributes one of the windows ending 0 / 1024 / 2048 tokens before the end, with TTA over the same windows | data augmentation for long documents; Shanmugam et al. 2021 |
| C24 | dev-weighted ensemble of C20–C23 + the strong earlier members | — |

**Fold-0 results (held-out dev half, 1,772 cases):**

| run | held-out dev F1 | threshold | baseline |
|---|---|---|---|
| C20F0 (S3 adapters + dev + window augmentation) | **0.7900** | 0.51 | C2 0.7543 / C18 0.7646 |
| C21F0 (S1 LoRMA+ + dev + window augmentation) | **0.7738** | 0.46 | C1 0.7577 |

- Adding the in-distribution dev half lifts single models by +1.6 to +2.5 points.
- The tuned threshold moves to about 0.5: the balanced dev data removes the class-prior shift.

**Joined round-3 results (honest dev F1 over all 3,505 dev cases):**

| id | setup | dev F1 | test F1 |
|---|---|---|---|
| C20 | S3 adapters | 0.7821 | — |
| C21 | S1 LoRMA+ | 0.7518 | — |
| C22 | S1 partial | 0.7446 | — |
| C23 | S1 full | 0.7854 | **0.8332** |
| C24 | ensemble | 0.8116 | — |

- Full fine-tuning gains the most from in-distribution data (+4.8 over C5); LoRMA+ gains nothing.
- The test score exceeds the dev score because the test average uses fold models that together saw all of dev.

### 4.7 Round 4: more in-distribution data + covariate-shift correction
| id | method | citation |
|---|---|---|
| C25 / C26 | Full fine-tuning of S1 / S3 with **4 dev folds** (train + 75% of dev per run, test = mean of 4 models), **dev oversampled ×3**, **length-bucket importance weights** p_dev(b)/p_train(b) on the training documents (b = ⌊log₂ tokens⌋, clipped to [0.25, 4]), window augmentation + TTA | Shimodaira 2000 (importance weighting under covariate shift); Sugiyama et al., JMLR 2007; cross-fitting |
| C27 | dev-weighted ensemble of all dev-cross-fitted members (C20–C23, C25, C26) | — |

**Data notes (from `prep`):**
- train: 37,387 docs, 59/41 (0/1); dev: 3,505 docs, balanced; test: 1,517 docs, balanced.
- Median tokens/doc 3.3k–4.3k; 31% of train but only 13% of dev/test fit in 2048 tokens.
- Train and dev contain repeated case IDs (scored as 32,305 / 2,511 unique). Test IDs are unique.
- The class-prior shift (train 41% positive vs dev/test 50%) is why the dev-tuned threshold sits far below 0.5.

### 4.8 Data issue: the test set is a subset of dev
The test cases also occur in dev, with labels. **Any model trained on dev has therefore seen test labels.**
- C20–C27 trained on dev folds, so their test scores (e.g. C23 0.8332 vs honest dev 0.7854) are inflated by label
  leakage. C19 (0.7916), trained on train only, is the clean result.
- C9, C11 and C19 were trained on train only. Their member weights and thresholds were tuned on a dev set that
  contained test cases (milder leakage).
- `cj.loadtok` can remove every train / dev case whose id or exact token content occurs in test before training,
  cross-fitting, threshold tuning and ensemble weighting (`cj.notest: true`). `phase4/configs/base.yaml` keeps it off
  so that C1–C27 are reproduced as submitted.

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | Phase 4 (CJPE, binary) defaults. Backbones are the trained Phase-1 checkpoints; nothing external is loaded. |
| `c1_gpt_lorma.yaml` | C1 | submitted | LegalGPT-SC (S1) + LoRMA+ (compulsory; Bihany, Patel, Modi 2025), last 2048 tokens, unmasked, attention + mean pooling, MLP head; dev-tuned decision threshold. |
| `c2_hyb_adapter.yaml` | C2 | submitted | LegalHybrid-SC (S3) + Houlsby bottleneck adapters (64) after every mixer / attention / FFN, last 2048 tokens. |
| `c3_long_adalora.yaml` | C3 | submitted | LongLegal (L1) + AdaLoRA (rank 12 -> average 4 by importance), last 4096 tokens (window-512 local attention + global layers make the longer context affordable); gradient checkpointing. |
| `c4_seq_bitfit.yaml` | C4 | submitted | LegalSeq (Q1) encoder + BitFit (Ben Zaken et al. 2022): zero-initialised biases on every projection + RMSNorm gains are the only backbone parameters trained; last 1024 tokens (the encoder was pretrained on 1280-token prefixes). |
| `c5_gpt_full.yaml` | C5 | submitted | LegalGPT-SC (S1) full fine-tuning, last 2048 tokens (reference for PEFT vs full adaptation). |
| `c6_gpt_unfreeze.yaml` | C6 | submitted | LegalGPT-SC (S1) with gradual unfreezing + discriminative learning rates (ULMFiT, Howard & Ruder 2018): all blocks frozen at the start, the top ceil(f * 20) opened as training proceeds (all open at 50%), lr * 0.85^depth-from-top. |
| `c7_gpt_partial.yaml` | C7 | submitted | LegalGPT-SC (S1) partial / layer-wise fine-tuning: only the top 8 of 20 blocks (+ final norm) are trained. |
| `c8_hier.yaml` | C8 | submitted | hierarchical model over the WHOLE judgment: frozen chunk embeddings (512-token chunks from the end, up to 64) of all four Phase-1 architectures -> relative-position embedding -> BiGRU -> attention pooling -> MLP (ILDC hierarchical models, Malik et al. 2021; HAN, Yang et al. 2016); bag of 3 seeds; attention mass per document decile is reported. |
| `c9_ens.yaml` | C9 | submitted | weighted ensemble of C1-C8 (member weights and decision threshold tuned on dev); also the distillation teacher. |
| `c10_kd.yaml` | C10 | submitted | knowledge distillation (Hinton et al. 2015): teacher = the C9 ensemble (eight Phase-1-based models), student = the first 8 of 20 LegalGPT-SC blocks (~54M), last 2048 tokens, L = alpha * CE + (1 - alpha) * T^2 * KL(teacher // student). |
| `c11_ens_final.yaml` | C11 | submitted | final weighted ensemble of C1-C8 + the C10 student, weights and threshold tuned on dev. |
| `c12_hier_first.yaml` | C12 | submitted | C8 restricted to the FIRST half of each judgment's chunks. |
| `c13_hier_last.yaml` | C13 | submitted | C8 restricted to the LAST half of each judgment's chunks. |
| `c14_gpt_lorma_first.yaml` | C14 | submitted | C1 reading the FIRST 2048 tokens instead of the last. |
| `c15_gpt_lorma_hier.yaml` | C15 | submitted | end-to-end hierarchical fine-tuning (Dai et al., Findings EMNLP 2022; Pappagari et al. 2019): LegalGPT-SC + LoRMA+; the last 3 windows of 1536 tokens (4608 tokens, evenly overlapping on shorter judgments) share the fine-tuned encoder; window vectors + window-position embedding (from the end) -> transformer layer across windows -> attention pooling. |
| `c16_hyb_adapter_hier.yaml` | C16 | submitted | C15's end-to-end hierarchical model on LegalHybrid-SC + Houlsby adapters (C2's setup). |
| `c17_gpt_lorma_tta.yaml` | C17 | submitted | C1's trained model scored with multi-window test-time augmentation: mean P(1) over the final 2048-token window and the windows ending 1024 / 2048 tokens earlier; dev-tuned threshold. No training. |
| `c18_hyb_adapter_tta.yaml` | C18 | submitted | C2's trained model with the same multi-window test-time augmentation as C17. |
| `c19_ens_all.yaml` | C19 | submitted | dev-weighted ensemble of every strong member (C11's members + hierarchical C15 / C16 + TTA C17 / C18). |
| `c20_hyb_adapter_dcv.yaml` | C20 | submitted | C2 (LegalHybrid-SC + adapters), dev cross-fitted. Two dev-fold runs (train + one half of dev each, window augmentation over the windows ending 0 / 1024 / 2048 tokens before the end, TTA over the same windows); each dev case scored by the run that held it out, test = mean. |
| `c21_gpt_lorma_dcv.yaml` | C21 | submitted | C1 (LegalGPT-SC + LoRMA+), dev cross-fitted. Two dev-fold runs (train + one half of dev each, window augmentation over the windows ending 0 / 1024 / 2048 tokens before the end, TTA over the same windows); each dev case scored by the run that held it out, test = mean. |
| `c22_gpt_partial_dcv.yaml` | C22 | submitted | C7 (LegalGPT-SC, top-8-block partial fine-tuning), dev cross-fitted. Two dev-fold runs (train + one half of dev each, window augmentation over the windows ending 0 / 1024 / 2048 tokens before the end, TTA over the same windows); each dev case scored by the run that held it out, test = mean. |
| `c23_gpt_full_dcv.yaml` | C23 | submitted | C5 (LegalGPT-SC full fine-tuning), dev cross-fitted. Two dev-fold runs (train + one half of dev each, window augmentation over the windows ending 0 / 1024 / 2048 tokens before the end, TTA over the same windows); each dev case scored by the run that held it out, test = mean. |
| `c24_ens_dcv.yaml` | C24 | submitted | dev-weighted ensemble of the dev-cross-fitted members (C20-C23, honest out-of-fold on all of dev) and the strong earlier members (C1, C2, C5, C7, C8, C17, C18); member weights and threshold tuned on all 3.5k dev cases. |
| `c25_gpt_full_dcv4.yaml` | C25 | submitted | LegalGPT-SC full fine-tuning (C23's recipe) with 4 dev folds (train + 75% of dev per run, test = mean of 4 models), in-distribution dev docs oversampled x3 and training docs reweighted by p_dev / p_train of their length bucket (covariate-shift importance weighting, Shimodaira 2000), window augmentation + TTA over windows ending 0/1024/2048 tokens before the end. Each dev case is scored by the fold model that held it out. |
| `c26_hyb_full.yaml` | C26B | member | Base for C26's fold runs: LegalHybrid-SC (S3), full fine-tuning, last 2048 tokens. |
| `c26_hyb_full_dcv4.yaml` | C26 | submitted | C25's recipe with LegalHybrid-SC (S3) fully fine-tuned (S3 adapted best among the PEFT runs, C20). |
| `c27_ens_dcv4.yaml` | C27 | submitted | dev-weighted ensemble of the dev-cross-fitted members (C25, C26, C23, C20, C21, C22; each honest out-of-fold on all 3.5k dev cases); member weights and threshold tuned on dev. |
