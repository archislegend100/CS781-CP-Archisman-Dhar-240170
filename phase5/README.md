# Phase 5: Legal statute identification (LSI)

Predict the set of applicable statutes (100 labels) for each case. A label-wise attention head with asymmetric loss and per-statute thresholds sits on each fine-tuned Phase-1 encoder; chunk aggregation, dev cross-fitting, a distribution-balanced loss and multiple-instance chunk training feed the final weighted ensemble.

| | |
|---|---|
| Final submission | **L29** (`phase5/configs/l29_ens_final.yaml`) |
| Codabench | macro-F1 0.4288; rank 19 (30 valid submissions) |
| Compute | 59.9 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/lsi.py`, `src/cj.py`, `src/peft.py` |
| Needs | Phase-1 checkpoints S1, S3, L1, Q1 |

## Running

```bash
bash phase5/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase5/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase5/run.sh                           # print the command sequence only
bash phase5/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** predict the set of applicable statute IDs per case. **Metric:** macro-F1 over statutes (micro-F1 and subset
accuracy secondary). The evaluator takes the label dimension from the test truth. Code: `src/lsi.py`, `phase5/configs/`,
`phase5/run.sh`. Train / dev cases that also occur in test are removed (`lsi.notest`, on by default).

### 5.1 Model, loss, decisions
| component | citation | where |
|---|---|---|
| **Label-wise attention head (LWAN)**: one attention query per statute over the encoder's token states → statute-specific document vector → score | Mullenbach et al., *CAML*, NAACL 2018; Chalkidis et al., *Large-Scale Multi-Label Text Classification on EU Legislation*, ACL 2019 | `LSIModel` (head `lwan`) |
| **Asymmetric loss (ASL)**: γ+ = 0, γ− = 4, probability margin 0.05 for the heavy positive/negative imbalance | Ridnik et al., *Asymmetric Loss for Multi-Label Classification*, ICCV 2021 | `lossfn` |
| BCE with capped positive weights √(neg/pos) ≤ 20 (ablation L11, with a pooled head) | — | `lossfn` |
| **Per-statute thresholds** tuned on dev for each statute's F1 (global threshold for statutes with < 5 dev positives) + an **at-least-one-statute** rule kept only if it raises dev macro-F1 | Lipton et al. 2014; Fan & Lin 2007 (SCut thresholding) | `tune`, `decide` |
| Reporting: per-statute F1, rare third vs frequent third of statutes | PS §11.5 | `finish` |

### 5.2 Submissions
| id | Phase-1 model | technique |
|---|---|---|
| L1 | LegalGPT-SC (S1) | **LoRMA+** (compulsory) |
| L2 | LegalHybrid-SC (S3) | **adapters** |
| L3 | LongLegal (L1) | **QLoRA** (NF4), 4096-token window |
| L4 | LegalSeq encoder (Q1) | **AdaLoRA**, 1024 tokens |
| L5 | S1 | full fine-tuning |
| L6 | S1 | partial fine-tuning (top 8 blocks) |
| L7 | S1 | **prompt tuning** (32 soft tokens) |
| L8 | — | weighted ensemble of L1–L7 (dev weights, per-statute thresholds); distillation teacher |
| L9 | first 8 of 20 S1 blocks (≈54M) | **knowledge distillation** from L8 (α·ASL + (1−α)·BCE to teacher probabilities) |
| L10 | — | L8 + L9 |
| L11 | S1 + LoRMA+ | ablation: pooled head + BCE instead of LWAN + ASL |

### 5.3 Results (round 1)
| id | system | dev macro-F1 | dev micro-F1 | test macro-F1 |
|---|---|---|---|---|
| L1 | S1 LoRMA+ | 0.4066 | 0.5035 | — |
| L2 | S3 adapters | 0.4096 | 0.5097 | — |
| L3 | LongLegal QLoRA 4096 | 0.4127 | 0.5100 | — |
| L4 | LegalSeq AdaLoRA 1024 | 0.3010 | 0.3949 | — |
| L5 | S1 full | 0.4115 | 0.5044 | — |
| L6 | S1 partial | 0.4045 | 0.4923 | — |
| L7 | S1 prompt tuning | 0.3267 | 0.4127 | — |
| L8 | ensemble L1–L7 | 0.4344 | 0.5269 | — |
| L9 | KD student (54M) | **0.4135** (best single) | 0.5076 | — |
| L10 | ensemble + L9 | **0.4351** | 0.5284 | — |
| L11 | ablation: pooled head + BCE | 0.4055 | 0.4954 | — |

- Per-statute thresholds beat a global threshold by +0.02 in every model.
- The at-least-one rule has no effect (documents carry 3.9 statutes on average).
- Data: 100 statutes, each with ≥ 272 training positives; 66–70% of documents fit in 2048 tokens.
- Test overlap: 253 train and 96 dev cases also appear in test; they are removed.

### 5.4 Round 2: chunk aggregation and label-correlation stacking
| id | method | citation |
|---|---|---|
| L12–L17 | **Chunk-level prediction + document-level aggregation**: trained members re-scored over consecutive windows from the start (up to 3 × 2048 tokens; 2 × 4096 for LongLegal), per-statute max over chunks | PS §11.1; Chalkidis et al. 2019; Dai et al. 2022 (long-document classification) |
| L18 | weighted ensemble of chunked + original members | — |
| L19 / L20 | **Stacked binary relevance with label correlations**: an MLP over all members' probabilities for all 100 statutes (residual on the mean member logit) learns statute co-occurrence; 5-fold cross-fitted on dev, thresholds on out-of-fold scores; L20 uses original members only | Wolpert 1992; Godbole & Sarawagi, PAKDD 2004; Montañés et al., *Dependent Binary Relevance Models*, Pattern Recognition 2014 |

L18 (chunked + original members) reached about 0.42 on the test, against a leaderboard top of about 0.47.

### 5.5 Round 3: dev cross-fitting
| id | method | citation |
|---|---|---|
| L21 / L22 / L23 | **Dev cross-fitting** of the KD student / S1 full / S3 adapters: 2 runs, each trained on train + one half of dev (the 96 dev cases that also occur in test removed first) and selected on the other half; chunk aggregation at prediction; per-statute thresholds on honest out-of-fold dev scores; test = mean of the 2 runs. The student's dev documents get the L8 ensemble's dev probabilities as soft targets | cross-fitting (Chernozhukov et al. 2018); Hinton et al. 2015 |
| L24 / L25 | weighted ensemble / label-correlation stacker over the cross-fitted, chunked and original members | as above |

**Round 2–3 results (dev macro-F1; cross-fitted members are honest out-of-fold):**

| members / system | dev macro-F1 | baseline |
|---|---|---|
| chunked members (L12–L17) | 0.410–0.417 | originals 0.410–0.4135 → +0.002 to +0.006 |
| L21 KD student, dev cross-fitted | **0.4243** | L9 0.4135 → +0.011 |
| L22 S1 full, dev cross-fitted | 0.4236 | L5 0.4115 → +0.012 |
| L23 S3 adapters, dev cross-fitted | 0.4202 | L2 0.4096 → +0.011 |
| **L24** ensemble (13 members) | **0.4425** | L10 0.4351 |
| L25 stacker | 0.4274 | below L24 and the mean-logit ensemble (0.4413): stacking does not help |

- Dev data adds about +0.011 to every member.
- Chunk aggregation helps slightly.
- Learned label-correlation stacking hurts compared with a plain weighted average.

### 5.6 Round 4: rare statutes and long documents
L24 scored 0.4276 on the test (leaderboard top ≈ 0.47). Our scores already exceed IL-TUR's published LSI baselines
(InLegalBERT 26.23, SOTA 28.08 macro-F1), so this competition split is easier. The remaining gap is in rare statutes:
rare-third F1 ≈ 0.33 vs frequent-third ≈ 0.54.

| id | method | citation |
|---|---|---|
| L26 | S3 adapters (dev cross-fitted) trained with the **distribution-balanced loss**: rebalanced weighting r = (1/n_i) / mean over positives (1/n_j), smoothed α + σ(β(r − μ)) (α 0.1, β 10, μ 0.05), plus negative-tolerant regularisation (λ 2) with a class-prior logit bias (κ 0.05) | Wu et al., *Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets*, ECCV 2020; Huang et al., *Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution*, EMNLP 2021 |
| L27 | Distilled 8-block student (dev cross-fitted) with **multiple-instance chunk training**: up to 3 windows of 2048 tokens per document through the shared encoder, chunk logits pooled by log-sum-exp (minus log k) and trained against the document's statutes; the same aggregation at prediction | Pinheiro & Collobert, CVPR 2015 (LSE pooling); multiple-instance learning for long documents; PS §11.1 |
| L28 / L29 | weighted ensembles of all useful members (interim after L27, final after L26) | — |

Considered but not used:
- NyayGraph-style knowledge-graph inference (2025). It needs statute texts or the IPC structure, which the competition
  does not provide.
- The data-driven label-correlation stacker (L25). It did not beat a weighted average.

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | Phase 5 (LSI, multi-label statute identification) defaults. Backbones are our Phase-1 checkpoints only. |
| `l1_gpt_lorma.yaml` | L1 | submitted | LegalGPT-SC (S1) + LoRMA+ (compulsory), label-wise attention head (one attention query per statute), asymmetric loss, per-statute dev thresholds. |
| `l2_hyb_adapter.yaml` | L2 | submitted | LegalHybrid-SC (S3) + Houlsby adapters, label-wise attention head, asymmetric loss. |
| `l3_long_qlora.yaml` | L3 | submitted | LongLegal (L1) + QLoRA (NF4 4-bit frozen base with double-quantised scales, LoRA r16 on top), 4096-token window. |
| `l4_seq_adalora.yaml` | L4 | submitted | LegalSeq (Q1) encoder + AdaLoRA (rank 12 -> average 4 by importance), first 1024 tokens. |
| `l5_gpt_full.yaml` | L5 | submitted | LegalGPT-SC (S1) full fine-tuning, label-wise attention head, asymmetric loss. |
| `l6_gpt_partial.yaml` | L6 | submitted | LegalGPT-SC (S1) partial / layer-wise fine-tuning (top 8 of 20 blocks), label-wise attention head. |
| `l7_gpt_prompt.yaml` | L7 | submitted | LegalGPT-SC (S1) + prompt tuning (Lester et al. 2021): 32 soft tokens, frozen backbone, label-wise attention head. |
| `l8_ens.yaml` | L8 | submitted | weighted ensemble of L1-L7 (member weights + per-statute thresholds tuned on dev); distillation teacher. |
| `l9_kd.yaml` | L9 | submitted | knowledge distillation (Hinton et al. 2015) -- teacher = the L8 ensemble, student = first 8 of 20 LegalGPT-SC blocks (~54M), L = alpha * ASL(gold) + (1 - alpha) * BCE(student, teacher probabilities). |
| `l10_ens_final.yaml` | L10 | submitted | final weighted ensemble of L1-L7 + the L9 student. |
| `l11_gpt_lorma_pool_bce.yaml` | L11 | submitted | L1 with attention/mean pooling + a linear multi-label layer and BCE (capped pos_weight). |
| `l12_gpt_full_chunk.yaml` | L12 | submitted | lsi_gpt_full (already trained) re-scored with chunk-level predictions + document-level aggregation: consecutive windows from the start of the document (up to 3), per-statute max over chunks (PS 11.1); per-statute thresholds re-tuned on dev. |
| `l13_hyb_adapter_chunk.yaml` | L13 | submitted | lsi_hyb_adapter (already trained) re-scored with chunk-level predictions + document-level aggregation: consecutive windows from the start of the document (up to 3), per-statute max over chunks (PS 11.1); per-statute thresholds re-tuned on dev. |
| `l14_kd_chunk.yaml` | L14 | submitted | lsi_kd_student (already trained) re-scored with chunk-level predictions + document-level aggregation: consecutive windows from the start of the document (up to 3), per-statute max over chunks (PS 11.1); per-statute thresholds re-tuned on dev. |
| `l15_gpt_lorma_chunk.yaml` | L15 | submitted | lsi_gpt_lorma (already trained) re-scored with chunk-level predictions + document-level aggregation: consecutive windows from the start of the document (up to 3), per-statute max over chunks (PS 11.1); per-statute thresholds re-tuned on dev. |
| `l16_long_qlora_chunk.yaml` | L16 | submitted | lsi_long_qlora (already trained) re-scored with chunk-level predictions + document-level aggregation: consecutive windows from the start of the document (up to 2), per-statute max over chunks (PS 11.1); per-statute thresholds re-tuned on dev. |
| `l17_gpt_partial_chunk.yaml` | L17 | submitted | lsi_gpt_partial (already trained) re-scored with chunk-level predictions + document-level aggregation: consecutive windows from the start of the document (up to 3), per-statute max over chunks (PS 11.1); per-statute thresholds re-tuned on dev. |
| `l18_ens_chunk.yaml` | L18 | submitted | weighted ensemble of the chunk-aggregated members (L12-L17) and the original members (L1-L3, L5, L6, L9); weights and per-statute thresholds tuned on dev. |
| `l19_stack.yaml` | L19 | submitted | stacked binary relevance with label correlations: an MLP over all members' probabilities for all 100 statutes (residual on the mean member logit), 5-fold cross-fitted on dev; per-statute thresholds on the out-of-fold dev scores. |
| `l20_stack_orig.yaml` | L20 | submitted | L19's stacker over the original (first-window) members only -- isolates the stacking gain from the chunk gain. |
| `l21_kd_dcv.yaml` | L21 | submitted | L9 (8-block LegalGPT-SC student distilled from the L8 ensemble; dev docs get the ensemble's dev probabilities as soft targets), dev cross-fitted -- two runs, each trained on train + one half of dev (dev cases that also occur in test are removed first) and selected on the other half; chunk aggregation (3 x 2048, per-statute max) at prediction; every dev case scored by the run that held it out (honest per-statute thresholds), test = mean of the two runs. |
| `l21_teach.yaml` | L21T | member | Teacher for L21: the L8 ensemble's probabilities on train AND dev documents (soft targets for every training doc). |
| `l22_gpt_full_dcv.yaml` | L22 | submitted | L5 (LegalGPT-SC full fine-tuning, LWAN + ASL), dev cross-fitted -- two runs, each trained on train + one half of dev (dev cases that also occur in test are removed first) and selected on the other half; chunk aggregation (3 x 2048, per-statute max) at prediction; every dev case scored by the run that held it out (honest per-statute thresholds), test = mean of the two runs. |
| `l23_hyb_adapter_dcv.yaml` | L23 | submitted | L2 (LegalHybrid-SC + adapters, LWAN + ASL), dev cross-fitted -- two runs, each trained on train + one half of dev (dev cases that also occur in test are removed first) and selected on the other half; chunk aggregation (3 x 2048, per-statute max) at prediction; every dev case scored by the run that held it out (honest per-statute thresholds), test = mean of the two runs. |
| `l24_ens_dcv.yaml` | L24 | submitted | weighted ensemble of the dev-cross-fitted members (L21-L23) with the chunked (L12-L17) and original members; member weights and per-statute thresholds tuned on dev. |
| `l25_stack_dcv.yaml` | L25 | submitted | label-correlation stacker (as L19) over the dev-cross-fitted, chunked and original members, 5-fold cross-fitted on dev. |
| `l26_hyb_adapter_db_dcv.yaml` | L26 | submitted | LegalHybrid-SC + adapters (L2/L23 setup) trained with the distribution-balanced loss (Wu et al. ECCV 2020; Huang et al. EMNLP 2021: rebalanced weighting + negative-tolerant regularisation, for rare statutes / macro-F1), dev cross-fitted (2 folds on train + half of dev, test-overlapping dev cases removed), chunk aggregation at prediction. |
| `l27_kd_mil_dcv.yaml` | L27 | submitted | the 8-block distilled student (L21 setup) trained with multiple-instance chunk learning: up to 3 windows of 2048 tokens per document through the shared encoder, chunk logits pooled by log-sum-exp (Pinheiro & Collobert 2015) and trained against the document's statutes, the same aggregation at prediction; dev cross-fitted as L21. |
| `l28_ens_final.yaml` | L28 | submitted | weighted ensemble of every useful member: dev cross-fitted (L21-L23, L26 DB loss, L27 MIL), chunked (L12-L17) and original members; member weights and per-statute thresholds tuned on dev (members that are missing are skipped). |
| `l29_ens_final.yaml` | L29 | submitted | weighted ensemble of every useful member: dev cross-fitted (L21-L23, L26 DB loss, L27 MIL), chunked (L12-L17) and original members; member weights and per-statute thresholds tuned on dev (members that are missing are skipped). |
