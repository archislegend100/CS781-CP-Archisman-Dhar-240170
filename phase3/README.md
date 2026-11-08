# Phase 3: Rhetorical role prediction (RR)

Assign one of 13 rhetorical roles to every sentence of corporate-law and IT-law judgments. Fine-tuned hierarchical context models (chunk encoder, BiGRU, CRF) and whole-document taggers over frozen Phase-1 features are cross-fitted over train + dev, and the final ensemble is calibrated on out-of-fold predictions of all 90 labelled documents.

| | |
|---|---|
| Final submission | **R30** (`phase3/configs/r30_cvens.yaml`) |
| Codabench | macro-F1 0.5751; rank 14 (32 valid submissions) |
| Compute | 31.4 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/rr.py`, `src/peft.py` |
| Needs | all Phase-1 checkpoints (frozen features) and Z3 |

## Running

```bash
bash phase3/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase3/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase3/run.sh                           # print the command sequence only
bash phase3/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** one of 13 roles per sentence. **Metric:** macro-F1 over all sentences of both domains (sklearn,
`zero_division=0`). Our implementation was checked identical to `evaluate_phase3.py` on dev.

**Rules:** ≥ 6 substantially different experiments, ≥ 5 adaptation techniques, LoRMA compulsory, knowledge
distillation mandatory (larger teacher → smaller student), all four Phase-1 architectures. Code: `src/rr.py`,
`phase3/configs/`, `phase3/run.sh`.

### 3.1 Model
| component | citation | where |
|---|---|---|
| **Hierarchical sequence labelling**: sentences encoded in context, sentence-level BiGRU, CRF over roles | Brack et al., *Sequential Sentence Classification in Research Papers using Cross-Domain Multi-Task Learning* (HSLN), 2021; Kalamkar et al., *Corpus for Automatic Structuring of Legal Documents*, LREC 2022; Bhattacharya et al., JURIX 2019 | `RRModel` |
| **Packed sentence chunks** (≤ 1024 tokens, 2048 for LongLegal, 50% sentence-aligned overlap), bidirectional backbone, mean-pooled sentence vectors; each sentence read from its most central chunk | Dukić & Šnajder 2024 (unmasking); Cohan et al., *Pretrained LMs for Sequential Sentence Classification*, EMNLP 2019 (joint encoding of neighbouring sentences) | `chunks`, `emissions` |
| **Linear-chain CRF** over the role sequence (exact Viterbi over the whole document) | Lafferty, McCallum, Pereira, ICML 2001; Lample et al. 2016 | `CRF`, `viterbi` |
| **Auxiliary label-shift prediction** (does the role change from the previous sentence?) | Malik et al., *Semantic Segmentation of Legal Documents via Rhetorical Roles*, NLLP 2022 (MTL-BiLSTM-CRF) | `RRModel.shift` |
| **Relative-position buckets** (10) and **domain embedding** (CL/IT) | Bhattacharya et al. 2019 (position features); multi-domain learning | `RRModel.pos/dom` |
| Class-weighted CE (max/count)^0.5 alongside the CRF loss | — (cost-sensitive learning) | `classweights` |
| **Sentence-only ablation** (no neighbours, no position, GRU or CRF) | context analysis required by PS §9.5 | `mode: sent` (R7) |

### 3.2 Adaptation techniques
| id | Phase-1 model | technique | citation |
|---|---|---|---|
| R1 | LegalGPT-SC (S1) | **LoRMA+** (compulsory) r = α = 8 | Bihany, Patel, Modi 2025 |
| R2 | LegalHybrid (S3) | **rsLoRA** r16, scale α/√r | Kalajdzievski 2023, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* |
| R3 | LongLegal (L1) | **QLoRA**: frozen projections in NF4 (4.13 bits/param, blockwise-64 absmax, double-quantised scales), LoRA r16 on top; pure-torch NF4 | Dettmers et al., NeurIPS 2023 |
| R4 | LegalSeq encoder (Q1) | **AdaLoRA**: SVD-form updates, rank 12 → average 4 by sensitivity × uncertainty importance, cubic budget schedule, orthogonality penalty | Zhang et al., ICLR 2023 |
| R5 | S1 | full fine-tuning | — |
| R6 | S1 | **IA³** | Liu et al., NeurIPS 2022 |
| R11 | S1 | **prompt tuning** (32 soft tokens) | Lester et al., EMNLP 2021 |
| R9 | student = first 8 of 20 S1 layers (≈54M) | **knowledge distillation** from the R8 ensemble (six models): α·(CE + CRF + shift) + (1−α)·T²·KL at T = 2, plus **intermediate-representation distillation** (MSE between projected student and teacher sentence vectors) | Hinton et al. 2015; Jiao et al., *TinyBERT*, Findings EMNLP 2020 |

### 3.3 Ensembling and calibration
| id | method | citation |
|---|---|---|
| R8 | Mean of member sentence probabilities, averaged CRF transitions; transition scale and **per-label decision biases tuned for macro-F1 on the official dev set** | Lipton et al. 2014; Dietterich 2000 |
| R10 | Top-3 single context models **refit on train + dev** (fixed best-epoch count) + the KD student, with R8's dev-tuned biases | Phase-2 lesson: more training data plus held-out calibration |

### 3.4 Round 2: frozen features + whole-document tagger (after R1–R11 plateaued at 0.5392 test)
**Diagnosis:**
- Single fine-tuned models are weak and noisy (dev 0.36–0.49) and peak by epoch 4–5.
- Context matters (sentence-only R7 0.33 vs R1 0.47), but R1–R6 only see one ~30-sentence chunk at a time.

| id | method | citation |
|---|---|---|
| feats | **Frozen Phase-1 sentence features**: each backbone run once as pretrained (causal; LegalSeq encoder bidirectional), mean sentence state at 50/75/100% depth, concatenated; causal models take each sentence from the chunk with the most left context | transfer of Phase-1 representations (PS §9.1); feature-based transfer as in Peters et al. 2018 |
| R12 | **Whole-document hierarchical BiLSTM-CRF** over the four architectures' frozen features (S1, S3, L1, Q1) + position (20 buckets) + domain embeddings, label-shift auxiliary loss, class weights, feature dropout 0.5; bag of 5 seeds | Bhattacharya et al., *Identification of Rhetorical Roles of Sentences in Indian Legal Judgments*, JURIX 2019 (Hier-BiLSTM-CRF); Kalamkar et al. 2022 (HSLN); Malik et al. 2022 |
| R13 | R12 + three more of our Phase-1 checkpoints as feature views (G1, H1, Z3) | multi-view feature ensembling |
| R14 | Dev-calibrated ensemble of R12, R13 and R1–R6 with **tuned member weights**, transition scale and per-label biases | Lipton et al. 2014 |
| R15 / R17 | Calibration ensemble (R13 + R1/R2/R3) and the same structure **refit on train + dev** (R16 document tagger at its median best epoch + R1A/R2A/R3A) with R15's weights and biases | — |

### 3.5 Round 3: cross-fitting, lexical view, out-of-fold calibration (R8 0.5392 / R13 0.5393 plateau)
**Diagnosis:**
- Dissent and PrecedentOverruled score 0.00 on dev in every model, a fixed 2/13 ≈ 0.154 of macro-F1. Dev-tuned
  biases cannot fix labels that dev barely contains.
- Calibration fitted on a small dev set transfers poorly.

| id | method | citation |
|---|---|---|
| R18 | **Cross-fitted document tagger**: 5 folds × 3 seeds over all labelled docs (train + dev); out-of-fold predictions for every labelled doc; CRF transition scale and per-label biases tuned on all of them; test = mean of the 15 fold models | Wolpert 1992 (stacked generalisation); Chernozhukov et al. 2018 (cross-fitting); Lipton et al. 2014 |
| R18 | **Lexical LSA view**: TF-IDF word 1–2-grams over official text → 256-d truncated SVD, concatenated with the frozen Phase-1 features (cue phrases such as "learned counsel for the appellant", "overruled") | Deerwester et al. 1990 (LSA); Saravanan et al. 2008, Bhattacharya et al. 2019 (cue features for legal RR) |
| R19 | R18 with per-domain (CL / IT) decision biases | domain-specific calibration |
| R20 | R18 without the lexical view (ablation) | — |
| R21 | R18 (honest OOF on dev) blended with R1–R6, member weights and biases tuned on dev | — |

### 3.6 Round 4: per-view PCA
R18's input was 16,384-d (7 frozen views × 2,304 + 256 lexical) for about 72 training documents per fold.
- R18 OOF: 0.4734 on all 90 labelled docs, dev part 0.4519. Dissent rose from 0.00 to 0.153 through out-of-fold
  biases (Dissent +3.0).
- PrecedentOverruled has only 3 sentences in train + dev.

| id | method | citation |
|---|---|---|
| R22 | Unsupervised **per-view PCA** to 256 dims before concatenation (2,048-d input; lexical view 1/8 instead of 1/64) | Pearson 1901; Halko et al. 2011 (randomised PCA) |
| R23 | R22 with a smaller, more regularised tagger and 5 seeds per fold (25 models) | bagging (Breiman 1996) |
| R24 | R22 + R23 + R18 blended with R1–R6, weights and biases tuned on dev | — |

### 3.7 Round 5: full sparse lexical channel (R24 reached 0.5494 test)
Every earlier system saw a sentence only through mean-pooled states of a 5k-BPE, 130M model, plus a 256-d LSA
compression. Rhetorical roles are largely signalled by specific cue phrases.

| id | method | citation |
|---|---|---|
| R25 | Cross-fitted document tagger (R22) + **sparse TF-IDF channel**: all word uni/bigrams (df ≥ 2) through a learned EmbeddingBag with word dropout, trained jointly with the Phase-1 features | Saravanan, Ravindran, Raman 2008 (cue phrases for legal RR); Bhattacharya et al. 2019; Joulin et al. 2017 (fastText bag of n-grams) |
| R27 | R25 with df ≥ 3, word dropout 0.35, 5 seeds | — |
| R26 | R25 + R27 + R22 + R23 + R1/R2/R3/R6, dev-tuned weights and biases | — |

### 3.8 Round 6: cross-fitted fine-tuned models + OOF-calibrated ensemble
| id | method | citation |
|---|---|---|
| R28 / R29 | **Cross-fitted fine-tuned models**: R2 (hybrid rsLoRA) and R1 (LoRMA+) retrained in the same 5 domain-stratified folds over train + dev (fixed epochs 4 / 8); each labelled doc predicted by the fold model that did not see it; test = mean of the 5 fold models; OOF-calibrated | Wolpert 1992; Chernozhukov et al. 2018 |
| R30 | **Ensemble of cross-fitted members only** (R25, R27, R22, R23, R18, R28, R29): member weights, transition scale and per-label biases tuned on the out-of-fold predictions of all 90 labelled docs (~21k sentences) instead of the ~20 dev docs | stacking / super learner (van der Laan et al. 2007) |
| R31 | R30 with per-domain biases | — |

### 3.9 Round 7: more cross-fitted fine-tuned members (R30 reached 0.5751 test)
- R30 OOF 0.508: members R28 0.483, R29 0.472, document taggers 0.468–0.473; the sparse channel (R25/R27) got
  weight 0.
- R31 (per-domain biases) OOF 0.517.
- R30 is the final submission (test 0.5751).

### 3.10 Results
| id | system | dev F1 (CL / IT) | test F1 |
|---|---|---|---|
| R8 | fine-tuned ensemble | 0.525 | 0.5392 |
| R12 / R13 | document tagger (4 / 7 frozen views) | 0.479 / 0.504 | — / 0.5393 |
| R14 / R15 | blended ensembles | 0.542 / 0.547 | — |
| R18 | cross-fitted, frozen 7 views + lexical LSA, OOF-calibrated | OOF 0.4734 (dev part 0.4519) | — |
| R20 | R18 without the lexical view | OOF 0.4641 (dev part 0.4348) | — |
| R21 | R18 + R1–R6 blend, dev-tuned | 0.5229 (dev-tuned) | — |
| R22 | R18 + per-view PCA 256 | OOF 0.4716 (dev part 0.4714) | — |
| R23 | R22, smaller tagger, 5 seeds | OOF 0.4683 (dev part 0.4739) | — |
| R24 | R22 + R23 + R1–R6 blend, dev-tuned | 0.5382 (dev-tuned) | — |

**Findings for the report:**
- Every honest estimate lands at 0.46–0.47 OOF and 0.45–0.49 dev, across fine-tuned PEFT models, frozen-feature
  document taggers, cross-fitting, PCA and lexical features. The ceiling comes from the representations, not from
  the head or the calibration.
- The lexical view adds about +0.009 OOF (R18 vs R20).
- Out-of-fold calibration makes Dissent learnable (0.00 → 0.10–0.15).
- PrecedentOverruled (3 training sentences) stays at 0, a fixed 1/13 loss for every system.

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | Phase 3 (rhetorical roles) defaults. Backbones are the trained Phase-1 checkpoints; nothing external is loaded. |
| `feats.yaml` | F | member | Frozen Phase-1 sentence features for the document-level taggers (python rr.py phase3/configs/feats.yaml feats). Every backbone is one of our own Phase-1 checkpoints, run causally (as pretrained) or, for LegalSeq, its encoder. |
| `r1_gpt_lorma.yaml` | R1 | submitted | LegalGPT-SC (S1) + LoRMA+ (compulsory; Bihany, Patel, Modi 2025), r = alpha = 8 on all projections, unmasked, hierarchical context model (packed sentence chunks -> mean-pooled sentence vectors + position + domain -> BiGRU -> CRF). |
| `r2_hyb_rslora.yaml` | R2 | submitted | LegalHybrid-SC (S3) + rank-stabilised LoRA (Kalajdzievski 2023): scale alpha/sqrt(r) = 16/4, r = 16; attention layers unmasked (Hyena mixers stay causal); same hierarchical context model as R1. |
| `r3_long_qlora.yaml` | R3 | submitted | LongLegal (L1) + QLoRA (Dettmers et al. 2023): frozen projections stored in NF4 with double-quantised scales, LoRA r16/alpha32 on top; 2048-token chunks (symmetric local window 512 + bidirectional global layers) for more document context per chunk. |
| `r4_seq_adalora.yaml` | R4 | submitted | LegalSeq (Q1) encoder (natively bidirectional) + AdaLoRA (Zhang et al. 2023): SVD-form updates starting at rank 12 per projection, importance-based budget pruning to an average rank of 4, orthogonality penalty 0.1. |
| `r5_gpt_full.yaml` | R5 | submitted | LegalGPT-SC (S1), full fine-tuning (reference point for "can PEFT approach full adaptation?"). |
| `r6_gpt_ia3.yaml` | R6 | submitted | LegalGPT-SC (S1) + IA3 (Liu et al. 2022): learned rescaling of keys/values and FFN activations (~50K params). |
| `r7_gpt_lorma_sent.yaml` | R7 | submitted | context ablation -- R1's LoRMA+ on S1 but sentence-only: each sentence encoded alone, mean pooled, linear head (no neighbouring sentences, no position, no BiGRU, no CRF). |
| `r8_ens.yaml` | R8 | submitted | ensemble of the context models R1-R6 (mean sentence probabilities, averaged CRF transitions) with the CRF transition scale and per-label decision biases tuned for macro-F1 on the official dev set (Lipton et al. 2014). |
| `r9_kd.yaml` | R9 | submitted | knowledge distillation (Hinton et al. 2015). Teacher = the R8 ensemble (six Phase-1-based models, ~0.4B params in total); student = the first 8 of 20 LegalGPT-SC layers (~54M), fully fine-tuned with L = alpha * (CE + CRF + shift) + (1 - alpha) * T^2 * KL(teacher probs // student) + beta * MSE(projected student sentence vector, R1's sentence vector) (intermediate-representation distillation, cf. TinyBERT, Jiao et al. 2020). |
| `r10_ens_refit.yaml` | R10 | submitted | the best R8 members refit on train + dev (fixed epoch count = their best dev epoch), ensembled with R8's dev-tuned biases and transition scale (members are set by phase3/run.sh: the top single models by dev F1) + the KD student. |
| `r11_gpt_prompt.yaml` | R11 | submitted | LegalGPT-SC (S1) + prompt tuning (Lester et al. 2021), 32 soft tokens, backbone frozen. |
| `r12_seq4.yaml` | R12 | submitted | whole-document hierarchical BiLSTM-CRF (Bhattacharya et al. 2019; HSLN, Kalamkar et al. 2022) over FROZEN sentence features of all four Phase-1 architectures (S1, S3, L1, Q1 encoder; mean state at 3 depths each), with relative-position and domain embeddings, label-shift auxiliary loss, class weights; bag of 5 seeds. Unlike R1-R6 the sequence model sees every sentence of the document at once, and nothing in the backbones is fitted to the small training set. |
| `r13_seq_wide.yaml` | R13 | submitted | R12 with three more of our Phase-1 checkpoints as feature views (G1 general LegalGPT, H1 general LegalHybrid, Z3 self-ranked LegalGPT): more diverse frozen representations of the same sentences. |
| `r14_ens.yaml` | R14 | submitted | dev-calibrated ensemble of the two document taggers (R12, R13) and the fine-tuned context models (R1-R6): member weights, CRF transition scale and per-label biases tuned for macro-F1 on the official dev set. |
| `r15_ens4.yaml` | R15 | submitted | calibration ensemble with the same structure as R17 (document tagger + R1/R2/R3); its dev-tuned weights and biases are reused by R17, whose members are all refit on train + dev. |
| `r16_seq_wide_all.yaml` | R16 | submitted | R13's document tagger refit on train + dev for the median best epoch of its seeds (set by phase3/run.sh). |
| `r17_ens_all.yaml` | R17 | submitted | everything refit on train + dev (R16 document tagger + R1/R2/R3 refits) with R15's dev-tuned member weights, transition scale and per-label biases. |
| `r18_cv_lex.yaml` | R18 | submitted | cross-fitted document tagger. 5 folds x 3 seeds over ALL labelled documents (train + dev); input = frozen features of seven Phase-1 checkpoints + a lexical LSA view (TF-IDF word 1-2-grams -> 256-d SVD over the official text); the CRF transition scale and per-label biases are tuned on the out-of-fold predictions of every labelled document (so rare roles that dev lacks, e.g. Dissent / PrecedentOverruled, are calibrated too); test = average of the 15 fold models (stacking / cross-fitting, Wolpert 1992). |
| `r19_cv_lex_dom.yaml` | R19 | submitted | R18 with separate per-label biases for CL and IT (26 decision biases, tuned out-of-fold on all labelled docs). |
| `r20_cv_nolex.yaml` | R20 | submitted | ablation of R18 without the lexical view (frozen Phase-1 features only), cross-fitted and OOF-calibrated. |
| `r21_ens_cv.yaml` | R21 | submitted | R18 (cross-fitted, honest out-of-fold on dev) blended with the fine-tuned context models R1-R6; member weights, transition scale and per-label biases tuned on the official dev set. |
| `r22_cv_pca.yaml` | R22 | submitted | R18 with each frozen Phase-1 view reduced to 256 dims by unsupervised PCA before concatenation (7 x 256 + 256 lexical = 2048 inputs instead of 16384), so the lexical view is 1/8 of the input instead of 1/64 and the BiLSTM-CRF overfits less on ~72 training documents per fold; cross-fitted and OOF-calibrated as R18. |
| `r23_cv_pca_small.yaml` | R23 | submitted | R22 with a smaller, more regularised tagger (projection 256, BiLSTM 128 x 2, feature dropout 0.6) and 5 seeds per fold (25 models): lower variance for a small training set. |
| `r24_ens_pca.yaml` | R24 | submitted | R22 + R23 (cross-fitted) blended with the fine-tuned R1-R6; member weights, transition scale and per-label biases tuned on the official dev set (where the cross-fitted members are honest out-of-fold). |
| `r25_cv_sparse.yaml` | R25 | submitted | R22 (cross-fitted, per-view PCA Phase-1 features + LSA) plus a FULL sparse lexical channel: each sentence's TF-IDF over all word uni/bigrams (df >= 2) enters the document tagger through a learned EmbeddingBag (word dropout 0.2), trained jointly with the Phase-1 features, so cue phrases keep their identity (cf. cue-phrase features in legal RR: Saravanan et al. 2008; Bhattacharya et al. 2019). OOF-calibrated as R18. |
| `r26_ens_sparse.yaml` | R26 | submitted | R25 + R22 + R23 blended with the fine-tuned R1-R6 (R24's members + R25); weights, transition scale and per-label biases tuned on the official dev set. |
| `r27_cv_sparse_reg.yaml` | R27 | submitted | R25 with stronger regularisation of the sparse channel (df >= 3, word dropout 0.35) and 5 seeds per fold. |
| `r28_cf_hyb.yaml` | R28 | submitted | cross-fitted R2 (LegalHybrid-SC + rsLoRA, hierarchical context model): 5 fold models, each trained on the labelled docs (train + dev) outside its fold for R2's best epoch count (4); every labelled doc predicted by the model that did not see it; test = mean of the 5; transition scale + per-label biases tuned on all OOF predictions. |
| `r29_cf_lorma.yaml` | R29 | submitted | cross-fitted R1 (LegalGPT-SC + LoRMA+, hierarchical context model), 5 folds x 8 epochs, as R28. |
| `r30_cvens.yaml` | R30 | submitted | ensemble of cross-fitted members only -- document taggers (R25, R27, R22, R23, R18) and cross-fitted fine-tuned models (R28 hybrid rsLoRA, R29 LoRMA+), all with honest out-of-fold predictions on the same 90 labelled docs, so member weights, transition scale and per-label biases are tuned on ~21k sentences instead of the dev set alone. |
| `r31_cvens_dom.yaml` | R31 | submitted | R30 with per-domain (CL / IT) decision biases, tuned on all OOF predictions. |
