# Phase 2: Legal named entity recognition (L-NER)

Tag 12 entity types as character spans in Indian judgments. Phase-1 decoders are fine-tuned bidirectionally (causal mask removed) with PEFT; the final system is a bag of 14 PEFT members on two frozen backbones whose decision biases, confidence thresholds and document-level rules are all chosen out-of-fold.

| | |
|---|---|
| Final submission | **N24** (`phase2/configs/n24_oof_wide_doc.yaml`) |
| Codabench | strict macro-F1 0.6241; rank 13 (26 valid submissions) |
| Compute | 11.4 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/ner.py`, `src/peft.py` |
| Needs | Phase-1 checkpoints S1, S3, L1, Q1 |

## Running

```bash
bash phase2/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase2/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase2/run.sh                           # print the command sequence only
bash phase2/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** 12 labels, character spans. **Metric:** strict macro-F1 (start, end and label must all match; every
label weighs 1/12).

**Rules:** at least 6 different submissions, at least 5 adaptation techniques, LoRMA compulsory, at least 1
distillation, all four Phase-1 architectures, our own Phase-1 models only, official data only.

**Benchmark:** IL-TUR (Joshi et al., ACL 2024); the paper's best baseline is InLegalBERT + CRF at 48.58.

### 2.1 Representation and decoding
| method | citation | where |
|---|---|---|
| Removing the causal mask in decoder backbones (bidirectional fine-tuning); symmetric band mask for LongLegal; only the attention layers unmasked in the hybrid | Dukić & Šnajder, *Looking Right is Sometimes Right*, Findings ACL 2024; BehnamGhader et al., *LLM2Vec*, COLM 2024 | `models/common.py` (`Block.causal`, `bandmask`), `ner.unmask` |
| Exact character spans per BPE piece (from UTF-8 byte lengths); alignment ceiling 0.970 | — | `ner.tokenize`, `ner.ceiling` |
| BIO tagging + **BIO-constrained Viterbi** | Lample et al. NAACL 2016 (BIO + CRF decoding) | `ner.viterbi` |
| **Biaffine span head** + greedy non-overlap selection (N5) | Yu, Bohnet, Poesio, *NER as Dependency Parsing*, ACL 2020 | `ner.Biaffine` |
| Sliding windows 1024/256 (LongLegal 2048/512); each token read from its most central window | Devlin et al. NAACL 2019 (document stride) | `ner.windows`, `ner.tokenlogp` |
| Inverse-frequency class weights (max/count)^γ | — (standard cost-sensitive learning) | `ner.weights` |
| Punctuation trim rules and end-offset convention learned from training gold | — | `ner.learntrim`, `ner.endconvention` |

### 2.2 Adaptation techniques (≥ 5 required; LoRMA compulsory)
| technique | citation | submissions |
|---|---|---|
| Full fine-tuning | — | N1 (LegalSeq encoder), N7, N8 (LegalGPT) |
| **LoRMA+** (multiplicative low-rank, h = ((α/r)BA + I)W₀x, computed as y + (α/r)B(Ay)) | Bihany, Patel, Modi, *LoRMA*, Findings ACL 2025 (arXiv 2506.07621) | N2, N161/N162, N201/N202, N221/N222 |
| **LoRMAπ** (rotated BA, identity at initialisation) | same paper | N15 |
| LoRA | Hu et al., ICLR 2022 (arXiv 2106.09685) | N3 (hybrid), N171/N172, N203, N223/N224, N225/N226, N227 |
| Houlsby adapters | Houlsby et al., ICML 2019 | N4 (LongLegal) |
| Partial fine-tuning (top 8 layers) | Lee et al. 2019, *What would Elsa do?* (layer freezing) | N5 |
| IA³ | Liu et al., *Few-Shot PEFT is Better and Cheaper than ICL*, NeurIPS 2022 | N13 |
| Prompt tuning (32 soft prompts) | Lester, Al-Rfou, Constant, EMNLP 2021 | N14 |
| **Knowledge distillation**: 8-layer student, KL at T = 2, α = 0.5, transfer set of 2,000 unlabelled Phase-1 Supreme Court windows | Hinton, Vinyals, Dean 2015; Sanh et al. 2019 (DistilBERT, layer-truncated students) | N6, N9, N10 |

### 2.3 Model selection and ensembling
| method | citation | submissions |
|---|---|---|
| Early stopping on 10 held-out documents with the official strict macro-F1; all-docs refit for a fixed number of epochs | Prechelt 1998 (early stopping) | all; N8, N10 refits |
| Ensemble = mean of per-token probabilities, subset chosen on held-out documents (≤ 3 models, ≤ 300M) | Dietterich 2000 (ensembles); Lakshminarayanan et al. NeurIPS 2017 | N11, N12, N18 |
| 2-fold cross-validation ensembles | Kohavi IJCAI 1995 | N16, N17 |
| **Out-of-fold (OOF) calibration**: every training document is predicted by models that never saw it, and all decisions are tuned on all 70 documents instead of 10 | Wolpert 1992 (stacked generalisation); standard OOF practice | N19–N24 (`ner.oof`) |
| **Bagging**: all-docs refits and extra seeds; PEFT members share frozen backbones, so each extra member costs only its adapter | Breiman 1996 (bagging) | N20, N22, N24 |

### 2.4 Decision calibration for macro-F1 (chosen out-of-fold)
| method | citation | submissions |
|---|---|---|
| **Per-label decision bias** on B/I scores before Viterbi; greedy coordinate search, ties keep the gentlest bias | Lipton, Elkan, Narayanaswamy, *Optimal Thresholding of Classifiers to Maximize F1*, ECML-PKDD 2014 | N19–N24 |
| **Per-label span-confidence threshold** (mean ensemble probability of the span's tags) | same + confidence-based filtering | N21, N23, N24 |

### 2.5 Post-processing rules (each kept only where it improves F1; per label when possible)
| rule | citation | submissions |
|---|---|---|
| Document-level label consistency (copy a label to unlabelled exact repeats of a name) | Krishnan & Manning, *An Effective Two-Stage Model for Exploiting Non-Local Dependencies in NER*, ACL 2006; Finkel, Grenager, Manning ACL 2005 | N11 onwards |
| Role-cue APP/RESP relabelling (nearest "appellant/respondent") | Kalamkar et al., *Named Entity Recognition in Indian Court Judgments*, NLLP 2022 (preamble rules) | N12, N18, N19+ (selected out-of-fold; off in N21) |
| Learned boundary repair from word inside/outside statistics | — (data-driven boundary correction) | N18, N19+ |
| **Gazetteer** from training gold (frequent, ≥ 80% label-pure strings), built out-of-fold | Ratinov & Roth, CoNLL 2009 (gazetteers in NER) | N19+ |
| **Two-stage majority vote** over case-folded mentions | Krishnan & Manning ACL 2006 | N23, N24 |
| **Case-insensitive consistency** (capitalised cause-title names ↔ body mentions) | Kalamkar et al. 2022; Krishnan & Manning 2006 | N23, N24 |
| **Prosecution-witness cue** (`(PW-5)`, `PW 6-` → WIT) | trigger-based labelling as in Kalamkar et al. 2022 | N23, N24 |
| **Cause-title party parsing**: names before "Versus" in the preamble or final order → APP, after it → RESP | Kalamkar et al. 2022 (preamble party extraction) | N25 |
| **Counsel-side fix**: A.COUNSEL vs R.COUNSEL from the following "for the appellant/respondent" | trigger rule, same source | N25 |

### 2.6 Phase-2 results (Codabench strict macro-F1)
| id | system | test F1 |
|---|---|---|
| N1 | LegalSeq encoder, full fine-tuning | 0.2980 |
| N2 | LegalGPT-SC + LoRMA+ | 0.5213 |
| N3 | LegalHybrid + LoRA | 0.5120 |
| N4 | LongLegal + adapters | 0.4830 |
| N5 | partial fine-tuning + biaffine | 0.4111 |
| N6 | KD student (8 layers) | 0.5086 |
| N7–N18 | full fine-tuning, refits, KD v2, ensembles, IA³, prompt tuning, LoRMAπ, CV ensembles, boundary repair | — (best of these ≤ 0.55) |
| N19 | OOF-calibrated 4 fold models | — |
| N20 | N19 + all-docs refits (7 members, 280M deployed) | **0.5914** |
| N21 | N20 + confidence thresholds | **0.6027** |
| N22 | wide bagged OOF (14 members) | — |
| N23 | N21 + vote / case-insensitive consistency / witness cue | — |
| N24 | N22 + N23 rules | **0.6241** |
| N25 | N24 + cause-title parsing + counsel-side fix + finer bias grid (3 passes) | 0.6220 (within noise of N24) |
| N26 | N24 + four more PEFT members in the test ensemble (N2, N3, N13 IA³, N15 LoRMAπ) | 0.6237 |

**Plateau (report discussion):** N24, N25 and N26 differ by at most 0.002 (0.6241 / 0.6220 / 0.6237), about one
or two entities on a rare label over 35 test documents. More ensemble members (N26) and more rules (N25) no longer
help. The remaining error lives in the backbone representations: APP, RESP, WIT, AUTH and PREC need
document-level reasoning. Gains so far came from decision quality: OOF calibration +0.04, confidence thresholds
+0.011, document rules +0.021.

**Future work:**
- Semi-supervised self-training on unlabelled Phase-1 Supreme Court text (Xie et al. CVPR 2020, Noisy Student;
  Liang et al. KDD 2020, BOND).
- A jointly trained CRF layer (Lafferty et al. ICML 2001).
- Longer-context members (LongLegal at 4k) in the OOF ensemble.

**N21 out-of-fold progression:**

| step | F1 |
|---|---|
| ensemble | 0.4997 |
| bias | 0.5202 |
| confidence | 0.5310 |
| boundary | 0.5326 |
| gazetteer | 0.5335 |
| consistency | 0.5531 |

Weakest labels at that point: APP 0.29, RESP 0.33, WIT 0.38, AUTH 0.40, PREC 0.42.

### 2.7 Error analysis and inference benchmark
- **RESP** is mostly confused with APP.
- **PREC** errors are mostly boundary errors.
- **APP** has many misses; **WIT**, **AUTH** and **COURT** have spurious hits.
- Party names repeat within a document (APP 77%, RESP 80%, WIT 81%, AUTH 93%), but headers are in capitals and
  body mentions in title case. This motivated 2.5's vote, case-insensitive consistency and witness-cue rules.
- **T4 inference benchmark** (Kaggle, fp16 exports, `ner.bench`):
  - peak memory 0.6–2.4 GB;
  - agreement with submitted predictions 0.970–0.993;
  - native GQA fell back to the math kernel on the T4 (sm75); fixed by expanding K/V on GPUs older than sm80.
  - PEFT reduces trainable/storage parameters, not inference cost.

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | Phase 2 (L-NER) defaults. Backbones are the trained Phase-1 checkpoints; nothing external is loaded. |
| `n1_seq_full.yaml` | N1 | submitted | LegalSeq (Q1) bidirectional encoder, full fine-tuning, BIO token head + constrained Viterbi. |
| `n2_gpt_lorma.yaml` | N2 | submitted | LegalGPT (S1, Supreme-Court adapted), LoRMA+ (compulsory; r = alpha = 8 as in the paper) on all attention/FFN projections, causal mask removed in every layer, BIO token head. |
| `n3_hyb_lora.yaml` | N3 | submitted | LegalHybrid (S3), LoRA r=16 (alpha 32) on attention/FFN/Hyena projections, attention layers unmasked (Hyena mixers stay causal), BIO token head. |
| `n4_long_adapter.yaml` | N4 | submitted | LongLegal (L1), Houlsby bottleneck adapters (64) after every attention and FFN sublayer, 2048-token windows, local layers made symmetric (/i - j/ < 512) and global layers fully bidirectional, BIO token head. |
| `n5_gpt_partial_span.yaml` | N5 | submitted | LegalGPT (S1), partial fine-tuning (top 8 of 20 layers + final norm), biaffine span head (span-based vs token-based comparison with N2), unmasked attention. |
| `n6_kd.yaml` | N6 | submitted | knowledge distillation. Teacher = best token-head model among N1-N4 (set on the command line by phase2/run.sh); student = the first 8 of 20 LegalGPT (S1) layers (~54M of 130M params), fully fine-tuned on L = alpha * CE(gold) + (1 - alpha) * T^2 * KL(teacher // student), plus KL on 2000 unlabeled windows of official Phase-1 Supreme Court text (transfer set, Hinton et al. 2015). |
| `n7_gpt_full.yaml` | N7 | submitted | LegalGPT (S1), full fine-tuning, causal mask removed in all layers, BIO token head. Same backbone and head as N2 (LoRMA+), so N7 vs N2 isolates full vs parameter-efficient fine-tuning. |
| `n8_gpt_full_all.yaml` | N8 | submitted | N7 retrained on all 70 training docs (no held-out docs) for N7's best epoch count (set by phase2b.sh). |
| `n9_kd2.yaml` | N9 | submitted | distillation from the stronger full-fine-tuned teacher N7 into the 8-layer LegalGPT student. |
| `n10_kd2_all.yaml` | N10 | submitted | N9's student trained on all 70 docs, distilled from N8 (teacher also trained on all docs), for N9's best epoch count. |
| `n11_ens.yaml` | N11 | submitted | ensemble of token-head runs (mean of per-token tag probabilities -> constrained Viterbi). The member subset (<= 3 models, total params <= 300M) and the labels that get document-level consistency are chosen on the shared held-out docs; members must be held-out runs (not all-docs ones). |
| `n12_ens_rules.yaml` | N12 | submitted | N11's ensemble selection + role-cue APP/RESP re-labelling + document-level consistency, each switched on only if it improves held-out macro-F1. Kept as a separate submission so N11 vs N12 measures the rules. |
| `n13_gpt_ia3.yaml` | N13 | submitted | LegalGPT (S1), IA3 (Liu et al., NeurIPS 2022): learned rescaling of keys/values and FFN activations (~50K trainable backbone params), unmasked, BIO token head. |
| `n14_gpt_prompt.yaml` | N14 | submitted | LegalGPT (S1), prompt tuning (Lester et al., EMNLP 2021): 32 soft-prompt vectors initialised from vocabulary embeddings, backbone frozen, unmasked (every token sees the prompt), BIO token head. |
| `n15_gpt_lormapi.yaml` | N15 | submitted | LegalGPT (S1), LoRMA-pi (permutation-based rank inflation, Bihany et al. 2025) on q / kv / o, r = 8, unmasked, BIO token head. Compare with N2 (LoRMA+). |
| `n16_lorma_cv.yaml` | N16 | submitted | 2-fold cross-validated LoRMA+ ensemble: the fold1- and fold2-trained models (each selected on a full held-out fold) averaged at the token level; APP consistency kept from N11's held-out finding. 263.8M parameters. |
| `n17_hyb_cv.yaml` | N17 | submitted | 2-fold cross-validated LegalHybrid + LoRA ensemble (as N16). 268.8M parameters. |
| `n18_ens_boundary.yaml` | N18 | submitted | ensemble selection over all held-out runs (incl. IA3 / prompt / LoRMA-pi) + role-cue fix + learned boundary repair + document-level consistency; each post-processing step per label only if it helps on the held-out docs. |
| `n19_oof.yaml` | N19 | submitted | out-of-fold calibrated system. Each training doc is predicted by the 2-fold models trained on the other fold (LoRMA+ on S1 and LoRA on S3, token-probability mean), so all 70 docs give honest predictions. On them we choose: per-label decision biases on the B/I scores (macro-F1 thresholding: rare labels are worth 1/12 each), the role-cue APP/RESP fix, learned boundary repair, a training-gold gazetteer and document consistency (each per label, kept only if OOF F1 rises). Trim rules, boundary statistics and the gazetteer are rebuilt from the other fold for OOF docs and from all 70 docs for test. Test ensemble = the same 4 fold models (one frozen S1 + one frozen S3 backbone shared by the PEFT members). |
| `n20_oof_refit.yaml` | N20 | submitted | N19's OOF-chosen decisions applied to a larger test ensemble: the 4 fold models + LoRMA+ refits on all 70 docs (2 seeds) + hybrid LoRA refit on all 70 docs. Still two frozen Phase-1 backbones, so the deployed size stays < 300M. |
| `n21_oof_conf.yaml` | N21 | submitted | N20 (OOF-calibrated, 4 fold models + 3 all-docs refits) plus a per-label span-confidence threshold chosen out-of-fold: predicted spans whose mean ensemble tag probability is below the label's threshold are dropped. |
| `n22_oof_wide.yaml` | N22 | submitted | wide bagged OOF system. Out-of-fold predictions now average 5 models per doc (LoRMA+ x2 seeds and LoRA on S1, hybrid LoRA x2 seeds on S3); test = all 10 fold models + 4 all-docs refits. Every member is a PEFT module on one of two frozen Phase-1 backbones (S1, S3). Bias, confidence thresholds and rules chosen out-of-fold as N21. |
| `n23_oof_doc.yaml` | N23 | submitted | N21 + document-level rules chosen out-of-fold: prosecution-witness cue (PW-n -> WIT), two-stage majority label over case-folded mentions (Krishnan & Manning 2006), and case-insensitive consistency (cause-title capitals <-> body mentions; Kalamkar et al. 2022). Each rule is kept only where it raises OOF F1 (per label where possible). |
| `n24_oof_wide_doc.yaml` | N24 | submitted | N22's wide bagged ensemble (10 fold PEFT models + 4 all-docs refits on shared S1/S3) + N23's document rules. |
| `n25_oof_header.yaml` | N25 | submitted | N24 (wide bagged OOF ensemble + document rules) + cause-title party parsing (names before/after "Versus" in the preamble and final order -> APP/RESP; Kalamkar et al. 2022), counsel-side fix from the following role word, and a finer per-label bias grid (3 passes). Every step is kept only if it raises out-of-fold F1. |
| `n26_oof_diverse.yaml` | N26 | submitted | N24 (best: 0.6241) with four more already-trained PEFT members on the same frozen backbones added to the test ensemble only (method diversity: LoRMA+ N2, hybrid LoRA N3, IA3 N13, LoRMA-pi N15; trained on 60 docs). Calibration (bias, thresholds, rules) is still chosen out-of-fold exactly as in N24. |
| `n161_lorma_fold1.yaml` | N161 | member | 2-fold CV member: N2's LoRMA+ setup trained on fold1, epoch selected on all 35 docs of fold2. |
| `n162_lorma_fold2.yaml` | N162 | member | 2-fold CV member: N2's LoRMA+ setup trained on fold2, epoch selected on all 35 docs of fold1. |
| `n171_hyb_fold1.yaml` | N171 | member | 2-fold CV member: N3's hybrid LoRA setup trained on fold1, epoch selected on all 35 docs of fold2. |
| `n172_hyb_fold2.yaml` | N172 | member | 2-fold CV member: N3's hybrid LoRA setup trained on fold2, epoch selected on all 35 docs of fold1. |
| `n201_lorma_all.yaml` | N201 | member | N20 member: N2's LoRMA+ setup refit on all 70 training docs (epoch count = mean best epoch of the fold runs). |
| `n202_lorma_all_s2.yaml` | N202 | member | N20 member: as N201 with another seed (bagging; adds only the 2.2M LoRMA+ parameters on the shared S1 backbone). |
| `n203_hyb_all.yaml` | N203 | member | N20 member: N3's hybrid LoRA setup refit on all 70 training docs (epoch count = mean best epoch of the fold runs). |
| `n221_lorma_s2_fold1.yaml` | N221 | member | N22 member: N161 (LoRMA+ on S1, trained on fold1) with another seed. |
| `n222_lorma_s2_fold2.yaml` | N222 | member | N22 member: N162 (LoRMA+ on S1, trained on fold2) with another seed. |
| `n223_gptlora_fold1.yaml` | N223 | member | N22 member: LegalGPT (S1) + LoRA r8/alpha16 (unmasked), trained on fold1, epoch selected on fold2. |
| `n224_gptlora_fold2.yaml` | N224 | member | N22 member: LegalGPT (S1) + LoRA r8/alpha16 (unmasked), trained on fold2, epoch selected on fold1. |
| `n225_hyb_s2_fold1.yaml` | N225 | member | N22 member: N171 (hybrid LoRA on S3, trained on fold1) with another seed. |
| `n226_hyb_s2_fold2.yaml` | N226 | member | N22 member: N172 (hybrid LoRA on S3, trained on fold2) with another seed. |
| `n227_gptlora_all.yaml` | N227 | member | N22 member: S1 + LoRA r8/alpha16 refit on all 70 docs (epoch count = mean best epoch of its fold runs). |
