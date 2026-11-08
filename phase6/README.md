# Phase 6: Abstractive legal summarization

Summarize long Indian judgments. Each Phase-1 model learns chunk-to-chunk-summary generation from sentence-aligned supervision; summaries are selected by document-level MBR across seven systems and topped up with salient source sentences to a held-out-tuned length (S13).

| | |
|---|---|
| Final submission | **S13** (`phase6/configs/s13_hyb_mbr.yaml`) |
| Codabench | BERTScore-F1 0.8468; rank 33 (23 valid submissions) |
| Compute | 15.2 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/sm.py`, `src/peft.py` |
| Needs | Phase-1 checkpoints S1, S3, L1, Q1 |

## Running

```bash
bash phase6/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase6/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase6/run.sh                           # print the command sequence only
bash phase6/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** generate a summary per judgment. **Metric:** BERTScore F1 (ROUGE-1 / ROUGE-L secondary).

**Data and rules:**
- No dev split is provided, so 100 training documents are held out.
- Code: `src/sm.py`, `phase6/configs/`, `phase6/run.sh`.
- Literature context: on IN-Abs (7,030 / 100 Indian Supreme Court judgments with abstractive summaries), strong supervised
  and LLM systems reach BERTScore F1 ≈ 0.86.

### 6.1 Long documents, decoding, post-processing
| component | citation | where |
|---|---|---|
| **Chunk-level summarization with sentence-aligned supervision**: sentence-aligned chunks (1280 tokens; 2560 for LongLegal, 1024 for LegalSeq); every reference sentence assigned to the chunk with the largest content-word overlap → chunk → chunk-summary pairs; chunk summaries concatenated in document order | Shukla et al., *Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation*, AACL 2022 (IN-Abs); Gidiotis & Tsoumakas, *A Divide-and-Conquer Approach to the Summarization of Long Documents*, IEEE TASLP 2020 | `chunkdoc`, `align` |
| Empty-target chunks down-sampled to 25% in training, plus a minimum chunk-summary length at decoding (no early stop) | min-length constraint in summarization decoding | `examples`, `generate` |
| Greedy decoding with KV cache, no-repeat-4-gram, repetition penalty 1.1 | Paulus et al. 2018; Keskar et al. 2019 | `generate` |
| **Cross-chunk trigram blocking** (redundancy reduction) | Paulus et al., ICLR 2018; Liu & Lapata, EMNLP 2019 | `assemble` |
| **Document word budget** = multiplier × median summary/document ratio × document words, multiplier tuned on the held-out split | PS §12.5 (generation length vs metrics) | `finish` |
| **Quality-filtered, quality-weighted document-level MBR** over member summaries (ROUGE-1 + ROUGE-L utility) | Kumar & Byrne, NAACL 2004; Freitag et al., TACL 2022 | `mbr` |

### 6.2 Submissions
| id | Phase-1 model | technique |
|---|---|---|
| S1 | LegalGPT-SC | **LoRMA+** (compulsory), r = α = 16 |
| S2 | LegalHybrid-SC | **Houlsby adapters** |
| S3 | LongLegal | **QLoRA** (NF4), 2560-token chunks / 1024-token chunk summaries |
| S4 | LegalSeq encoder-decoder | **LoRA** on encoder, decoder self- and cross-attention, FFNs |
| S5 | LegalGPT-SC | **Pfeiffer adapters** (after FFN only) |
| S6 | LegalGPT-SC | **AdaLoRA** |
| S7 | first 8 of 20 LegalGPT-SC blocks | **knowledge distillation**, word-level KD from S1 (α·CE + (1−α)·T²·KL; Kim & Rush, EMNLP 2016) |
| S8 | — | MBR ensemble of S1–S7 |
| S9 | S1's model | ablation: first chunk only (coverage) |

### 6.3 Results (round 1) and round 2
| id | held-out ROUGE-1 / ROUGE-L | test mean words | test BERTScore |
|---|---|---|---|
| S1 LoRMA+ | 0.4072 / 0.2316 | 325 | — |
| S2 Houlsby | 0.4033 / 0.2262 | 318 | — |
| S3 QLoRA LongLegal | 0.3914 / 0.2178 | 357 | — |
| S4 LegalSeq LoRA | 0.3182 / 0.1838 | 266 | — |
| S5 Pfeiffer | 0.3818 / 0.2129 | 311 | — |
| S6 AdaLoRA | 0.3706 / 0.2103 | 289 | — |
| S7 KD student | 0.3691 / 0.2065 | 285 | — |
| S8 MBR (picks S1 60%) | 0.3920 / 0.2267 | — | **0.8457** |
| S9 first chunk only | 0.1698 / 0.1151 | 85 | — (coverage ablation) |

The chosen length budgets reach the generated length: the models stop on their own end token before the budget, so
the summaries are likely shorter than the references.

| id | method | citation |
|---|---|---|
| S10–S13 | **Abstract-then-extract fill** on S1 / S2 / S3 / S8: the generated summary is the backbone, topped up to a held-out-tuned word budget with the most salient document sentences (content overlap with the generated summary + opening/closing position prior), in document order, trigram-blocked | Pilault et al., EMNLP 2020; Extract-Explain-Abstract (NLLP 2025); Shukla et al. 2022 (extractive strength on IN-Abs) |
| S14 | quality-weighted MBR over S10–S13 | Kumar & Byrne 2004; Freitag et al. 2022 |

**Round 2 results:**
- Fill raised held-out ROUGE-1 from 0.39 to 0.55 and ROUGE-L from 0.23 to 0.29; summaries became reference-length (~880 words).
- Test BERTScore moved only from 0.8457 (S8) to 0.8468.
- So length was not what limited BERTScore. The fluency and faithfulness of the generated text is the likely limit,
  and ROUGE cannot see it.

### 6.4 Round 3: tuning on a local BERTScore proxy
| id | method | citation |
|---|---|---|
| S15 / S16 | Hybrid on S8 / S1 tuned on a **local BERTScore proxy** instead of ROUGE: greedy cosine-matching F1 of token states of the frozen Phase-1 LegalGPT-SC (layers 10 / 13 / 16, no idf; our Phase-1 embedding utility). Searched: **faithfulness filter** τ on generated sentences (content overlap with the source), fraction of generated text kept, word budget | Zhang et al., *BERTScore*, ICLR 2020; Maynez et al., *On Faithfulness and Factuality in Abstractive Summarization*, ACL 2020 |
| S17 | S15 restricted to ≥ 25% generated text (keeps an abstractive core) | PS §12.1 (abstractive task) |

The proxy is calibrated by printing it for the S8 and S13 configurations, whose official scores are known (0.8457 / 0.8468).

### 6.5 Round 4: BERTScore's 512-token window
None of S15–S17 beat 0.8468.

**Diagnosis:** the official `bert_score` (roberta-large, layer 17) truncates each text to the model's 512-token limit
(≈ 400 words). So only the beginning of the summary and of the reference are compared.
- Round 2 appended extracted sentences after ~325 generated words, so they mostly fell outside the window: ROUGE rose
  (+0.16 ROUGE-1), BERTScore barely moved.
- The round-3 proxy compared up to 3,072 tokens, so it was misaligned with the metric.

| id | method | citation |
|---|---|---|
| S18 / S19 / S20 | **Chunk-interleaved hybrid** on S1 / S2 / S3: chunk by chunk in document order, faithfulness-filtered generated chunk summary then the chunk's most salient sentences, within a per-chunk word allowance (which also sets how far into the judgment the first 400 words reach); trigram blocking | Zhang et al., *BERTScore*, ICLR 2020 (512-token encoder, layer 17); Shukla et al. 2022; Maynez et al. 2020 |
| selection | held-out selection with the **official scorer** (local roberta-large, the organisers' evaluator, selection only) when available; otherwise the Phase-1 proxy truncated to the first 400 words | — |

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | Phase 6 (abstractive legal summarization) defaults. Generators are our Phase-1 checkpoints only. |
| `s1_gpt_lorma.yaml` | S1 | submitted | LegalGPT-SC + LoRMA+ (compulsory; r = alpha = 16 on all projections), chunk-level summarization with sentence-aligned chunk targets, causal decoder kept causal. |
| `s2_hyb_houlsby.yaml` | S2 | submitted | LegalHybrid-SC + Houlsby adapters (after every mixer / attention / FFN sublayer). |
| `s3_long_qlora.yaml` | S3 | submitted | LongLegal + QLoRA (NF4 4-bit frozen base, LoRA r16 / alpha 32): twice-longer chunks (2560 tokens) with 1024-token chunk summaries in its 4096-token context -> fewer chunk boundaries (does chunking lose information?). |
| `s4_seq_lora.yaml` | S4 | submitted | LegalSeq encoder-decoder + LoRA (r16 / alpha 32) on encoder self-attention, decoder self- and cross-attention and FFNs; 1024-token source chunks, 448-token chunk summaries (its pretraining lengths: 1280 / 448). |
| `s5_gpt_pfeiffer.yaml` | S5 | submitted | LegalGPT-SC + Pfeiffer adapters (one bottleneck adapter after each FFN only; Pfeiffer et al., EACL 2021). |
| `s6_gpt_adalora.yaml` | S6 | submitted | LegalGPT-SC + AdaLoRA (SVD-form updates, rank 12 -> average 4 by importance, orthogonality penalty 0.1). |
| `s7_kd.yaml` | S7 | submitted | knowledge distillation (Hinton et al. 2015; word-level KD for generation, Kim & Rush 2016): student = first 8 of 20 LegalGPT-SC blocks (~54M), fully fine-tuned on alpha * CE(reference) + (1 - alpha) * T^2 * KL(teacher // student) over the summary tokens; teacher = S1 (LoRMA+ LegalGPT-SC, same tokenisation). |
| `s8_mbr.yaml` | S8 | submitted | document-level minimum-Bayes-risk selection over the summaries of S1-S7 (consensus candidate by ROUGE-1 + ROUGE-L). |
| `s9_lead_ablation.yaml` | S9 | submitted | S1's model summarising only the FIRST chunk of each judgment -> how much does covering every chunk add? |
| `s10_hyb_s1.yaml` | S10 | submitted | abstract-then-extract fill on S1 (LoRMA+): generated summary kept as the backbone, topped up to a held-out-tuned word budget with the most salient document sentences (overlap with the generated summary + opening/closing position prior), in document order, trigram-blocked. No new training or generation (rebuilt from the saved outputs). |
| `s11_hyb_s2.yaml` | S11 | submitted | abstract-then-extract fill on S2 (Houlsby adapters): generated summary kept as the backbone, topped up to a held-out-tuned word budget with the most salient document sentences (overlap with the generated summary + opening/closing position prior), in document order, trigram-blocked. No new training or generation (rebuilt from the saved outputs). |
| `s12_hyb_s3.yaml` | S12 | submitted | abstract-then-extract fill on S3 (LongLegal QLoRA): generated summary kept as the backbone, topped up to a held-out-tuned word budget with the most salient document sentences (overlap with the generated summary + opening/closing position prior), in document order, trigram-blocked. No new training or generation (rebuilt from the saved outputs). |
| `s13_hyb_mbr.yaml` | S13 | submitted | abstract-then-extract fill on the S8 MBR summaries: generated summary kept as the backbone, topped up to a held-out-tuned word budget with the most salient document sentences (overlap with the generated summary + opening/closing position prior), in document order, trigram-blocked. No new training or generation (rebuilt from the saved outputs). |
| `s14_mbr_hybrid.yaml` | S14 | submitted | quality-weighted document-level MBR over the hybrid summaries (S10-S13). |
| `s15_proxy_mbr.yaml` | S15 | submitted | hybrid on the S8 MBR summaries tuned on a local BERTScore proxy (greedy-matching F1 of token states of the frozen Phase-1 LegalGPT-SC, layers 10/13/16, as our Phase-1 embedding utility) instead of ROUGE: faithfulness filter on generated sentences (tau), fraction of generated text kept, summary-guided extractive fill to the word budget. |
| `s16_proxy_s1.yaml` | S16 | submitted | hybrid on S1 (LoRMA+) tuned on a local BERTScore proxy (greedy-matching F1 of token states of the frozen Phase-1 LegalGPT-SC, layers 10/13/16, as our Phase-1 embedding utility) instead of ROUGE: faithfulness filter on generated sentences (tau), fraction of generated text kept, summary-guided extractive fill to the word budget. |
| `s17_proxy_mbr_abs.yaml` | S17 | submitted | hybrid on the S8 MBR summaries tuned on a local BERTScore proxy (greedy-matching F1 of token states of the frozen Phase-1 LegalGPT-SC, layers 10/13/16, as our Phase-1 embedding utility) instead of ROUGE: faithfulness filter on generated sentences (tau), fraction of generated text kept, summary-guided extractive fill to the word budget. |
| `s18_win_s1.yaml` | S18 | submitted | chunk-interleaved hybrid on S1 (LoRMA+), built for BERTScore's 512-token window (only ~the first 400 words of candidate and reference are compared): chunk by chunk in document order, the faithfulness-filtered generated summary then the chunk's most salient sentences within a per-chunk allowance. Settings chosen on the held-out split by the official scorer (local roberta-large, selection only) if present, else by the Phase-1 proxy on the first 400 words. |
| `s19_win_s2.yaml` | S19 | submitted | chunk-interleaved hybrid on S2 (Houlsby adapters), built for BERTScore's 512-token window (only ~the first 400 words of candidate and reference are compared): chunk by chunk in document order, the faithfulness-filtered generated summary then the chunk's most salient sentences within a per-chunk allowance. Settings chosen on the held-out split by the official scorer (local roberta-large, selection only) if present, else by the Phase-1 proxy on the first 400 words. |
| `s20_win_s3.yaml` | S20 | submitted | chunk-interleaved hybrid on S3 (LongLegal QLoRA), built for BERTScore's 512-token window (only ~the first 400 words of candidate and reference are compared): chunk by chunk in document order, the faithfulness-filtered generated summary then the chunk's most salient sentences within a per-chunk allowance. Settings chosen on the held-out split by the official scorer (local roberta-large, selection only) if present, else by the Phase-1 proxy on the first 400 words. |
