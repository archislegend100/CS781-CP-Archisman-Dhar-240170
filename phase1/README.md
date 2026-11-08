# Phase 1: Legal language model pretraining

Continue a 512-word legal prompt with exactly 128 words. Four architectures (LegalGPT, LegalSeq, LongLegal, LegalHybrid; 129.7M-145.4M parameters) are trained from scratch on 2B tokens, adapted to Supreme Court text, post-trained with self-ranking DPO and decoded with minimum-Bayes-risk selection over 64 candidates per prompt.

| | |
|---|---|
| Final submission | **Z13** (`phase1/configs/z13_final.yaml`) |
| Codabench | BERTScore-F1 0.8445; rank 10 (35 valid submissions) |
| Compute | 225.5 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/train.py`, `src/models/`, `src/generate.py`, `src/selfrank.py`, `src/pool.py`, `src/submit.py` |
| Needs | the official Phase-1 corpus (downloaded by `setup.sh`) |

## Running

```bash
bash phase1/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase1/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase1/run.sh                           # print the command sequence only
bash phase1/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** continue a 512-word legal prompt with exactly 128 words. **Metric:** BERTScore F1 (primary), ROUGE-L, BLEU-4.

**Rules:** 50M–300M parameters, random initialisation, official four-source corpus only (acts, high courts, Supreme
Court, tribunals).

### 1.1 Data and tokenizer
| method | citation | what we did / where |
|---|---|---|
| Assignment-1 byte-level BPE (5,000 vocab, ▁ word marker) | Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units*, ACL 2016; byte-level variant as in Radford et al. 2019 (GPT-2) | reused as permitted (§7.3), fingerprint `1928c1a635a95dd1`; `tokenizer.py`, `reportnotes.md` |
| Document-level hashed validation split | — (standard leakage control) | blake2b hash of the text, 0.5% of documents; `data.py` |
| Temperature-smoothed source mixing p_i ∝ n_i^α (α = 0.7) | Conneau & Lample, *Cross-lingual LM Pretraining (XLM)*, NeurIPS 2019; Arivazhagan et al. 2019; Chalkidis et al., *LeXFiles and LegalLAMA*, ACL 2023 | `cache.py`; B2 and all core runs |
| From-scratch in-domain pretraining (motivation) | Chalkidis et al., *LEGAL-BERT*, Findings EMNLP 2020; Zheng et al., *When does pretraining help?*, ICAIL 2021; Paul et al., *InLegalBERT*, ICAIL 2023 | `literature_notes.md` §1 |

### 1.2 Architectures (all four are used again in Phase 2)
| model (id) | key methods | citations |
|---|---|---|
| **LegalGPT** (G1), 20×768, 129.7M | pre-norm RMSNorm, RoPE, GQA 12/4, SwiGLU, tied embeddings, residual-scaled initialisation | Xiong et al. ICML 2020; Zhang & Sennrich NeurIPS 2019; Su et al. 2021 (RoPE, arXiv 2104.09864); Ainslie et al. EMNLP 2023 (GQA); Shazeer 2020 (SwiGLU); Press & Wolf EACL 2017; Radford et al. 2019 |
| **LegalSeq** (Q1), encoder–decoder 10+10 | prefix→continuation objective that matches the test task | Raffel et al. JMLR 2020 (T5); Tay et al. 2022 (UL2) |
| **LongLegal** (L1) | sliding-window attention w = 512 with a global layer every 4th layer, 4k context | Beltagy et al. 2020 (Longformer); Jiang et al. 2023 (Mistral); Gemma Team 2024 (Gemma 2); Xiao et al. 2021 (Lawformer) |
| **LegalHybrid** (H1) | order-2 Hyena long-convolution mixers (FFT) + attention every 4th layer, short convolutions on q/k/v | Poli et al. ICML 2023 (Hyena); Fu et al. ICLR 2023 (H3); So et al. NeurIPS 2021 (Primer); Arora et al. ICML 2024 (Based) |
| Deep-thin shape (20×768 over 12×1024) | sub-billion design | Liu et al. ICML 2024 (MobileLLM) |
| Token budget of about 2B for 130M | compute-optimal training | Hoffmann et al. 2022 (Chinchilla) |
| Attention kernels | SDPA flash / memory-efficient kernels | Dao et al. NeurIPS 2022; Dao 2023; Rabe & Staats 2021 |

### 1.3 Ablations
| id | question | citation | BERTScore |
|---|---|---|---|
| A1 / A2 | RoPE vs ALiBi (0.5B tokens) | Su et al. 2021; Press, Smith, Lewis ICLR 2022 | 0.8302 / 0.8319 |
| A3 | GQA vs MHA at matched parameters | Ainslie et al. 2023; Shazeer 2019 (MQA) | 0.8308 |
| B1 / B2 | Assignment-1 BPE; then all four sources with temperature mixing | as 1.1 | 0.8002 / 0.7959 |
| R4 / P1 | hybrid with no attention layers (`hyb_attn0_s`); 70/30 instead of 85/15 Supreme Court replay mix (`legalhybrid_sc70`) | Fu et al. 2023; Gururangan et al. ACL 2020 | — |

### 1.4 Training-side improvements
| id | method | citation | BERTScore |
|---|---|---|---|
| G1 / Q1 / L1 / H1 | core models | above | 0.8352 / 0.8208 / 0.8349 / 0.8357 |
| S1 / S2 / S3 | **domain-adaptive continued pretraining** toward the Supreme Court (test prompts are about 98% Supreme Court, `testsrc.py`), 15% replay of the other sources, re-warmed cosine schedule | Gururangan et al., *Don't Stop Pretraining*, ACL 2020; replay against forgetting: Ibrahim et al. 2024, *Simple and Scalable Strategies to Continually Pre-train LLMs* | — |
| W1 | checkpoint averaging of late checkpoints (`avgckpt.py`) | Izmailov et al. UAI 2018 (SWA); Wortsman et al. ICML 2022 (model soups) | — |
| Z1 / Z3 | **self-ranking** post-training: K = 4 samples, reward ROUGE-L + 0.5·BLEU − repetition, best/worst pairs, DPO with a frozen reference + 0.5·CE regulariser | Rafailov et al. NeurIPS 2023 (DPO); Yuan et al. 2023 (RRHF); Zhao et al. 2023 (SLiC); Welleck et al. ICLR 2020 (unlikelihood motivation) | — |
| Z11 | self-ranking with an **embedding reward** (hidden-state F1, as in BERTScore) | Zhang et al. ICLR 2020 (BERTScore) + DPO | — |

### 1.5 Decoding and inference-time methods
| id | method | citation | BERTScore |
|---|---|---|---|
| D1 | greedy decoding + no-repeat 4-gram | Paulus et al. ICLR 2018 | 0.8358 |
| sampling options | nucleus sampling, repetition penalty | Holtzman et al. ICLR 2020; Keskar et al. 2019 (CTRL) | — |
| E1 | ensemble of two models (mean of log-probs) | — (standard product of experts) | — |
| M1 / M2 / Z5 / Z6 | **minimum-Bayes-risk (MBR) decoding** over greedy + samples, lexical utility | Kumar & Byrne NAACL 2004; Eikema & Aziz COLING 2020; Freitag et al. TACL 2022 | — |
| F1 / F2 / F3, Z7 / Z8 | MBR over **pooled candidates** from two models; lexical / embedding / mixed utility | Freitag et al. 2022; Zhang et al. 2020 | — |
| Z9 | **quality-aware MBR**: z(expected utility) + 0.3·z(log-likelihood) | Fernandes et al., *Quality-Aware Decoding for NMT*, NAACL 2022 | — |
| Z10 | multi-model utility (embedding F1 averaged over two models) | Kovacs et al. 2024 (multi-metric / multi-model MBR) | — |
| Z13 | Z10 with utility also averaged over three layers (10, 13, 16): **final submission** | Zhang et al. ICLR 2020 (BERTScore is layer-sensitive) | **0.8445** |

### 1.6 Reproducibility and hardware notes
- T4 *inference* was measured on Kaggle (Phase 2, see 2.7).

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | shared defaults |
| `e1_ens.yaml` | E1 | submitted | ensemble of S2 (LegalHybrid-SC, last) and Z3 (self-ranked LegalGPT-SC, best); mean of member log-probs. |
| `f1_lex.yaml` | F1 | submitted | MBR over pooled Z3 + S2 candidates (2 x 16), lexical utility ROUGE-L + 0.5 BLEU. |
| `f2_emb.yaml` | F2 | submitted | MBR over pooled Z3 + S2 candidates, BERTScore-style utility on Z3 hidden states. |
| `f3_mix.yaml` | F3 | submitted | MBR over pooled Z3 + S2 candidates, utility ROUGE-L + 0.5 BLEU + EmbF1. |
| `gpt_alibi_s.yaml` | A2 | submitted | Position ablation: identical to gpt_rope_s except ALiBi replaces RoPE (both parameter-free). |
| `gpt_decode.yaml` | D1 | submitted | Decoding experiment: the trained LegalGPT checkpoint, deterministic decoding instead of sampling. |
| `gpt_mha_s.yaml` | A3 | submitted | Attention ablation: MHA (kv = q = 12 heads) vs gpt_rope_s GQA (kv = 4). ff 2048 -> 1728 keeps parameters matched (130.68M vs 129.70M), so the comparison is at equal size and budget. |
| `gpt_rope_s.yaml` | A1 | submitted | Ablation control: LegalGPT (RoPE, GQA 12/4) at the shared ablation budget of 0.5B tokens. |
| `hyb_attn0_s.yaml` | R4 | submitted | Attention-ratio ablation: no attention at all (pure Hyena, attnevery > layers), ff 1728 (132.0M). |
| `legalbpe.yaml` | X4 | member | Optional legal-domain BPE (disabled by default). Train on the remote server only: python tokenizer.py phase1/configs/legalbpe.yaml train |
| `legalgpt.yaml` | G1 | submitted |  |
| `legalgpt_sc.yaml` | S1 | submitted | Domain-adaptive continued pretraining of G1 toward the test distribution (test prompts are ~98% Supreme Court, testsrc.py). Starts from G1's weights with a fresh optimizer and a re-warmed cosine schedule; 15% replay of the other sources limits forgetting. Official training data only. |
| `legalhybrid.yaml` | H1 | submitted |  |
| `legalhybrid_sc2.yaml` | S3 | submitted | second Supreme-Court adaptation stage for the hybrid: S2 -> +0.4B tokens (85/15), re-warmed to 1e-4. |
| `legalhybrid_sc70.yaml` | P1 | submitted | Replay-ratio ablation: H1 adapted with SC 0.70 / 0.10 each other source (S2 uses 0.85 / 0.05). |
| `legalhybrid_sc.yaml` | S2 | submitted | Same domain-adaptive continued pretraining for H1 (LegalHybrid). |
| `legalseq.yaml` | Q1 | submitted |  |
| `longlegal.yaml` | L1 | submitted |  |
| `m1_mbr.yaml` | M1 | submitted | S2 decoded with minimum-Bayes-risk selection over greedy + 8 samples (utility ROUGE-L + 0.5 BLEU). |
| `m2_ensmbr.yaml` | M2 | submitted | E1 ensemble + MBR over greedy + 8 samples. |
| `selfrank.yaml` | Z1 | submitted | self-ranking post-training of S1 (LegalGPT-SC, last checkpoint) with the pairwise ranking loss. The run never overwrites its base run; the pairs file is reused by Z3 and Z11. |
| `selfrank_emb.yaml` | Z11 | submitted | self-ranking with an embedding reward. The existing S1 candidate pairs are re-ranked by S1's own hidden-state F1 to the gold continuation (BERTScore-style), then standard DPO (summed log-probs) as in Z3. |
| `selfrank_hybrid.yaml` | Z2 | submitted | self-ranking on S2 (LegalHybrid-SC, last checkpoint) with the pairwise ranking loss; Supreme Court prompts, greedy decoding. Separate run directory, so the S2 run is never modified. |
| `starter.yaml` | B0 | member | BASE-0: official starter reproduced with the custom trainer (reference only; starter predictions do not count toward the submission minimum). |
| `starter_bpe.yaml` | B1 | submitted | BASE-1: starter family with the Assignment-1 BPE. With /V/ = 5000 the embedding shrinks by 25.3M and 8 layers would give 34.9M (< 50M), so depth goes 8 -> 13 (52.96M). Everything else as BASE-0, except bf16 in place of fp16 (same hardware path as all later runs). |
| `starter_improved.yaml` | B2 | submitted | BASE-2: BASE-1 plus one change: train on all four sources with temperature mixing (alpha 0.7) instead of Supreme Court only. Same model, tokens, schedule and decoding. |
| `w1_avg.yaml` | W1 | submitted | S2 with averaged late checkpoints (run avgckpt.py first, which writes runs/legalhybrid_sc/ckpt/avg.pt). |
| `z3_selfrank_dpo.yaml` | Z3 | submitted | Z1 with the DPO objective (reference = S1), same pairs; greedy decoding of the best checkpoint. |
| `z4_selfrank_hybrid_dpo.yaml` | Z4 | submitted | Z2 with the DPO objective (reference = S2), same pairs. |
| `z5_mbr.yaml` | Z5 | submitted | Z3's self-ranked checkpoint decoded with MBR over greedy + 8 samples. |
| `z6_ensmbr.yaml` | Z6 | submitted | Z3 (main) + S2 ensemble, MBR over greedy + 16 samples. |
| `z7_pool.yaml` | Z7 | submitted | MBR over pooled Z3 (self-ranked) + S3 candidates, mixed utility. |
| `z8_pool.yaml` | Z8 | submitted | MBR over 64 candidates/prompt (Z3 + S3, 32 each, two temperatures), embedding utility (F2 won). |
| `z9_qmbr.yaml` | Z9 | submitted | quality-aware MBR (Fernandes et al. 2022): z(expected embedding utility) + 0.3 z(mean log-lik under Z3 and S3). |
| `z10_mmu.yaml` | Z10 | submitted | multi-model utility (cf. Kovacs et al. 2024): embedding F1 averaged over Z3 and S3 hidden states. |
| `z12_pool.yaml` | Z12 | submitted | MBR over Z11 (embedding-reward self-ranked) + S3 candidates, 64/prompt, embedding utility on Z11. |
| `z13_final.yaml` | Z13 | submitted | Z10 (64-candidate Z3 + S3 pool, embedding utility averaged over both models) with the utility also averaged over three depths (layers 10, 13, 16 of 20; BERTScore is layer-sensitive, Zhang et al. 2020). |
