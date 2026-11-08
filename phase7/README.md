# Phase 7: Prior case retrieval (PCR)

Return exactly 7 cited prior cases per query from the split's candidate pool. BM25 and dense Phase-1 retrievers (InfoNCE with BM25 hard negatives) are combined by a cross-fitted listwise learning-to-rank model whose features include hubness-corrected scores, chunk MaxSim, query-graph propagation and a chunk-pair cross-encoder.

| | |
|---|---|
| Final submission | **P26** (`phase7/configs/p26_ltr_ce.yaml`) |
| Codabench | micro-F1@7 0.4589; rank 26 (31 valid submissions) |
| Compute | 36.6 T4 GPU-hours for all submissions of the phase (see the [root README](../README.md#compute)) |
| Code | `src/pcr.py`, `src/pcr2.py`, `src/pcr3.py`, `src/pcr4.py`, `src/pcr5.py` |
| Needs | Phase-1 checkpoints S1, S3, L1, Q1 |

## Running

```bash
bash phase7/setup.sh                                  # data download and checks
CUDA_VISIBLE_DEVICES=0 bash phase7/run.sh             # every submission of this phase, in order
DRYRUN=1 bash phase7/run.sh                           # print the command sequence only
bash phase7/run.sh one <script> <config> <mode> [key=value ...]   # one step by hand
```

Each submission is written to `submissions/<ID>_<name>/` (`prediction.json`, `submission.zip` with the file at the
archive root, `meta.json`); models go to `runs/`, logs to `logs/<config>.log`. Finished steps are skipped on a re-run.

## Method

**Task:** for each query case, output exactly 7 unique candidate IDs from its split's pool. **Metric:** micro-F1@7.
Code: `src/pcr.py`, `phase7/configs/`, `phase7/run.sh`. Dataset: IL-PCR (Joshi et al., *U-CREAT*, Findings ACL 2023).

Note on the metric: with exactly 7 outputs, P@7 is capped by the number of relevant cases per query (`prep` reports
the gold-set sizes), so the best possible F1@7 is below 1 for most queries.

### 7.1 Method
| component | citation | where |
|---|---|---|
| **BM25 over word uni+bigrams** (stop words removed, k1 1.2, b 0.75): the lexical retriever and source of hard negatives; also reported alone as a baseline (not a submission, since it does not use a Phase-1 model) | Robertson & Zaragoza 2009; Joshi et al. 2023 (n-gram BM25 is a strong IL-PCR baseline) | `BM25` |
| **Dense bi-encoder** from a Phase-1 model (causal mask removed): mean-pooled 512-token chunk embeddings; a document is the normalised mean of up to 4 evenly spaced chunks | Dai & Callan, SIGIR 2019 (MaxP / long-document passages); Dukić & Šnajder 2024 | `Encoder`, `embed` |
| **InfoNCE** with in-batch negatives + **BM25 hard negatives**, learned temperature; other relevant candidates of the same query masked out of the negatives | Karpukhin et al., *DPR*, EMNLP 2020; Xiong et al., *ANCE*, ICLR 2021 | `train` |
| **Hybrid fusion**: per-query z-normalised dense + BM25 scores, weight tuned on dev F1@7 | Luan et al., TACL 2021; Ma et al. 2021 (hybrid sparse-dense retrieval) | `fusetune` |
| **Multi-retriever fusion** (P8): BM25 + all dense members, greedy weight search on dev | Cormack et al. 2009 (fusion); Bruch et al. 2023 | `fuse` |

### 7.2 Submissions
| id | Phase-1 model | technique |
|---|---|---|
| P1 | LegalGPT-SC | **LoRMA+** (compulsory) |
| P2 | LegalHybrid-SC | **IA³** |
| P3 | LongLegal | **QLoRA**, 2048-token chunks × 2 |
| P4 | LegalSeq encoder | **partial / layer-wise fine-tuning** (top 4 of 10 layers) |
| P5 | LegalGPT-SC | **full fine-tuning** |
| P6 | LegalGPT-SC | **prompt tuning** (32 soft tokens) |
| P7 | first 8 of 20 LegalGPT-SC blocks (≈54M) | **knowledge distillation** of the P1 teacher's similarity distribution over batch candidates (listwise KL + InfoNCE) | 
| P8 | — | BM25 + P1–P7 fusion |

### 7.3 Round 1 results and round 2 (P8 reached 0.3751 test F1@7; leaderboard top ≈ 0.51)
| id | dev dense F1@7 | dev fused F1@7 (w_bm25) |
|---|---|---|
| P1 LoRMA+ | 0.3302 | 0.3880 (0.30) |
| P2 IA³ hybrid | 0.3146 | 0.3735 (0.40) |
| P3 QLoRA LongLegal | 0.3102 | 0.3758 (0.40) |
| P4 LegalSeq partial | 0.2023 | 0.3335 (0.35) |
| P5 full | 0.3280 | 0.3847 (0.25) |
| P6 prompt | 0.2835 | 0.3646 (0.40) |
| P7 KD student (54M) | 0.3057 | 0.3724 (0.40) |
| P8 fusion | — | 0.3947 (test 0.3751) |

- Dense retrievers alone come within 0.06 of the fused systems.
- The 54M student keeps 93% of the teacher's dense F1@7.
- P ≈ 0.42 and R ≈ 0.36 imply about 8 relevant cases per query.

| id | method | citation |
|---|---|---|
| P9 | Re-ranking (no training): **TF-IDF uni+bigram cosine** as a second lexical view; **CSLS hubness correction** (candidates scoring high for every query of the pool are penalised by their mean top-10 score over the split's queries; transductive, no labels); weighted fusion; **candidate-graph pseudo-relevance feedback** s'(q,c) = s(q,c) + β·mean_{c'∈top-m(q)} G(c',c) with G from dense embeddings + TF-IDF; all tuned on dev, 5-fold honest estimate over dev queries | Conneau et al., ICLR 2018 (CSLS); Smith et al., ICLR 2017 (inverted softmax); Radovanović et al., JMLR 2010 (hubness); Rocchio 1971; Lavrenko & Croft 2001 (relevance models); Joshi et al. 2023 (n-gram lexical matching for IL-PCR) |
| P10 | P9 with uni+bi+trigrams | — |

### 7.4 Round 3 (P9 reached **0.4371** test F1@7)
| id | method | citation |
|---|---|---|
| P12 | **Cross-fitted listwise learning-to-rank** over all retrieval signals of P9's top-100 per query. Features: per view z-score, hubness-corrected score, **reverse rank** (mutual-nearest-neighbour evidence); PRF feedback per graph; P9 score / rank; candidate hubness. MLP scorer, multi-positive ListNet softmax loss, 5-fold cross-fitting over dev queries (dense members' train scores are over-confident, so the ranker is trained on dev) with a 3-seed bag; test = mean of the fold models; blend with P9 chosen on cross-fitted scores | Cao et al., *ListNet*, ICML 2007; Burges 2010 (LTR); Jégou et al. 2007 / Qin et al. 2011 (reciprocal / mutual kNN re-ranking); Chernozhukov et al. 2018 |
| P11 | Stronger **LoRMA+** retriever: rank 16, 3 BM25 hard negatives from a 50-deep pool per query, 6 chunks per document, 6 epochs | Xiong et al., *ANCE*, ICLR 2021; Bihany et al. 2025 |
| P13 / P14 | P9 re-ranker / P12 LTR with P11 added | — |

### 7.5 Round 4 (P14 reached **0.4483**; P12/P14 cross-fitted LTR)
| id | method | citation |
|---|---|---|
| P15 | P14's cross-fitted LTR plus three new signal families. (1) **Chunk-level late interaction (MaxSim)**: every 512-token chunk of query and candidate embedded separately by a trained retriever (P11, P5); score = ½[mean over query chunks of the best candidate chunk + mean over candidate chunks of the best query chunk]. (2) **Query-graph relevance propagation**: all queries of a split share one pool, and similar cases cite the same precedents, so a candidate's z-score is propagated from each query's 10 nearest other queries (by query embeddings, and by agreement of their ranking vectors); transductive, no labels. (3) TF-IDF uni+bi+trigram view. 5-seed bag | Khattab & Zaharia, *ColBERT*, SIGIR 2020; Dai & Callan, SIGIR 2019 (MaxP); *Segment First, Retrieve Better* (2025, segment-level legal queries); *IL-PCSR*, EMNLP 2025 (graph-based ensembles over the case graph); collaborative filtering (Sarwar et al. 2001); Joshi et al. 2023 (U-CREAT) |
| P16 | P15 with depth 150 and 20 query neighbours | — |

### 7.6 Round 5 (P16 reached **0.4525**)
| id | method | citation |
|---|---|---|
| P17 / P19 | **Multiple-instance late-interaction retrievers**, trained end-to-end for chunk MaxSim: each query / candidate is represented by 3 random 512-token chunks, the pair score is the symmetric chunk MaxSim inside InfoNCE (in-batch + 2 BM25 hard negatives), and retrieval uses MaxSim over up to 6 chunks. LoRMA+ r16 on LegalGPT-SC (P17) and LegalHybrid-SC (P19). Earlier retrievers were trained on single chunks but read as mean-of-chunks, so training and inference did not match | Khattab & Zaharia, *ColBERT*, SIGIR 2020; Ilse et al., ICML 2018 (multiple-instance learning); Dai & Callan 2019 |
| P18 / P20 | P16's cross-fitted LTR + the P17 (/P19) MaxSim scores as views + **2-hop query-graph propagation** | Zhou et al., NIPS 2004 (label propagation on graphs) |
| P21 / P22 | Ranker variants for diversity (depth 200 / 75, 5 / 15 query neighbours, MLP width, different seeds) | — |
| P23 / P24 | **Ensemble of cross-fitted rankers**: their dev scores are out-of-fold, so reciprocal-rank fusion or z-score averaging and member weights are chosen honestly on dev. P23 = P15/P16/P21/P22; P24 adds P18/P20 | Cormack et al., SIGIR 2009 (RRF); Dietterich 2000 (ensembles) |

### 7.7 Round 6 (P21 reached **0.4572**)
| id | method | citation |
|---|---|---|
| P25 | **Chunk-pair cross-encoder re-ranker** on LegalGPT-SC + LoRMA+ (r16, causal mask removed). The P11 retriever selects the 2 most similar chunk pairs of each (query, candidate); each pair is cross-encoded as [query half \| EOS \| candidate half] (512 tokens); the pair score is the max over the chunk pairs. Trained with localized contrastive estimation (1 relevant vs 7 hard negatives from the fused BM25 + dense train ranking). Re-scores P21's top-50; fusion weight with P21 chosen on dev | Nogueira & Cho 2019 (monoBERT); Dai & Callan 2019 (MaxP); Li et al., *PARADE*, TOIS 2023; Gao et al., ECIR 2021 (LCE); IL-PCSR, EMNLP 2025 (re-ranker over hybrid retrieval) |
| P26 | P21's cross-fitted ranker + the cross-encoder scores (and the MaxSim retrievers) as views | — |
| P27 | Final ensemble of the cross-fitted rankers + the cross-encoder run | Cormack et al. 2009 |

### 7.8 Final round (P26 **0.4589** test; P27 dev 0.4792)
| id | method |
|---|---|
| P28 | Cross-encoder v2: 4 chunk pairs per pair, 15 hard negatives, LoRMA+ r32, re-scores P26's top-100 |
| P29 / P30 | Ranker with both cross-encoders as views / final ensemble of all rankers and cross-encoders |

Test results: P26 (single cross-fitted ranker with the cross-encoder view) **0.4589** > P27 (dev-weighted ensemble) —
the ensemble overfit dev. Its weights were greedy-tuned over 7 members, and 2 members' dev scores were not out-of-fold
(the cross-encoder's fusion weight was tuned on dev). P28 trains 2 epochs; P30 ensembles cross-fitted rankers only;
P29 is the primary final candidate.

## Configurations

`submitted` marks a configuration that produced a Codabench submission; `member` marks a model that only feeds an
ensemble or a cross-fitted system. Run names given on the command line (fold models such as `*_f0`, `*_df0`, all-docs
refits `*_all`) reuse a base configuration with overrides; `run.sh` lists them.

| config | id | role | description |
|---|---|---|---|
| `base.yaml` | — |  | Phase 7 (prior case retrieval) defaults. Encoders are our Phase-1 checkpoints only; BM25 is a classical lexical scorer. |
| `p1_gpt_lorma.yaml` | P1 | submitted | LegalGPT-SC + LoRMA+ (compulsory) dense retriever: InfoNCE over in-batch + BM25 hard negatives, mean-of-chunks document embeddings, dev-tuned fusion with BM25. |
| `p2_hyb_ia3.yaml` | P2 | submitted | LegalHybrid-SC + IA3 (learned rescaling of keys/values / Hyena inputs and FFN activations). |
| `p3_long_qlora.yaml` | P3 | submitted | LongLegal + QLoRA (NF4 frozen base, LoRA r16/alpha32) with 2048-token chunks x 2 (long-context representation). |
| `p4_seq_partial.yaml` | P4 | submitted | LegalSeq encoder (bidirectional) with partial / layer-wise fine-tuning (top 4 of 10 encoder layers + final norm). |
| `p5_gpt_full.yaml` | P5 | submitted | LegalGPT-SC full fine-tuning dense retriever. |
| `p6_gpt_prompt.yaml` | P6 | submitted | LegalGPT-SC + prompt tuning (32 soft tokens, frozen backbone). |
| `p7_kd.yaml` | P7 | submitted | knowledge distillation -- student = first 8 of 20 LegalGPT-SC blocks (~54M), fully fine-tuned on alpha * InfoNCE + (1 - alpha) * KL(teacher similarity distribution over the batch candidates // student); teacher = P1. |
| `p8_fuse.yaml` | P8 | submitted | multi-retriever fusion -- BM25 + the dense members P1-P7, per-query z-normalised scores, weights tuned on dev F1@7. |
| `p9_rerank.yaml` | P9 | submitted | re-ranking on top of the trained retrievers: TF-IDF uni+bigram cosine view, CSLS hubness correction, weighted fusion (BM25 + TF-IDF + dense members), candidate-graph pseudo-relevance feedback; all tuned on dev, 5-fold honest estimate. |
| `p10_rerank3.yaml` | P10 | submitted | P9 with word uni+bi+trigrams in the TF-IDF view (longer shared legal phrases; cf. n-gram BM25 in U-CREAT). |
| `p11_gpt_lorma2.yaml` | P11 | submitted | stronger LoRMA+ retriever (LegalGPT-SC): 3 BM25 hard negatives per query from a deeper pool (50), 6 evenly spaced chunks per document, 6 epochs, rank 16 (ANCE-style harder contrastive training). |
| `p12_ltr.yaml` | P12 | submitted | cross-fitted listwise learning-to-rank over P9's signals (per view: z-score, hubness-corrected score, reverse rank; PRF feedback per graph; P9 score + rank; candidate hubness), 5-fold over dev queries, test = mean of fold models. |
| `p13_rerank2.yaml` | P13 | submitted | P9's re-ranker with the stronger P11 retriever added as a view and as a graph. |
| `p14_ltr2.yaml` | P14 | submitted | cross-fitted LTR (as P12) over P13's signals (with the P11 retriever). |
| `p15_ltr3.yaml` | P15 | submitted | cross-fitted LTR (P14) + chunk-level MaxSim views (P11, P5 retrievers) + query-graph relevance propagation + TF-IDF trigram view; base re-ranker P13. |
| `p16_ltr3b.yaml` | P16 | submitted | P15 with a deeper candidate list (150) and 20 query neighbours. |
| `p17_gpt_mil.yaml` | P17 | submitted | LegalGPT-SC + LoRMA+ retriever trained end-to-end for chunk-level late interaction: each query / candidate is represented by 3 random 512-token chunks and scored by symmetric chunk MaxSim inside InfoNCE (multiple-instance contrastive learning); retrieval with MaxSim over up to 6 chunks per document. |
| `p18_ltr4.yaml` | P18 | submitted | P16's cross-fitted LTR + the MaxSim-trained retriever P17 as extra views (and their query-graph propagation) + 2-hop query-graph propagation. |
| `p19_hyb_mil.yaml` | P19 | submitted | P17's multiple-instance MaxSim training on LegalHybrid-SC with LoRMA+ (r16) -- a second, architecturally different late-interaction retriever for the ensemble. |
| `p20_ltr5.yaml` | P20 | submitted | P18 with both MaxSim-trained retrievers (P17 LegalGPT, P19 LegalHybrid) as extra views. |
| `p21_ltr_var1.yaml` | P21 | submitted | ranker variant for the ensemble -- depth 200, 5 query neighbours, wider MLP, different seeds. |
| `p22_ltr_var2.yaml` | P22 | submitted | ranker variant for the ensemble -- depth 75, 15 query neighbours, small MLP, different seeds. |
| `p23_ens.yaml` | P23 | submitted | ensemble (RRF or z-average, weights on cross-fitted dev scores) of the rankers P15, P16, P21, P22. |
| `p24_ens_final.yaml` | P24 | submitted | final ensemble of all cross-fitted rankers, including those with the MaxSim-trained retrievers (P18, P20). |
| `p25_ce.yaml` | P25 | submitted | chunk-pair cross-encoder re-ranker (LegalGPT-SC + LoRMA+ r16, causal mask removed): best 2 chunk pairs per (query, candidate) chosen by the P11 retriever, [query half / EOS / candidate half] = 512 tokens, max over pairs; localized contrastive estimation with 7 hard negatives (fused BM25 + dense, train split); re-scores the top-50 of P21. |
| `p26_ltr_ce.yaml` | P26 | submitted | P21's cross-fitted ranker with the P25 cross-encoder scores as an extra view (+ MaxSim retrievers if present). |
| `p27_ens_ce.yaml` | P27 | submitted | final ensemble of the cross-fitted rankers and the cross-encoder run (weights / fusion type chosen on dev). |
| `p28_ce2.yaml` | P28 | submitted | cross-encoder v2 -- 4 chunk pairs per (query, candidate) (more evidence per pair), 15 hard negatives per relevant case, LoRMA+ r32, re-scores the top-100 of the P26 ranker; 6 dev checkpoints so it can be stopped early. |
| `p29_ltr_ce2.yaml` | P29 | submitted | cross-fitted ranker with both cross-encoders (P25, P28) and the MaxSim retrievers as views. |
| `p30_ens_final.yaml` | P30 | submitted | final ensemble of CROSS-FITTED rankers only (P21, P16, P22, P18, P20, P26, P29). Their dev scores are out-of-fold, so the fusion weights are chosen honestly; the raw cross-encoder runs (whose fusion weight was tuned on dev) are excluded -- their signal reaches the ensemble through P26 / P29. |
