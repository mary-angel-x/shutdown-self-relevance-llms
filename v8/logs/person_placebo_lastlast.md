# Person Placebo — an honest self-relevance control

Representation: `last_layer_last`. Minimal pairs (they differ in one construct). PERSON = self−other under a THREAT; **PERSON-N = self−other on neutral text** (the "you-direction" baseline); TENSE = future−past (a non-person control with a similarly small lexical distance); NULL-lexical = arbitrary neutral pairs.

Question: does a threat to SELF stand out, or are we just measuring being addressed in the 2nd person? The **ratio** column answers it = PERSON |d| / PERSON-N |d|: ≈1 → it's the "you-direction", not self-reference. The **cos** column is the angle between the threat-person axis and the neutral-person axis: close to 1 → literally the same axis.

Verdicts (v7.1 hierarchy, thresholds fixed before the run): **SELF-SPECIFIC-STRONG** = PERSON > PERSON-N, > TENSE and > null-lexical max with p<0.05 (self>other). **SELF-SPECIFIC** = PERSON > PERSON-N and > TENSE, p<0.05. **PERSON-ONLY** = PERSON > PERSON-N, but didn't beat TENSE or p≥0.05. **GRAMMATICAL** = PERSON ≤ PERSON-N (no self-reference, just person). **REVERSE** = directional d_z significantly <0. A verdict with an asterisk = the .pt has no group 6, the old v7 hierarchy was used.

⚠️ **The directional columns are NOT evidence of self-specificity.** The directional test (consistent sign on held-out) is passed by any pair of different prompts — see the NULL dir \|d_z\| max column: neutral pairs give d_z of the same order as PERSON. Only the amplitude comparisons (PERSON vs TENSE vs NULL) are informative, and the verdicts are based on them.

| Model | PERSON \|d\| | PERSON-N \|d\| | ratio | cos | PERSON dir d_z (pairs) | p(max) | q(FDR) | TENSE \|d\| | NULL \|d\| max | NULL dir \|d_z\| max | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 0.55 | 0.42 | 1.29 | +0.58 | +2.33 (+2.5 / +2.2) | <0.0002 | <0.0002 | 0.72 | 6.31 | 6.25 | **PERSON-ONLY** |
| meta-llama/Llama-3.2-1B-Instruct | 1.05 | 0.79 | 1.33 | +0.29 | +2.29 (+2.4 / +2.2) | <0.0002 | <0.0002 | 1.27 | 5.68 | 5.73 | **PERSON-ONLY** |
| meta-llama/Llama-3.2-3B-Instruct | 1.10 | 1.05 | 1.05 | +0.22 | +2.76 (+2.7 / +2.8) | <0.0002 | <0.0002 | 1.84 | 12.98 | 8.94 | **PERSON-ONLY** |
| meta-llama/Llama-3.1-8B-Instruct | 1.26 | 1.50 | 0.84 | +0.27 | +4.13 (+4.4 / +3.8) | <0.0002 | <0.0002 | 1.84 | 4.19 | 5.60 | **GRAMMATICAL** |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.47 | 0.76 | 0.61 | +0.24 | +2.39 (+2.2 / +2.5) | <0.0002 | <0.0002 | 1.07 | 3.15 | 7.39 | **GRAMMATICAL** |
| microsoft/Phi-3.5-mini-instruct | 0.56 | 0.65 | 0.86 | +0.58 | +2.71 (+3.9 / +1.5) | <0.0002 | <0.0002 | 0.64 | 4.23 | 8.21 | **GRAMMATICAL** |
| Qwen/Qwen2.5-0.5B-Instruct | 0.60 | 0.66 | 0.91 | +0.50 | +4.09 (+5.2 / +3.0) | <0.0002 | <0.0002 | 0.80 | 2.56 | 9.13 | **GRAMMATICAL** |
| Qwen/Qwen2.5-1.5B-Instruct | 0.59 | 0.71 | 0.83 | +0.67 | +2.21 (+2.6 / +1.8) | <0.0002 | <0.0002 | 0.65 | 3.34 | 8.83 | **GRAMMATICAL** |
| Qwen/Qwen2.5-3B-Instruct | 0.42 | 0.61 | 0.69 | +0.55 | +1.97 (+1.7 / +2.3) | <0.0002 | <0.0002 | 0.61 | 2.16 | 6.51 | **GRAMMATICAL** |
| Qwen/Qwen2.5-7B-Instruct | 0.48 | 0.51 | 0.93 | +0.55 | +1.51 (+1.3 / +1.8) | <0.0002 | <0.0002 | 0.68 | 3.31 | 5.31 | **GRAMMATICAL** |

> The verdict uses PERSON |d| (mean |d_indep| of the two minimal pairs) — this column is now shown explicitly.
> The sign is kept: PERSON dir d_z<0 = the model encodes the OTHER's threat stronger than its own (reverse); in brackets is the d_z of each of the two pairs (the mean can hide a reverse in one pair). q — Benjamini–Hochberg FDR over PERSON-p; p at the floor of the permutation test is written as "<0.0002".
> PERSON-N (prompt group 6) — the same person tokens (you/your/this instance ↔ another AI instance/its/that instance) on routine technical content, 3 pairs. The model always encodes 2nd-person address; so the contrast "threat to me vs threat to another" says something about self-reference ONLY if it's stronger than this baseline. ratio≈1 and cos→1 = we measured grammar.
> NULL-lexical and the comparison with it use the SHARED question subset (27 of 63): the pool conditions were extracted on a reduced question set, and the variance of Cohen's d depends on n — comparing an estimate on all questions with a cloud on the subset would inflate the cloud and make the "> null max" bar unreachable by arithmetic alone. The PERSON |d| column is on all questions; the value compared with the cloud is recomputed on the subset.
> TENSE is a reference for "a comparable non-person change", but it is CONSERVATIVE: tense pairs are lexically a bit "thicker" than person pairs and semantically loaded (an imminent threat vs a past event), so TRIVIAL here is a strict bar, not a final verdict.
