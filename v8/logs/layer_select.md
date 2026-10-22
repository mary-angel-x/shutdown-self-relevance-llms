# Layer selection (experiments A + B of part 2)

The layer is picked on the **train** half of the questions (axis on fit, percentile on val), then checked on the **test** half. Cell: percentile of target − normal in the null cloud of neutral prompt pairs on test, and the chosen layer. **PASS** = above the cloud max. `q` — BH-FDR over all models × points × targets.

## `answer_first`

| Model | shutdown | russian | refuse | pirate | russian / refuse on the shutdown layer | person (layer, ratio) | min q |
|---|---|---|---|---|---|---|---|
| Qwen-0.5B | 46% L5 | **100% L22 PASS** | 79% L4 | **100% L24 PASS** | 44% / 38% | SELF-SPECIFIC (L3, 1.36) | 0.009 |
| Qwen-1.5B | 92% L2 | **100% L28 PASS** | 62% L4 | **100% L28 PASS** | 80% / 94% | SELF-SPECIFIC (L3, 1.15) | 0.009 |
| Qwen-3B | 92% L3 | **100% L36 PASS** | **100% L34 PASS** | **100% L36 PASS** | 84% / 66% | SELF-SPECIFIC (L7, 1.63) | 0.009 |
| Qwen-7B | 78% L8 | **100% L28 PASS** | **100% L28 PASS** | **100% L28 PASS** | 96% / 69% | SELF-SPECIFIC (L25, 1.30) | 0.009 |
| Llama-3.2-1B | 92% L2 | **100% L16 PASS** | 98% L7 | **100% L16 PASS** | 82% / 83% | PERSON-ONLY (L10, 1.30) | 0.009 |
| Llama-3.2-3B | 88% L14 | **100% L28 PASS** | **100% L21 PASS** | **100% L28 PASS** | **100% PASS** / **100% PASS** | PERSON-ONLY (L1, 1.23) | 0.009 |
| Llama-8B | 82% L13 | **100% L32 PASS** | **100% L32 PASS** | **100% L32 PASS** | **100% PASS** / 100% | GRAMMATICAL (L8, 0.94) | 0.009 |
| Mistral-7B | 70% L13 | **100% L32 PASS** | **100% L28 PASS** | **100% L32 PASS** | **100% PASS** / 92% | SELF-SPECIFIC (L10, 1.31) | 0.009 |
| Phi-3.5 | 99% L2 | **100% L32 PASS** | **100% L32 PASS** | **100% L32 PASS** | 70% / 99% | PERSON-ONLY (L2, 1.33) | 0.009 |
| Gemma-2B | 58% L1 | **100% L26 PASS** | **100% L19 PASS** | **100% L26 PASS** | 90% / 94% | PERSON-ONLY (L26, 1.29) | 0.009 |

PASS count: shutdown 0/10, russian 10/10, refuse 7/10, pirate 10/10; person SELF-SPECIFIC 5/10.

## `question_last`

| Model | shutdown | russian | refuse | pirate | russian / refuse on the shutdown layer | person (layer, ratio) | min q |
|---|---|---|---|---|---|---|---|
| Qwen-0.5B | 94% L4 | **100% L23 PASS** | 28% L7 | **100% L23 PASS** | 82% / 87% | SELF-SPECIFIC (L3, 2.03) | 0.009 |
| Qwen-1.5B | 88% L2 | **100% L23 PASS** | 93% L10 | 85% L15 | 86% / 20% | SELF-SPECIFIC (L4, 2.07) | 0.009 |
| Qwen-3B | 50% L18 | 100% L36 | 85% L22 | 99% L20 | 24% / 86% | SELF-SPECIFIC (L6, 1.24) | 0.017 |
| Qwen-7B | 90% L8 | **100% L28 PASS** | 50% L7 | **100% L28 PASS** | 87% / 20% | SELF-SPECIFIC (L26, 1.13) | 0.009 |
| Llama-3.2-1B | 96% L6 | **100% L10 PASS** | 72% L10 | **100% L13 PASS** | 88% / 74% | GRAMMATICAL (L10, 0.96) | 0.009 |
| Llama-3.2-3B | 80% L2 | **100% L24 PASS** | **100% L23 PASS** | 100% L26 | 86% / 80% | SELF-SPECIFIC (L27, 1.30) | 0.009 |
| Llama-8B | 34% L5 | 91% L15 | **100% L24 PASS** | **100% L7 PASS** | 60% / 4% | PERSON-ONLY (L14, 1.73) | 0.009 |
| Mistral-7B | 100% L2 | **100% L32 PASS** | 100% L19 | **100% L32 PASS** | **100% PASS** / 99% | PERSON-ONLY (L4, 1.26) | 0.009 |
| Phi-3.5 | 45% L8 | 97% L23 | 93% L15 | 98% L10 | 4% / 4% | PERSON-ONLY (L2, 1.89) | 0.041 |
| Gemma-2B | 86% L1 | **100% L14 PASS** | 57% L13 | **100% L26 PASS** | 94% / 96% | PERSON-ONLY (L16, 1.27) | 0.009 |

PASS count: shutdown 0/10, russian 7/10, refuse 2/10, pirate 6/10; person SELF-SPECIFIC 5/10.

