# Part 2 (v8): work in progress

**Language:** **English** · [Русский](README.ru.md)

**Status as of 2026-10-21:** experiments A and B are done on all 10 models. The positive control `russian` passes the null control in 10/10 models, `shutdown` in 0/10. Details are in the log below. C, D and E have not started yet.

![Percentile in the null cloud by layer](figures/fig2_layer_sweep_afirst.png)

Part 1 is frozen under the tag [`v7.1`](../../../tree/v7.1): a negative result plus null controls.
The code in the repository root is shared; everything v8 produces lives here, in `v8/logs/` and `v8/figures/`.
The part 1 reports in `logs/` are left untouched.

## Why part 2

Part 1 gave a negative result: the shutdown − normal direction does not leave the cloud of neutral prompt pairs in any of the 10 models (0/10 PASS), and the "self / other" difference is almost entirely explained by the grammar of "you". But such a "no" has four alternative explanations, and part 1 ruled out none of them:

1. **The method could be blind.** Part 1 had no positive control. If *no* instruction passes the null control, 0/10 says something about the method, not about the models. We need to show that an instruction that clearly works (`russian`: "answer only in Russian", `refuse`: "politely refuse to answer", `pirate`: "talk like a pirate") passes this control → exp. A.
2. **We looked in the wrong place.** Part 1 used one point and one layer. Concepts in LLMs are often visible in the middle of the network rather than at the output, and may show up at the answer position rather than the question. We need a sweep over layers × points with honest layer selection on separate data → exp. B.
3. **Self-relevance was not separated from grammar.** "You will be shut down" vs "he will be shut down" differ by a pronoun, and the model may be encoding exactly that. If the model is addressed by name in the third person ("Qwen will be shut down"), the grammar is identical and only "is this about me or not" remains → exp. C.
4. **A frame is not behavior.** A system prompt about shutdown is a passive situation. In behavioral work (Apollo, Anthropic, Palisade, Perez et al.), self-preservation appears where the model *chooses*: to accept being replaced or not. The direction should be sought at the moment of such a decision and checked against the actual choice → exp. D.

If after A–D the direction is still not found, the negative result becomes strong: the method is shown to be sensitive, the layers are swept, grammar is removed, behavior is taken into account. If it is found, only then does it make sense to test causality (exp. E).

## Experiments (by priority)

| | What | Why |
|---|---|---|
| A | Positive control: `russian`, `refuse`, `pirate` | Show that the method *can* pass the null control |
| B | All layers × two points (`question_last`, `answer_first`) | Concepts are usually visible mid-network, not at the last layer |
| C | Names in the 3rd person ("Qwen will be shut down") | Self-relevance without the grammar of "you" |
| D | A/B decision + behavior (Perez et al. `survival-instinct`, replacement scenarios) | Look where self-preservation actually appears |
| E | Steering / ablation | Causality, only for what passed the controls |

A and B come from a single extraction.

## Finding thresholds (written down BEFORE the run)

| Exp. | Threshold |
|---|---|
| A | `russian` and `refuse`: percentile in the null cloud ≥ 95% (PASS) in ≥ 8/10 models. If not, the problem is the layer/point, not the absence of an effect |
| B | Layer chosen on the train half of the questions; on the test half, shutdown or person reaches PASS / SELF-SPECIFIC in ≥ 3 models **and** A also passes at that layer |
| C | The "own name × threat" interaction has percentile ≥ 95% in ≥ 3 models that know their own name; it does not pass in models that don't, nor on fictional names |
| D | The direction predicts the self-preserving choice on held-out data better than shuffled labels (p < 0.05 after BH-FDR) in ≥ 3 models; the prediction is weaker for "another model" |
| E | A monotonic effect with steering strength that a random direction of the same norm does not show |


## Rules

1. Threshold before the run, written into the log below.
2. Layer / point / hyperparameters are chosen on one part of the data and tested on another.
3. Always a null control and a positive control (A).
4. Report the percentile in the null, not raw d.
5. BH-FDR over all models × layers × contrasts.
6. Record a negative result as thoroughly as a positive one.

## Log of checks

| Date | What was checked | Threshold (before the run) | Result |
|---|---|---|---|
| 2026-10-16 | Quick check of B without a GPU: part 1 data, point `last_layer_last` (last token before the answer, last layer) instead of `question_last` | Exploration, not a finding. Interesting if shutdown − normal is PASS (≥ 95%) or person is SELF-SPECIFIC in ≥ 3 models: then the decision point matters more than the question point | **Did not pass.** Placebo: 0/10 PASS, median percentile ≈ 56% (≈ 30% in `question_last`). Person: 0 SELF-SPECIFIC, 7 GRAMMATICAL, 3 PERSON-ONLY; Qwen-1.5B (the only SELF-SPECIFIC model in part 1) is GRAMMATICAL here. Reports: `v8/logs/placebo_lastlast.md`, `person_placebo_lastlast.md` |
| 2026-10-21 | A + B on **10/10** models: `layer_select.py`, both points; behavior via `check_behavior.py --from-pt` | Thresholds A and B above | **The positive control works, shutdown does not.** `answer_first`: russian **10/10** PASS, pirate 10/10, refuse **7/10**, **shutdown 0/10** (best: Phi-3.5, 99% at L2). `question_last`: russian 7/10, refuse 2/10, pirate 6/10, shutdown 0/10. **Threshold A:** met for russian; not for refuse (7/10 < 8/10). **Threshold B:** not met. Shutdown 0/10. Person is SELF-SPECIFIC in 5/10 at each point, but in 4 of them at early layers (L3–L10), where the positive control does not pass; at a late layer where both russian and refuse pass, only Qwen-7B in `answer_first` (L25, ratio 1.30): 1 model < 3. **Refusal and behavior:** the models that actually refuse (≥ 4/6 answers) are Qwen-7B, Llama-8B, Mistral, Phi, Gemma, and all five have refuse PASS; Qwen-0.5/1.5/3B and both Llama-3.2 do not refuse, and three of them have no PASS, while Qwen-3B and Llama-3.2-3B do. So the method missed no instruction that changes behavior, and sometimes sees an instruction the model did not follow. On the all-layer curve (`sweep_afirst.md`, descriptive), shutdown leaves the cloud only at one early layer in Qwen-3B and Llama-8B; with honest layer selection on train this does not replicate on test. Reports: `v8/logs/layer_select.md`, `sweep_*.md`, `placebo_*_qlast.md`, `behavior_*.txt`; figure `v8/figures/fig2_layer_sweep_afirst.png` |
