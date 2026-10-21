# Results appendix: every run

All figures come from the runs of 19 and 20 October 2026, labelled as in the report: the **engine** (program BF, the full method) and the **baseline build** (program BC, the first frozen build) on the machine described in the README (64 GB DDR4, RTX 3060 12 GB, two NVMe drives). Each prompt generates 160 new tokens, twice in one process: pass 1 and pass 2. Speeds are new tokens per second. The baseline build clears the expert cache before each pass; the engine keeps it from pass 1 into pass 2.

Source folders (under `logs/`): `final1017` (baseline build speeds), `base1017j` (plain decoding), `audit1017b` (three-prompt audit), `gsm1017b` (GSM8K), `keep1017` (warm session), `finalbf1017`, `auditbf1017`, `auditbf1017L`, `auditbf1017LX`, `gsmbf1017` (engine).

## A. Baseline build (BC), speed

Engine program `v41_twolane_gen2t_BC.exe`; correction gate 0.45, expert skip 0.12, near-tie acceptance 0.5, five guess lanes.

| Prompt | Repeat | Pass 1 | Pass 2 |
|---|---|---|---|
| Code | 1 | 5.500 | 5.803 |
| Code | 2 | 5.647 | 6.010 |
| Reasoning (r22) | 1 | 4.881 | 3.556 |
| Reasoning (r22) | 2 | 4.873 | 5.414 |
| Agent | 1 | 3.889 | 4.421 |
| Agent | 2 | 5.053 | 4.697 |

The baseline figures are the means of the two pass-2 runs: code 5.91, reasoning 4.49, agent 4.56.

Plain exact decoding (no 2-bit copy, no draft head, no corrections), measured earlier the same day in a separate session (`base1017j`): pass 1 1.893 and 2.089, pass 2 2.158 and 2.115. The mean of the two pass-2 values is 2.14.

## B. Quality audits

A slip is a committed word that differs from the word the engine's own exact correction lane would have produced. The unchecked count includes words the engine kept at a near-tie and overturned words, counted when their confidence margin is at or above the gate, so the engine never checked them. The reference is the engine's own exact lane, not DeepSeek's reference implementation.

**The three standard prompts** (code 30 tokens, reasoning 76 tokens, agent 1,163 tokens; 3 prompts x 160 tokens x 2 passes = 960 words):

| Gate | Baseline build, unchecked | Engine, unchecked | Engine, overturned words only |
|---|---|---|---|
| 0.45 | 10 slips, 98.96 percent | 8 slips, 99.17 percent | 6 slips, 99.38 percent |
| 0.30 | 12 slips, 98.75 percent | 11 slips, 98.85 percent | 8 slips, 99.17 percent |
| 0.25 | 14 slips, 98.54 percent | 13 slips, 98.65 percent | 9 slips, 99.06 percent |

The overturned-only count is a different measure and does not compare with the unchecked count.

**Counting every difference and the 2-bit copy alone** (gate 0.45). Counting every difference adds the words below the gate that the engine checked and deliberately kept at a near-tie. The 2-bit copy alone counts every word where the cheap pass and the exact path differ before any correction.

| Audit | Unchecked | Every difference | 2-bit copy alone |
|---|---|---|---|
| Engine, 960 words | 8 (99.17 percent) | 16 (98.33 percent) | 25 (97.40 percent) |
| Baseline build, 960 words | 10 (98.96 percent) | 16 (98.33 percent) | 29 (96.98 percent) |
| Engine, 1,101-token prompt, 318 words | 7 (97.80 percent) | 13 (95.91 percent) | 22 (93.08 percent) |
| Baseline build, 1,101-token prompt, 317 words | 9 (97.16 percent) | 14 (95.58 percent) | 22 (93.06 percent) | A difference of one or two slips in 960 words is too few to call a quality difference.

**Two further prompts** (a second agent prompt of 1,101 tokens and a 4,288-token trace prompt), engine, one run each, gate 0.45:

| Prompt | Words | Unchecked slips | Agreement (unchecked) | Overturned words only |
|---|---|---|---|---|
| Agent, 1,101-token prompt | 318 | 7 (2 overturned plus 5 near-ties kept) | 97.80 percent | 2 slips, 99.37 percent |
| Trace, 4,288-token prompt | 320 | 0 (one overturned word below the gate; no near-ties kept) | 100 percent | 0 |

**Baseline build on the 1,101-token agent prompt** (raw log `auditbc1017L/v41_auditbf1017L_AGENT2.log`; one run of two passes, unchecked count): gate 0.45, 8 overturned words plus 1 near-tie kept = 9 slips in 317 words, 97.16 percent; gate 0.30, 12 slips, 96.21 percent.

Earlier recipe, before near-tie acceptance existed (overturned-only and unchecked are the same count there): the 1,101-token agent prompt scored 98.12 percent (6 slips in 320 words) and the 4,288-token prompt 0 slips in 320 words. The baseline build has now been audited on both of these prompts (above).

## C. GSM8K, 10 problems

320 tokens, one pass each, key from the repository's seeded sample.

| Problem | Key | Baseline build: answer, tokens/s | Engine: answer, tokens/s |
|---|---|---|---|
| G01 | 14 | 14, 6.086 | 14, 4.060 |
| G02 | 1430 | 1430, 6.248 | 1430, 6.284 |
| G03 | 623 | 623, 5.872 | 623, 4.187 |
| G04 | 25 | 25, 5.602 | 25, 3.877 |
| G05 | 200 | 240, 3.645 | 240, 3.176 |
| G06 | 30 | 30, 6.485 | 30, 7.012 |
| G07 | 1 | 1, 3.318 | 1, 3.385 |
| G08 | 63 | 63, 5.737 | 63, 4.828 |
| G09 | 16 | 16, 4.381 | 16, 3.891 |
| G10 | 56 | 56, 5.532 | 56, 6.278 |
| Score, mean speed | | 9 of 10, 5.29 | 9 of 10, 4.70 |

On G05 the engine made 17 corrections, 6 overturned. We have not run the exact 1017 model on these problems.

## D. Warm sessions

"Warm" means the expert cache is kept from pass 1 into pass 2 of the same prompt. This is the best case for a cache.

**Baseline build with cache kept** (engine program `BE`, which is `BC` plus one driver edit; single runs, 20 October):

| Prompt | Pass 1 (cold) | Pass 2 (cache kept) | Notes |
|---|---|---|---|
| Code | 4.512 | 6.630 | The 160 tokens equal the cold runs' pass-2 tokens (compared side by side). Pass 1 ran slower than the cold runs of section A (4.512 against 5.500 and 5.647); the machine state differed. One overturned word in pass 2. |
| Reasoning (r22) | 4.700 | 4.081 | No gain; the cold pass-2 runs were 3.556 and 5.414. |
| Agent | 4.519 | 4.217 | No gain; the cold pass-2 runs were 4.421 and 4.697. The engine read about 277 GB of expert bytes in each pass (the counter includes the prompt read); this figure does not show why the agent prompt did not gain, and the report does not use it as a cause (see the round timers). |

## E. Engine (BF)

Engine program `BF` = `BD` plus the cache-keep edit. The draft head's experts are held in the 2-bit format, and the cache is kept from pass 1 into pass 2. Single runs, 20 October.

| Prompt | Pass 1 (cold) | Pass 2 (cache kept) | Details |
|---|---|---|---|
| Code, run 1 | 4.975 | 7.389 | Draft time 3.40 s, then 2.85 s. 114 of 114 guesses accepted in pass 2. 3 corrections in each pass, none overturned. 7 plain steps in pass 2. |
| Code, run 2 | 5.815 | 7.402 | Pass 1: 3 corrections, 1 overturned (10 tokens discarded), 120 of 127 guesses accepted, draft 3.39 s. Pass 2: 2 corrections, none overturned, 113 of 120 accepted (94.2 percent), draft 2.98 s, 9 plain steps. |
| Code, run 3 | 5.947 | 7.336 | Pass 1: 3 corrections, 1 overturned (3 tokens discarded), 112 of 117 accepted, draft 3.40 s. Pass 2: 1 correction, none overturned, 114 of 125 accepted (91.2 percent), draft 2.89 s, 7 plain steps. |
| Reasoning (r22), run 1 | 4.444 | 4.453 | Draft 4.43 and 4.30 s. 93 of 103 and 104 of 113 guesses accepted. 15 corrections (1 overturned) and 12 (4 overturned). 30 and 28 plain steps. |
| Reasoning (r22), run 2 | 4.905 | 4.400 | Pass 1: 8 corrections (2 overturned, 13 tokens discarded), 100 of 108 accepted, 29 plain steps, draft 4.29 s. Pass 2: 16 corrections (3 overturned, 21 tokens discarded), 107 of 121 accepted (88.4 percent), 23 plain steps, draft 4.28 s. |
| Agent, run 1 | 4.840 | 4.751 | Draft 3.62 and 3.71 s. 117 of 121 and 122 of 125 guesses accepted. 4 corrections (2 overturned) and 6 (2 overturned). 7 plain steps in pass 2. |
| Agent, run 2 | 5.602 | 4.795 | Pass 1: 3 corrections (1 overturned, 3 tokens discarded), 118 of 119 accepted, 6 plain steps, draft 3.28 s. Pass 2: 4 corrections (2 overturned, 11 tokens discarded), 118 of 125 accepted (94.4 percent), 11 plain steps, draft 3.57 s. |

Code: the warm pass was faster than the same run's cold pass by 1.49, 1.27 and 1.23 times. Median of the three warm passes 7.39; median of the three cold passes 5.82. The baseline build's cold runs were 5.500, 5.647, 5.803 and 6.010. In all three warm code passes no correction overturned a word; the baseline build's one warm pass (6.630) had one. The warm cache and the absence of an overturned word therefore cannot be separated in these runs.

Draft time: 3.8 to 4.0 s in the baseline build's code runs; 2.85, 2.98 and 2.89 s in the engine's warm passes.

**One like-for-like cold comparison.** The code run 2 pass 1 text equals the baseline build's pass 1 text exactly. It ran at 5.815 tokens/s against 5.647, with draft time 3.39 s against 4.08 s.

**First differing token against the baseline build's run of the same prompt** (counting from the first generated token): code run 1 pass 1 at the 33rd, pass 2 at the 10th; code run 2 pass 1 none, pass 2 at the 37th; reasoning run 1 pass 1 at the 157th, pass 2 at the 16th; agent run 1 pass 1 at the 27th, pass 2 at the 6th.

## F. Disk

| Folder group | Size |
|---|---|
| Raw weights (C: 221.0 GiB, F: 254.3 GiB) | 475.3 GiB |
| 2-bit copies and correction planes (C: 167.8, F: 153.2) | 321.0 GiB |
| FP4 copies and mirrors of the first layers (C: 20.0 + 55.1, F: 45.1 + 75.9) | 196.1 GiB |
| Non-expert model file (C:) | 16.7 GiB |
| Total | 1,009.1 GiB |

Measured by a size listing of the nine store folders on 20 October.

## G. Speed budget measurements

- One 160-token code answer (second pass, cache cleared): 97.9 GB of expert bytes read from files over the whole pass, which includes preparing the prompt; the 26.6 s is generation only, so no read rate is derived.
- Cold random reads of 11.5 MB records: C: 1.2 GB/s with one reader and 4.0 GB/s with eight; F: 1.6 and 5.9 GB/s.
- One record takes 7 to 11 ms to read whole, against about 5.5 ms of compute per layer.
- Engine, warm code pass (code run 2, pass 2): a clean round takes 404.8 ms and the draft 78 ms; at 3.97 words per round that is about 120 to 122 ms per word on clean rounds. Baseline build, second code pass: 455 ms and 97 ms, 4.13 words per round, 134 ms per word (arithmetic on measured values).
- An overturned word costs 2.6 to 3.2 s on the code prompt.

## H. Things we tried that did not help

| Form | Result |
|---|---|
| Draft block of 8 words instead of 5 | Code 5.541 against 5.449 tokens/s (control, same session); reasoning 3.914 and agent 4.189 tokens/s with more plain steps; later guesses are accepted less often. |
| Two staggered groups of rows through one engine pass (micro-benchmark, not the engine) | 1.13 times the time of one group at the production miss rate, 1.21 times at the higher rate. |
| Low-bit CPU decoder, EXL3-style layout (CPU kernels on stored expert matrices) | 1.46 to 8.7 times slower than the current 2-bit format. |
| Generic compression of stored experts | Size 0.905 of the original: too little saving to convert. |
| Scalar and block-wise re-quantisation at about 1.6 bits per weight | 1.7 to 2.1 times the local error of the current 2-bit format (local error on stored matrices, not model quality). |
| Uniform expert skip above 0.12 | At 0.15: code 6.16 against 5.91, reasoning 3.68 and 2.65 against 4.49. At 0.18: code 4.89 and 4.96. Stopped. |

Each row is a statement about that form, not about the idea behind it.
