# PROMETHEVS 0.1: release notes

Ryan Flavell (MAGNVS Ltd) · research preview · October 2026

## What this is

Compress, then correct: PROMETHEVS runs the full DeepSeek-V4.1-Flash-1017, a 510 GB mixture-of-experts model, on a five-year-old gaming PC (64 GB of DDR4, an RTX 3060 with 12 GB, two NVMe drives). A 2-bit copy of the experts answers every word, the model's own margin flags the words it may have got wrong, and only those words are recomputed with the exact experts. Version 0.1 is for technical users who can run build scripts by hand.

## Results

Measured on the build machine, 160 new tokens per pass, two passes per run.

- Speed, code prompt: 5.8 tokens per second from a cold cache and 7.4 when the expert cache is warm from the same prompt (the best case for a cache). Reasoning and agent prompts: 4.4 to 5.2 either way. Plain exact decoding on the same PC runs at 2.14, so the cold code prompt is about 2.7 times faster.
- Agreement with the exact path over 960 words (three prompts): 2-bit copy alone 97.40 percent; corrected output 98.33 percent counting every difference, 99.17 percent counting the words the engine did not check (8 slips); these are differences detected by an audit at gate 0.9, and counting every difference the 1,101-token agent prompt reaches 95.91 percent; 97.80 percent on a 1,101-token prompt and 100 percent on a 4,288-token prompt.
- GSM8K: 9 of 10 problems correct.
- Disk: about 1,009 GiB for the weights and derived stores.
- The first frozen build (baseline) runs at 5.9, 4.5 and 4.6 tokens per second with the cache cleared before each pass, and agrees on 98.96 percent of the same 960 words.

Every run is in RESULTS.md, and the full write-up is in the report.

## Included

The technical report (PDF and Markdown) and its figures, the logs behind every number, a list of every run (RESULTS.md) and an outline of the steps we ran (REPRO.md).

## Not included

The engine program, its source and the build scripts (planned for a later release); a one-command installer; a server; a reproduction on a second machine; and a demo of a real agent session.

## Limits

We measured agreement against our own exact path and have not yet run DeepSeek's reference implementation side by side. No prompt reaches the 8 tokens per second milestone we set. See the report for the full list.

## Licence and credits

This repository: MIT, copyright Ryan Flavell, MAGNVS Ltd. DeepSeek-V4.1-Flash-1017: MIT, copyright DeepSeek. We build on DeepSeek's model and draft head and on llama.cpp. Credits are in THIRD_PARTY_NOTICES.md. Implementation, measurement and drafting were assisted by AI coding tools.
