<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="figures/hero_transparent.png">
    <img src="figures/hero.png" alt="PROMETHEVS" width="420">
  </picture>
</p>

<h3 align="center">Run the full 510 GB DeepSeek-V4.1-Flash on a five-year-old gaming PC</h3>

<p align="center">The original model, not a smaller copy · 64 GB of RAM and one 12 GB NVIDIA card · free and open source</p>

<p align="center">
  <a href="https://github.com/Magnvs7/Promethevs/releases/latest"><img src="https://img.shields.io/github/v/release/Magnvs7/Promethevs?label=release" alt="Latest release"></a>
  <a href="engine-0.2/LICENSE"><img src="https://img.shields.io/badge/engine-Apache--2.0-blue" alt="Engine licence: Apache-2.0"></a>
  <a href="https://doi.org/10.5281/zenodo.23577839"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.23577839.svg" alt="DOI"></a>
</p>

PROMETHEVS runs **DeepSeek-V4.1-Flash-1017**, one of the strongest open models, from all 510 GB of its original weights, on an ordinary PC that cannot even load it: a Ryzen 9 3950X with 64 GB of DDR4, an RTX 3060 with 12 GB and two NVMe drives.

The **PROMETHEVS engine** gets there with **compress-then-correct**, a method devised by Ryan Flavell. Every word is first computed on a 2-bit copy of the model's experts. When the model itself is unsure (its top two choices are close), that word is recomputed with the original weights, rebuilt exactly from a stored correction plane. You get close to the speed of the compressed copy and close to the answers of the full model.

- **Full-model quality, counted.** On 960 audited words the output picks the full model's word 98.3% of the time (the 2-bit copy alone: 97.4%).
- **Up to 7.4 tokens/s** on a code prompt (warm cache, best case), against 2.14 for running the full model exactly on the same PC. Server runs: 5 to 8 on code, 4.5 to 5.4 on reasoning and agent prompts.
- **Fast follow-ups.** Only the first read of a long prompt is slow; the engine keeps what it has read, so the next message in a conversation starts in about 2 seconds.
- **Works with your tools.** An OpenAI-compatible endpoint on `localhost`, with streaming and tool calls, so apps and coding agents that speak the OpenAI API can use it.
- **Private.** The model runs on your PC; your chats stay there.
- **Built from the original weights, on your machine.** The installer downloads DeepSeek's weights and builds every store itself, checked against published hashes, so you do not have to trust our files.
- **Free.** Engine Apache-2.0; report, figures and logs CC BY 4.0.

**New here?** Check [what you need](#what-you-need), then [install](#install).

## What you need

| | |
|---|---|
| **System** | Windows 10 or 11, 64-bit |
| **CPU** | x86-64 with AVX2; 16 cores recommended |
| **RAM** | 64 GB |
| **GPU** | one NVIDIA RTX 30 or 40 series card with 12 GB (compute capability 8.6-8.9), driver 580 or newer |
| **Disk** | about 1.1 TB free on NVMe (one drive, or split across two) |
| **Download** | about 510 GB of DeepSeek's original weights, plus about 3 GB for the converter |
| **Time** | the store build takes about 4 hours on the build PC, after the download |

The hardware check in the bundle tests all of this and tells you up front if anything falls short. The full list is in [engine-0.2/INSTALL.md](engine-0.2/INSTALL.md).

## Install

1. Download `PROMETHEVS_0.2.zip` from the [latest release](https://github.com/Magnvs7/Promethevs/releases/latest) and extract it, for example to `C:\promethevs_pkg`.
2. In PowerShell, from that folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check_hardware.ps1        # checks the PC, picks the drives
powershell -ExecutionPolicy Bypass -File scripts\build_stores.ps1 --run    # downloads the weights, builds the stores (resumable)
powershell -ExecutionPolicy Bypass -File scripts\start_engine.ps1          # starts the engine (about 1 minute)
python scripts\py\verify_install.py --speed                                # checks it and measures the speed
```

3. Point any OpenAI-compatible app at `http://127.0.0.1:8100/v1`, model `deepseek-v4.1-flash-1017-promethevs`, any non-empty API key. Close other programs that use the graphics card first.

Every step, the time and disk each one takes, and the known gaps are in [engine-0.2/INSTALL.md](engine-0.2/INSTALL.md). Release notes: [engine-0.2/RELEASE_NOTES_0.2.md](engine-0.2/RELEASE_NOTES_0.2.md).

![Speed by prompt](figures/fig1_speed.svg)

## Speed and quality in detail

Measured on the build machine, 160 new tokens per prompt, two passes per run. **Cold** is a pass that starts with an empty expert cache. **Warm** is the second pass of the same prompt with the cache kept from the first, which is the best case for a cache; a real session that sends a different prompt lies between the two. Every individual run is in [RESULTS.md](RESULTS.md).

| | Engine (full method) | Baseline build (first frozen build) |
|---|---|---|
| Speed, code prompt, cold | 5.8 tokens/s (runs 4.98, 5.82, 5.95) | 5.9 (runs 5.80, 6.01) |
| Speed, code prompt, warm | 7.4 tokens/s (runs 7.39, 7.40, 7.34) | 6.6 (one run) |
| Speed, reasoning prompt, cold / warm | 4.7 / 4.4 | 4.5 (runs 3.56, 5.41) / 4.1 (one run) |
| Speed, agent prompt, cold / warm | 5.2 / 4.8 | 4.6 (runs 4.42, 4.70) / 4.2 (one run) |
| Plain exact decoding on the same PC, cold, for scale | 2.14 tokens/s (the engine's cold code prompt is about 2.7 times faster) | |
| Agreement with the exact path, 960 words | 99.17 percent (8 slips) | 98.96 percent (10 slips) |
| Agreement, 1,101-token prompt | 97.80 percent | 97.16 percent |
| Agreement, 4,288-token prompt | 100.00 percent | 99.69 percent |
| Agreement counting every difference, 960 words | 98.33 percent | 98.33 percent |
| 2-bit copy alone, before correction, 960 words | 97.40 percent | 96.98 percent |
| GSM8K, 10 problems | 9 correct | 9 correct |

Agreement is measured word by word against the word the exact path would choose given the same history. A slip is a difference the engine did not check, because the model was confident; the unchecked count counts those. Counting every difference also includes the words the engine checked and deliberately kept at a near-tie (both words almost equally likely). On its own the 2-bit copy picks the full-weight word on 97.4 percent of words; with the corrections, 98.33 percent counting every difference (99.17 percent by the unchecked count). The engine differs from the baseline build in two ways: it holds the draft head's experts in the 2-bit format too, and it keeps the expert cache from the first pass into the second. Keeping the cache lifted the code prompt but not the reasoning or agent prompts (the report gives the round timers behind this). In all three warm code passes no correction overturned a word, so the cache, the 2-bit draft head and the absence of an overturn are not separated, and 7.4 is a best case, not a typical session.

## How it works

The PROMETHEVS engine does four things at once.

**A 2-bit copy answers every word.** The experts are stored in a 2-bit format that is smaller than the original FP4, so each word needs fewer bytes from disk. That is what makes the machine fast enough to be useful.

**The model's own confidence picks the words to redo.** For each word the engine looks at the gap between the model's top two candidates. When the gap falls below a threshold, the gate (0.45 here), the word is flagged.

**Flagged words are corrected exactly.** A stored correction plane turns an expert's 2-bit copy back into the exact FP4 expert, so a correction is the full model's answer for that word.

**A draft head proposes several words at a time.** DeepSeek ships a draft head with the model. The engine checks its guesses together in parallel lanes, which spreads the cost of reading experts across several words. The experts the model uses most sit in GPU memory; the rest live in RAM and on the two drives.

![Method](figures/fig5_method.svg)

The method is compress-then-correct, devised by Ryan Flavell; the PROMETHEVS engine is built around it. Speculative decoding, quantised drafts and offloaded mixture-of-experts inference are the work it builds on; the report sets the method beside the closest references.

## Hardware and disk

The build machine is a Windows 11 desktop with 64 GB DDR4, an RTX 3060 12 GB, and two NVMe drives (2 TB and 1 TB). The stores need about **1,009 GiB**: 475 GiB of raw weights plus 534 GiB of derived stores (2-bit copies, correction planes, and FP4 copies of the first layers). Disk is the largest barrier for anyone repeating this. We kept each drive below about 90 percent full.

## Results from other machines

Every number above comes from one PC. If you run release 0.2 on your own machine, please send the output of `python scripts\py\verify_install.py --speed` and your engine log as a [benchmark report](https://github.com/Magnvs7/Promethevs/issues/new?template=benchmark_report.md). Reports are added here as they arrive.

| Machine | CPU | RAM | GPU | Code, warm (tok/s, generation only) | Reasoning / agent, warm (tok/s) | Reported by |
|---|---|---|---|---|---|---|
| Build PC | Ryzen 9 3950X | 64 GB DDR4 | RTX 3060 12 GB | 5.1-8.0 (two runs, 23 Oct); report best case 7.4 | 4.5-5.4 | Ryan Flavell |

## Why PROMETHEVS

Frontier-level AI today needs either a cloud subscription or hardware costing tens of thousands of pounds. Open-weight models are closing the gap with the closed frontier models, but running them still means someone else's server. PROMETHEVS, named after Prometheus, who in the myth brought fire to people, sets out to change that: frontier capability, at a speed close to the cloud, within 97 percent of what the original full weights would say, on a computer people already own. Release 0.1 published the method and every measurement; release 0.2 lets you run it.

## Technical report

> Ryan Flavell. *Compress-then-Correct Decoding: The Full DeepSeek-V4.1-Flash-1017 on a Five-Year-Old Gaming PC.* PROMETHEVS Technical Report 1, version 0.1. MAGNVS Ltd, October 2026. [doi:10.5281/zenodo.23577839](https://doi.org/10.5281/zenodo.23577839)

The report ([PDF](report/PROMETHEVS_report_0.1.pdf), [Markdown](PROMETHEVS_report_0.1.md)) sets out the method, the measurements, where the time goes, what did not help and the related work. Every run behind it is listed in [RESULTS.md](RESULTS.md), with the unedited logs in [logs/](logs/).

## What is included

- The technical report (PDF and Markdown) and its figures
- The logs behind every number in it, unedited ([RESULTS.md](RESULTS.md) lists them)
- The commands we ran to build the stores and the measurements ([REPRO.md](REPRO.md))
- Since 0.2: the engine server, its proxy, the hardware check, the store builder and the install check, as `PROMETHEVS_0.2.zip` on the [v0.2 release](https://github.com/Magnvs7/Promethevs/releases/tag/v0.2) (docs in [engine-0.2/](engine-0.2/))

## What is not included

- A source rebuild that reproduces the shipped engine program byte for byte (0.2 ships it prebuilt, with its frozen build inputs; see gaps G1 and G2 in [engine-0.2/INSTALL.md](engine-0.2/INSTALL.md)).
- A clean install of 0.2 on a second machine has not been run yet.
- Other platforms: 0.2 runs on Windows with one NVIDIA RTX 30 or 40 series card, and serves one request at a time.
- A reproduction on a second machine, and a demo of a real agent session.

## Limits

We measured agreement against our own exact path; we have not yet run DeepSeek's reference implementation side by side. Speeds vary by prompt and by run, so we give every run. No prompt reaches the 8 tokens/s milestone we set, and the reasoning and agent prompts run below 5.3. First-word time on long prompts is read-bound: on a 1,163-token prompt the first word arrives after about 126 s (98 s with a candidate prompt-reading setting whose effect on quality we have not fully measured). The 99.17 percent figure covers three prompts of 30, 76 and 1,163 tokens. On a further 1,101-token agent prompt the engine scores 97.80 percent and the baseline build 97.16 (9 slips in 317 words), 0.16 points above our 97 percent contract at the gate we use (at gate 0.30 it falls to 96.21 percent); on a 4,288-token prompt the engine scores 100 percent and the baseline build 99.69. Counting every difference, both builds fall below 97 percent on the 1,101-token prompt (95.91 and 95.58 percent). Each long-prompt audit is one run. The chat template matches DeepSeek's encoder on six plain conversations; tool results and thinking mode are unchecked. All numbers come from one machine, one model and one prompt family.

## Reproduce

[REPRO.md](REPRO.md) outlines the steps we ran: download and verify the weights, build the stores, set the profile, run the benchmark. The scripts themselves ship in release 0.2 (`PROMETHEVS_0.2.zip`, see [engine-0.2/INSTALL.md](engine-0.2/INSTALL.md)).

## Licence

Copyright (c) 2026 Ryan Flavell, MAGNVS Ltd. The report, figures, logs and text in this repository are licensed under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](LICENSE): you may share and adapt them for any purpose, including commercially, provided you give appropriate credit to Ryan Flavell, MAGNVS Ltd, link to the licence and say what you changed. DeepSeek-V4.1-Flash-1017 is MIT, copyright DeepSeek. Credits for third-party work are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The engine, scripts and tools of release 0.2 (`PROMETHEVS_0.2.zip`) are Copyright (c) 2026 Ryan Flavell, MAGNVS Ltd, licensed under the [Apache License 2.0](engine-0.2/LICENSE); every copy and derived work must keep the [NOTICE](engine-0.2/NOTICE) file. Third-party parts of the bundle keep their own licences ([engine-0.2/THIRD_PARTY_NOTICES.md](engine-0.2/THIRD_PARTY_NOTICES.md)).

## How to cite

If you use or build on this work, please cite:

> Ryan Flavell. *Compress-then-Correct Decoding: The Full DeepSeek-V4.1-Flash-1017 on a Five-Year-Old Gaming PC.* PROMETHEVS Technical Report 1, version 0.1. MAGNVS Ltd, October 2026. doi:10.5281/zenodo.23577839. https://doi.org/10.5281/zenodo.23577839

GitHub's "Cite this repository" button gives the same entry from [CITATION.cff](CITATION.cff).

## Acknowledgements

We build on DeepSeek's model and draft head and on the ggml and llama.cpp libraries. Ryan Flavell conceived the method and the ideas it tests, set the direction and the quality bar, decided what was built and tested, and checked the results. The engine code, the test runs and the drafts of this report were built together with AI coding assistants working under his direction.

---

<img src="figures/magnvs_logo_black.png" alt="MAGNVS Ltd" width="32" align="left"> PROMETHEVS is developed by MAGNVS Ltd.
