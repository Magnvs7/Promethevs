# PROMETHEVS 0.2: release notes

Ryan Flavell (MAGNVS Ltd) · installable engine server · research preview · October 2026

## What this is

0.1 published the report and its logs. 0.2 adds the engine itself, so anyone with a PC like the development PC can run DeepSeek-V4.1-Flash-1017 locally with compress-then-correct decoding: a 2-bit copy of the experts answers every word, the model's own margin flags the words it may have got wrong, and those words are recomputed with the original weights. The method, guess-and-check drafting and the memory layout are Ryan Flavell's ideas; the engine was built with AI coding assistants under his direction.

The bundle serves the model as an OpenAI-compatible endpoint at `http://127.0.0.1:8100/v1` (model `deepseek-v4.1-flash-1017-promethevs`), with streaming and tool calls, so existing apps can use it as a local provider.

## What is in the bundle

- The engine server and its GPU libraries (prebuilt, sha256 in `data\BINARIES.sha256`), the OpenAI-compatible proxy, start and stop scripts.
- A hardware check that refuses clearly if the PC falls short.
- A store builder that turns DeepSeek's original weights (downloaded from DeepSeek's official repository, pinned revision) into everything the engine loads, and checks every 2-bit and correction record against the hashes shipped here. You do not have to trust our stores: you build them yourself.
- `verify_install.py`, which checks the running server and measures speed the same way the report does.
- The frozen link inputs of the engine and the source of the pack tools (`source\`).
- `INSTALL.md`: requirements, time and disk, every step, and the known gaps.

## Requirements (short)

64 GB RAM, an NVIDIA GPU with 12 GB, about 1.1 TB free across NVMe drives, Windows, a current NVIDIA driver. The download is about 475 GiB and the store build about 4 hours on the development PC. Details in `INSTALL.md`.

## Measured on the development PC (23 Oct 2026)

Ryzen 9 3950X, 64 GB DDR4, RTX 3060 12 GB, two NVMe drives. The bundle's own scripts started the server; `verify_install.py --speed` passed every functional check (health, models, chat, streaming, tool call).

| Prompt (160 new tokens, warm second answer, generation only) | Run 1 | Run 2 | Report (0.1) |
|---|---|---|---|
| Code | 7.97 | 5.07 | 7.4 (best case) |
| Reasoning | 5.41 | 4.48 | 4.4-4.7 |
| Agent | 4.67 | 5.00 | 4.8-5.2 |

Speed varies from run to run because each answer's text, and so the number of words that need correcting, differs. The first word of an answer takes a few seconds more on short prompts and much longer on long ones. Quality figures are the report's: 98.33 percent agreement with the original weights over 960 audited words (99.17 percent counting only the words the engine did not check).

## Limits (nothing is papered over)

- A clean install from scratch (download, conversion and the full store build from this bundle on a second machine) has not been run yet. The one-layer store test rebuilt 128 records from the raw weights and all matched the shipped hashes.
- The engine exe is shipped prebuilt; a rebuild from source is not yet reproducible byte for byte (gaps G1 and G2 in `INSTALL.md`).
- One machine, one model, three prompts. Output is measured against the original weights, not yet against DeepSeek's reference implementation.
- Close other programs that use the graphics card before starting the engine: with too little free graphics memory, corrections slow from about 1.4 s to about 30 s.

## Licence

The engine, scripts and tools in this bundle: Copyright (c) 2026 Ryan Flavell, MAGNVS Ltd, licensed under the Apache License 2.0 (`LICENSE`). Every copy and every derived work must keep the `NOTICE` file, which names the author. The technical report stays under CC BY 4.0. Third-party parts keep their own licences (`THIRD_PARTY_NOTICES.md`).

## Credits

DeepSeek for the model (MIT). The ggml and llama.cpp authors and cpp-httplib (MIT). See `THIRD_PARTY_NOTICES.md`.

## How to cite

> Ryan Flavell. *Compress-then-Correct Decoding: The Full DeepSeek-V4.1-Flash-1017 on a Five-Year-Old Gaming PC.* PROMETHEVS Technical Report 1, version 0.1. MAGNVS Ltd, October 2026. doi:10.5281/zenodo.23577839. https://doi.org/10.5281/zenodo.23577839
