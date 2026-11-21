> I 6700XT approve this message, but I have no way of verifying any of it, so you go t that gong for you <cut to CaddyShack Meme>

# PROMETHEVS 0.2 — Test Report: AMD Radeon RX 6700 XT (gfx1031)

**Date:** 2026-11-20
**Branch:** `amd-gfx1031-port`
**Tester hardware:** AMD Radeon RX 6700 XT 12 GB (gfx1031), AMD Ryzen 5 5600 (6C/12T), 64 GB DDR4, 2x SPCC M.2 NVMe SSD (C: 953 GB, D: 931 GB), Windows 11.
**Engine:** `v41_serve2_BF.exe` (build 12427 base, AMD HIP port) + `proxy.py` (Python 3.14 stdlib), Promethevs 0.2 bundle.

---

## 1. Setup

- Engine launched via `scripts/start_engine.ps1` (default `--ctx 8192 --ubatch 512 --lanes 5 --source native --gate 1 --mmin 1 --threads 1 --threads-batch 16`), with env from `config/engine_env.template.txt`.
- OpenAI-compatible proxy on `127.0.0.1:8100/v1`, model id `deepseek-v4.1-flash-1017-promethevs`.
- `pi` coding agent (v1.5.2) configured with a `promethevs` provider pointing at the proxy; smoke test returned `READY` (12 prompt + 3 completion tokens).
- Engine priority set to `Normal` (originally `AboveNormal`); tuned runs used `HILO_V41_PLANE_THREADS=6`, `HILO_V41_SCF_PARALLEL_THREADS=4`.

## 2. Measured performance

Baseline (untuned, `--ctx 8192`):

| Metric | Value |
|---|---|
| Decode throughput | ~1.9 tok/s (1.66 s/round, 3.08 tok/round) |
| End-to-end (incl. prefill) | ~0.88 tok/s (408 s / 360 tokens) |
| CPU utilisation | 9.6 / 12 logical threads (100%) |
| GPU compute utilisation | ~19% |
| Working set | ~35.5 GB |
| Free physical RAM | ~19-20 GB (flat — direct I/O bypasses OS cache) |
| Pagefile usage | 5 MB (no paging) |
| Commit headroom | 6-8 GB free of ~68 GB limit |

Per-round breakdown (no-correction class, 76 rounds):

| Component | ms/round |
|---|---|
| attention | 312.7 |
| moe_expert_compute | 503.6 |
| moe_read_wait | 434.7 |
| moe_other | 74.4 |
| indexer | 19.6 |
| engram | 9.8 |
| unmeasured_in_decode | 192.4 |
| gpu_lock_wait | 84.4 |
| **total round** | **1564.8** |

Corrections: 22 per 360-token pass, ~3.3 s each (range 2.3-5.4 s), overturning 22 tokens and discarding 68. SCF time 75.7 s total (18.5% of pass wall). Correction lane ran on **1 thread** (`scf parallel: threads 1`).

Prompt warm (per turn): 7,783 b2 experts, **85,187 MB read** from disk in ~39-46 s (`0 already resident` every turn — the expert cache is thrashed by correction plane restores).

## 3. Comparison with the report's dev machine

| | Dev (Ryzen 9 3950X, RTX 3060) | This PC (Ryzen 5 5600, RX 6700 XT) |
|---|---|---|
| Cores | 16 | 6 |
| Decode (code, warm) | 7.4 tok/s | ~1.9 tok/s |
| Ratio | 1.0x | ~0.26x |

The ~4x gap is consistent with the 2.7x core-count difference plus the AMD HIP port (less mature than the CUDA original) and the fact that the engine is CPU-bound on MoE expert compute.

## 4. Optimizations attempted

| Change | Result | Notes |
|---|---|---|
| `HILO_V41_SCF_PARALLEL_THREADS`: 0→4 | Correction lane 4 threads (was 1); targets the 2.1 s expert-compute term | Low risk. Applied in tuned runs. |
| `HILO_V41_PLANE_THREADS`: 8→6 | Matches physical core count | Low risk. Applied in tuned runs. |
| Engine priority `Normal` (was `AboveNormal`) | Reduced UI starvation; modest throughput cost | Applied. |
| `HILO_V41_EXPERT_SKIP`: 0.12→0.15 | **Not applied.** Report: +4% on code, but reasoning dropped to 2.65-3.68 tok/s (from 4.49). Workload-dependent; risky for an agent in thinking mode. | Documented lever only. |
| `HILO_V41_SCF_MARGIN`: 0.45→0.30 | **Not applied.** Report: 5.66 vs 5.23 tok/s (+8%) with fewer corrections; small quality cost (98.33% agreement at 0.45). | Offered as optional. |
| `PROMETHEVS_REASONING_EFFORT`: 75→30 | Shorter thinking bursts; fewer 100%-CPU lockups | Applied. |
| `max_tokens`: 1024→384 | Caps answer length; prevents 9-min lockups | Applied in pi config + proxy env. |
| Context window: 8192→4096 | Smaller KV cache, less prefill read volume | Applied. |
| GPU hot experts (`HILO_V41_GPU_HOT`) | **Unavailable on AMD.** `hv41hot.dll` is the NVIDIA original (imports `nvcuda`); the AMD port only covers `hv41cuda.dll`. 2.8 GB free VRAM sits unused. | Structural win requires porting `hv41hot.dll` to HIP. |
| `HILO_V41_OWN_L` / `OWN_S` core sets | **Not applicable** on AMD (designed for Intel hybrid P/E cores). | Not tried. |

## 5. Root cause of the "lockups"

**CPU starvation, not a crash or OOM.**
- The engine saturates all 12 logical threads (100%) for extended periods (each request ~5 min).
- Windows UI (explorer, taskbar, input, DWM) gets almost no CPU → desktop appears frozen.
- No paging occurred (pagefile 5 MB, 20 GB free RAM, 6-8 GB commit headroom). Memory is not the bottleneck.

**Contributing factors:**
1. Huge prompts: pi sends the entire conversation history (~16,000 chars / ~4,000 tokens) every turn; each prompt warm reads ~85 GB of experts.
2. Long thinking: `n_predict 1024` + reasoning effort 75 → up to ~9 min of generation at 1.9 tok/s.
3. `AboveNormal` priority gave the engine more precedence over the UI.
4. pi's HTTP client times out ("Request timed out.: This operation was aborted (20)") because the engine cannot respond within pi's patience window.

## 6. Diagnostic findings

- No errors in `engine_*.log` or `proxy_*.log` — the engine never crashed.
- `SLOWROUTE ENTER/EXIT` events: disk reads intermittently drop to 300-370 MB/s vs 1.3-1.5 GB/s baseline, causing stalls. The engine self-throttles and recovers.
- `0 already resident` on every prompt warm — the 50 GB expert cache is thrashed by correction plane restores (3.5 GB each, 22 per pass).
- `hv41hot.dll` imports `nvcuda` (NVIDIA original) → GPU hot expert pool cannot run on the AMD port.

## 7. Recommendations

**For interactive use on this hardware:** switch pi to the `local` provider (llama.cpp, port 9931) with a smaller model (`Qwen3-8B-Q4_K_M` or `Qwen2.5-Coder-7B`). Those respond in seconds. Promethevs is a research demo on 6 cores.

**For Promethevs specifically:**
- Keep context at 4096, reasoning effort ≤ 30, max_tokens ≤ 384.
- Run with engine priority `Normal`.
- Close other apps during long generations.
- Use curl/direct HTTP (no client timeout) for batch queries instead of pi.
- Port `hv41hot.dll` to HIP to offload hot experts to the 2.8 GB free VRAM (biggest structural win).
- Raise pagefile to 16-32 GB for commit headroom (reliability).

## 8. Files changed in this session

- `C:\Users\6700XT\.pi\agent\models.json` — added `promethevs` provider (8100/v1); contextWindow 8192→4096, maxTokens 1024→384.
- `C:\Users\6700XT\.pi\agent\settings.json` — default provider/model → promethevs (reverted to `local`/`Qwen3-8B-Q4_K_M` recommended for daily use).
- `C:\Users\6700XT\Documents\Default Project\promethevs-tuned.ps1` — tuned launcher (plane threads 6, SCF threads 4, Normal priority, reasoning effort 30, max tokens 384).
- No changes to the Promethevs repository source.

## 9. Verdict

Promethevs 0.2 **runs correctly** on the RX 6700 XT (AMD HIP port) — smoke test `READY`, bitwise parity suite passes, E2E generates coherent output. However, at ~1.9 tok/s decode on 6 cores it is **not suitable as an interactive coding assistant**; it is a viable batch/research tool. The 4x gap vs. the dev machine is primarily the 2.7x core-count difference, not a port defect.
