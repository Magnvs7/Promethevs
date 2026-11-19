# PROMETHEVS 0.2 — AMD `gfx1031` (RDNA 2) Windows HIP Port

Native Windows 11 HIP port of `hv41cuda.dll` (`38` exports across `10` in-memory COMGR-compiled gfx1031 HIP kernels) and host binary rebuild for `PROMETHEVS_0.2` on **AMD Radeon RX 6700 XT (12 GB, `gfx1031`)** — compiled with standard MSVC `cl.exe` (`VS 2022 BuildTools`) and executed directly through AMD's driver-shipped `amd_comgr_2.dll` + `amdhip64_6.dll` (no Linux, no WSL, no HIP SDK installer required).

---

## ⚠️ Prerequisite: Windows Smart App Control (`SAC=0`)

Windows 11 **Smart App Control** (`VerifiedAndReputablePolicyState = 1`) **blocks locally compiled unsigned binaries** (`hv41cuda.dll` and `v41_serve2_BF.exe`) with:
- **`exit 4551` / `OSError: [WinError 4551] An Application Control policy has blocked this file`**
- **CodeIntegrity / Device Guard exit `1073751882` (`0x4000001A`)**

Even rebuilding from a regular PowerShell window or elevated Admin terminal is non-deterministically blocked while Smart App Control is `On` (`1`).

**Required reproduction step (one-time host change):**
1. Open **Windows Security** → **App & browser control** → **Smart App Control settings**.
2. Set Smart App Control to **Off** (`VerifiedAndReputablePolicyState = 0`) and approve the UAC prompt.
   - *Note:* Turning Smart App Control **Off** is a **one-way Windows 11 decision** (Microsoft only allows re-enabling it via Windows reset/reinstall). Windows Defender and SmartScreen stay active.
   - Takes effect **immediately** (no reboot required).
3. Verify in PowerShell before building/staging:
   ```powershell
   (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" -Name "VerifiedAndReputablePolicyState").VerifiedAndReputablePolicyState
   # Expected: 0
   ```

---

## Files in `amd_hip/`

| File | Purpose |
|---|---|
| `hv41cuda_hip.cpp` | C ABI exports (`38` functions) + embedded gfx1031 HIP kernel sources compiled at runtime via `amd_comgr_2.dll` |
| `build_promethevs_amd.cmd` | Builds `hv41cuda.dll` + links `v41_serve2_BF.exe` against frozen `deepseek41.cpp.obj`; `--stage` copies to `PROMETHEVS_0.2\bin\` (preserving `*_orig` backups) |
| `test_hv41c_bitwise.py` | Bitwise `uint32_t` parity suite (`DirectHipHv41c` WHQL path + staged `hv41cuda.dll` C ABI path) |
| `test_hv41c_bitwise.cpp` | Native C++ smoke harness for `hv41cuda.dll` |
| `build_and_test.cmd` | Quick native C++ build & test helper |
| `include/` | Reconstructed host headers (`ggml.h`, `llama.h`, `v41_utility.h`, `httplib.h`, `nlohmann/json.hpp`) |

---

## Build, Stage & Verify (Windows 11)

1. Confirm `VerifiedAndReputablePolicyState = 0` (see above).
2. Build and stage into `PROMETHEVS_0.2\bin\`:
   ```cmd
   build_promethevs_amd.cmd --stage
   ```
3. Run the full bitwise parity suite against the staged DLL:
   ```powershell
   uv run --with numpy --no-project --python "C:\Users\6700XT\AppData\Local\Programs\Python\Python313\python.exe" python test_hv41c_bitwise.py "..\release_0.2\extracted\PROMETHEVS_0.2\bin\hv41cuda.dll"
   ```
4. Launch the engine server + OpenAI proxy (`http://127.0.0.1:8100/v1`) from `release_0.2\extracted\PROMETHEVS_0.2\`:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_engine.ps1 -Ctx 2048 -Ubatch 256
   ```

---

## Verified Baseline (`SANDBOX`, RX 6700 XT 12 GB, 2026-11-18)

- **Bitwise suite (`test_hv41c_bitwise.py` on staged `hv41cuda.dll`):**
  - Phase 1/1b FP8 (`hv41c_linear`): **`5,320,704`** outputs checked, **`0`** mismatches (`74.85 GB/s` peak on `lm_head_chunk` `32768x1280` `M=1`).
  - Phase 2 optional kernels (`bf16_*`, `sattn` incl. `M=31` `[M, topk]` layout, `seq_*`, `corr_abi` FP8+FP4): **`172,288`** outputs checked, **`0`** mismatches.
- **Live DeepSeek-V4.1-Flash-1017 E2E (`SAC=0`, `-Ctx 2048 -Ubatch 256`):**
  - **9-token smoke (`prompt_n=15`):** Replied `PROMETHEVS_AMD_OK` (`1.156 tok/s`, `7/8` speculative accepted, `0/2,347,008` GPU self-check mismatches, `0/46,080` `gpu shx` mismatches).
  - **64-token sustained (`prompt_n=31`, `max_tokens=64`):** Generated valid Python `sha256_file` (`0.633 tok/s` overall incl. one `41.3 s` `scf history` overturn; `1,904.4 ms/round` decode wall, `66/69` speculative accepted), with **`650` GPU `gpucorr` experts**, **`515` GPU planes**, **`40` GPU `sattn` calls (`M>1`)**, **`0` `load_gc overflow`**, **`0` `load_fp4 overflow`**, **`0` `gpucorr` errors**, and **`104.6 MiB` `gpucorr` VRAM** (`2,882 MiB` free vs `2,487 MiB` prior = **`+394.6 MiB` VRAM freed**).
