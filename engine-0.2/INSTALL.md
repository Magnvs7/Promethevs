# PROMETHEVS 0.2: installable engine server

PROMETHEVS runs DeepSeek-V4.1-Flash-1017, a frontier mixture-of-experts model, on one ordinary Windows PC (64 GB RAM, one 12 GB NVIDIA GPU) and serves it as an OpenAI-compatible endpoint on `http://127.0.0.1:8100/v1`.

This bundle holds the prebuilt programs, a hardware check, a store builder that turns DeepSeek's **original weights** into everything the engine loads, and the start/stop scripts. You do not have to trust our stores: you build them yourself from DeepSeek's files, and every 2-bit and plane record is checked against the hashes we shipped.

> **Status of this document (read this first).** Release 0.2 was assembled on 22 Oct 2026 from the release's pipeline record. What has actually been run: the hardware check (on the development PC, read-only: it passes with the stores already built, and refuses a fresh 1.1 TB install because the free space is short), the orchestrator's `--plan` and `--dry-run --run` (every command line printed, nothing executed), the store check on the development PC's existing stores (all 9 pack files, all index entries, GGUF, DLLs, routes and exptable present), and `verify_install.py` against a fake server. **Run on 22 Oct with the owner's go (CPU only, no GPU): the one-layer store test, which rebuilt 64 b2 records and 64 plane records of layer 20 from the raw weights into a scratch folder; all 128 records equal the shipped reference in every column except the path (format, record size and per-record hash). **Not yet run:** the download, the converter, the full pack builds, the installed server answering requests (the development PC's live server is in use), and the speed run. Nothing here claims that a clean from-scratch install has been completed.

> **Run on 23 Oct 2026 on the development PC (existing stores):** the bundle's own `start_engine.ps1` and `proxy\proxy.py` served requests and `verify_install.py --speed` passed every functional check (health, models, non-streamed and streamed chat, tool call). Two back-to-back speed runs on one engine, warm second answer, generation only: code 7.97 and 5.07 words/s, reasoning (r22) 5.41 and 4.48, agent 4.67 and 5.00. Every correction took 1.0-1.7 s (one 3.1 s). **Still not run:** the download, the converter and the full store build from this bundle on a clean machine (G22).

## 1. Requirements

| Item | Needed | Why |
|---|---|---|
| OS | Windows 10 (1809) or 11, 64-bit | the programs and drive tools are Windows-only |
| CPU | x86-64 with **AVX2** (Intel Haswell+/AMD Zen+); 16 cores / 32 threads recommended | the engine and pack tools are built `/arch:AVX2`; the release was tuned on a Ryzen 9 3950X |
| RAM | **64 GB** | the expert cache is 50,000 MB; the engine's working set is about 50 GB |
| GPU | NVIDIA, **12 GB**, compute capability **8.6-8.9** (RTX 30/40-series class, e.g. RTX 3060 12 GB) | the shipped GPU DLLs contain `sm_86` code only |
| Driver | current NVIDIA driver (provides `nvcuda.dll`); release built with CUDA 13.4 and run on driver 616.92, so use driver 580 or newer | the CUDA runtime is linked statically; only the driver is needed |
| Disk | **about 1.09 TB (1,013 GiB)** free, on **NVMe** drives: one drive with about 1.14 TB free, or two drives (A: 561 GB, B: 617 GB free, both numbers include a 45 GB free-space floor) | see section 3 |
| Python | Python 3 on PATH for the proxy and the build scripts (developed on 3.14); **Python 3.13** installed for the GGUF converter's venv | the proxy and orchestrator are standard-library only |
| Internet | for the weights (510 GB) and, once, for the converter's pip packages (about 3 GB) | see section 4 |

`scripts\check_hardware.ps1` checks all of this, refuses clearly if the PC falls short, and writes the chosen drives to `config\promethevs.json`. It only reads system information.

## 2. Install in short

```powershell
# 1. extract the bundle anywhere (examples use C:\promethevs_pkg), then:
cd C:\promethevs_pkg
powershell -ExecutionPolicy Bypass -File scripts\check_hardware.ps1            # picks the drives, writes config\promethevs.json
#    weights already on this PC?  add:  -WeightsPresent -RawDirs "D:\dsv41_1017"
#    stores already built too (a machine that ran the release)?  -ExistingStores paths.json   (a JSON with a "paths" object: gguf, raw_dirs, index_dir, ...)
# 2. the weights revision is already pinned in the config (section 4)
# 3. see the plan, then build (resumable: run the same command again after any interruption)
powershell -ExecutionPolicy Bypass -File scripts\build_stores.ps1 --plan
powershell -ExecutionPolicy Bypass -File scripts\build_stores.ps1 --run
# 4. start / stop
powershell -ExecutionPolicy Bypass -File scripts\start_engine.ps1               # loads in about 55-80 s
python scripts\py\verify_install.py                                             # section 8
powershell -ExecutionPolicy Bypass -File scripts\stop_engine.ps1
```

Point any OpenAI-compatible client at `http://127.0.0.1:8100/v1` (model `deepseek-v4.1-flash-1017-promethevs`, any non-empty API key). `GET http://127.0.0.1:8100/health` and `:8099/health` tell you the proxy and the engine are up. The app contract (one request at a time, streaming, 15-minute client timeout, tools, thinking mode) is in `APP_INTEGRATION.md`.

## 3. Time and disk

**Disk** (measured sums of the release stores, GB = 10^9 bytes):

| Part | GB | Where |
|---|---|---|
| Raw DeepSeek weights (kept: the engine reads the Engram tables and expert offsets from them at run time; **never delete them**) | 510.3 (475 GiB) | drive A 237.3, drive B 273.1 |
| GGUF (non-expert weights) | 18.0 | A |
| 2-bit packs + correction planes | 180.1 on A, 168.9 on B | A, B |
| FP4 copies (layers 0-17), w2/w3 mirror | 48.4 + 81.5 | B |
| FP4 copies (6-10,13,15,16), reverse mirror (18-39) | 21.5 + 59.2 | A |
| **Derived stores** | **577.7 (538 GiB)** | |
| **Everything** | **1,088 (1,013 GiB)** | A 516 GB, B 572 GB |

Peak equals the final total (nothing is deleted during the build). The tools stop cleanly at a 45 GB free-space floor per drive. If the weights are already on the PC, only the 578 GB of derived stores are written.

**Time** (`scripts\build_stores.ps1 --plan` prints the table for your config, and `--run` prints the time left at every step):

| Part | Time | Basis |
|---|---|---|
| Download 510 GB | about 2 h 15 min at 500 Mbit/s (11 h at 100 Mbit/s, 1 h 10 min at 1 Gbit/s) | arithmetic; the 0.1 scripts never logged a download time |
| sha256 of the 49 large files while downloading | adds about 10-40 min | estimate [E] |
| Python venv + pip (converter) | about 10 min | estimate [E] |
| Expert index + GGUF | 14 s + 242 s | measured |
| 5 b2 packs | 6,434 s | measured with the raw weights on an HDD; from NVMe expect less [E] |
| 3 plane packs | 6,020 s | same |
| Draft pack, FP4 copies, mirrors | 75 + about 1,330 s | measured |
| **Store build in all** | **about 4.0 h measured (HDD source); about 2 h from NVMe by the original script's estimate [E]** | |

## 4. What is downloaded, from where

| What | From | Size | When |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017 weights, tokenizer, config, vendor `encoding/` and `inference/` code: 83 files | Hugging Face `deepseek-ai/DeepSeek-V4.1-Flash-1017`, `https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/<revision>/<file>` (curl.exe, resumable) | 510,311,045,295 bytes | build step `weights` |
| Python packages for the GGUF converter: numpy 2.5.3, torch 2.14.1, transformers 5.20.1, gguf 0.19.0, safetensors 0.8.0, tokenizers 0.23.3 (the versions of the venv that built the release GGUF) | PyPI | about 3 GB | build step `venv` |

Nothing else is downloaded. The prebuilt programs, lists, reference indexes and the proxy are in this bundle.

**Pinned revision (gap G20, closed).** The 0.1 scripts used the branch name `main`; this bundle pins commit `1bae0e1088993b9b82de08f59de5be5a34e42170` (looked up once on 22 Oct 2026: 83 files, every size and all 49 sha256 equal `data\hf_manifest.json`). The downloader still refuses to run unless `config\promethevs.json` -> `hf.revision` holds a 40-hex commit id, and it checks every file against the manifest whatever the revision says.

**Weights already present.** `check_hardware.ps1 -WeightsPresent -RawDirs "C:\a;F:\b"` records the folders (any split over drives; the engine takes a `;` list). The `weights` step then only verifies sizes (`download_weights.py --verify --hash` also checks sha256, which reads 510 GB) and nothing is downloaded.

## 5. The store build (what `build_stores.ps1 --run` does)

The 23 steps below are the 18 ordered steps of the inventory (column "inv"). Each is skipped when already done, so an interrupted build continues where it stopped. Progress, elapsed time and ETA are printed; logs are in `work\logs\`.

| Step id | inv | What | Verification before it is marked done |
|---|---|---|---|
| `weights` | 1-2 | download (pinned revision) or verify existing weights | sizes; sha256 of the large files |
| `venv` | 5 prep | Python 3.13 venv with the converter's packages | pip exit code |
| `xindex` | 4 | expert index from the shard headers (`--mtp`) | 15,744 lines and **byte-identical to the shipped index** |
| `gguf` | 5 | non-expert GGUF with vocabulary and Engram constants | 996 tensors, "0 differ" read-back, exactly 17,960,145,824 bytes |
| `b2hot` `b2f` `b2r0` `b2r1` `b2r2` | 6 | five 2-bit packs (`b2pack.exe --verify`, 14 threads) | tool OK line, exact pack size, **every record's hash equals the shipped reference** |
| `b2idx` | 7 | b2 split index (merge) | 15,360 unique rows, all equal to the shipped reference except the path |
| `plh` `plf` `plc` | 8 | three plane packs (`planepack.exe --verify`) | as above, plus the codebook header |
| `plidx` | 9 | plane index (merge) | 15,360 unique rows, equal to the shipped reference except the path |
| `draft` | 10 | draft-head b2 pack, layers 40-42 (`b2pack_draft.exe`), draft split index | 384 rows = layers 40-42 x 128, paths compared **normalised** (see G3), equal to the reference |
| `fpack1` `cpack1` | 11, 12 | FP4 copies for layers 0-17 and 6-10,13,15,16 | every expert read back unbuffered and compared (blake2b) by the tool; layer list equals the reference |
| `rmirror` | 13 | reverse mirror, layers 18-39 | byte-for-byte read-back by the tool |
| `selidx` | 14 | selection index (cpack1 + rmirror, 30 layers) | layers equal the reference |
| `mirror` | 15 | w2/w3 mirror, layers 0-17 | SHA-256 of source == read-back, per layer |
| `routes` | 16 | hot-route file (shipped data) | see G7 |
| `programs` | 17 | prebuilt programs | sha256 against `data\BINARIES.sha256` |
| `check` | 18 | every file the engine loads, and every file the indexes name, exists with a plausible size | |

**Why the per-record hash comparison is the verification.** The b2 and plane records are a deterministic function of the weights, and each index line carries a hash of its record. If the same weights and lists were used, all 15,360 + 15,360 index lines must equal the shipped copies in every column except the path. A different weight revision, a corrupted download or a tool problem shows up as hash differences and the step fails, with the first differences printed. `--no-reference-check` turns this off (steps are then recorded `done-unverified`), for experiments with other weights.

Resume rules: `b2pack.exe` continues from its index; the other tools restart their unfinished file. An unfinished file is renamed `*.partial_<time>` (never deleted; delete these yourself to get the space back). Other commands: `--only b2f,plf`, `--from b2r0`, `--redo`, `--status`, `--verify`, `--dry-run --run` (prints every command and runs nothing).

## 6. Layout of this bundle

```
bin\        v41_serve2_BF.exe, hv41cuda.dll, hv41hot.dll (engine, GPU kernels), b2pack.exe, b2pack_draft.exe, planepack.exe (store tools),
            vcomp140 / vcruntime140 / vcruntime140_1 / msvcp140 .dll (VC++ runtime, so no separate install is needed)
config\     promethevs.json (written by check_hardware.ps1: the ONE place every path comes from), engine_env.template.txt (the engine's settings)
data\       exptable.bin, lists\ (expert lists, G6), reference_indexes\ (the release's indexes: the verification target), hot_routes\, prompts\,
            reference_texts\ (the report's stored generations), hf_manifest.json, BINARIES.sha256, chat_template_v41.jinja
proxy\      proxy.py, encoding.py (DeepSeek's vendor encoding, unchanged)
scripts\    check_hardware.ps1, build_stores.ps1, start_engine.ps1, stop_engine.ps1, py\ (orchestrator, downloader, patched builder copies, tests)
pylib\      the gguf python package the converter was built against
source\     tool sources (b2pack, planepack) and their build scripts, for reference
state\      saved-context folder (see G8 / G19)    indexes\ tmp\ work\ logs\   created at run time
```

## 7. Run

`scripts\start_engine.ps1` fills `config\engine_env.template.txt` from your config, starts `bin\v41_serve2_BF.exe` (port 8099) and `proxy\proxy.py` (port 8100) and waits for both `/health`. Exit codes: 0 ready, 2 already running, 3 a file is missing (`-Check` lists them without starting anything), 4 engine exited during load, 5 timeout. `stop_engine.ps1` stops it (`-KeepProxy` leaves the proxy). Close other GPU programs first, and start it as a normal process, not from a background or low-priority job: a low-priority start made every correction take about 36 s instead of 1.4 s on the development PC. The engine also keeps 384 MB of graphics memory free beside its GPU expert pool (`HILO_V41_GPU_HOT_RESERVE_MIB=384` in the env template): with less room, graphics memory taken by other programs pushed corrections to about 30 s.

## 8. Verify speed and quality yourself

With the engine freshly started and the PC otherwise idle:

```powershell
python scripts\py\verify_install.py --speed
```

It checks `/health`, `/v1/models`, a non-streamed and a streamed chat, and a tool call (`get_weather`, London). With `--speed` it sends each prompt in `data\prompts\` **twice** (160 tokens) and prints the second, warm answer's speed: the request time, and the engine's own generation-only speed (first-word step excluded, which is how the report measures). The report's figure on the code prompt is **7.4 words/s** warm, a best case; on 23 Oct two runs in a row on one engine gave 7.97 and 5.07, because each run's text, and so its number of corrections, differs. The check passes when code reaches at least **4.5 words/s** generation-only, which a healthy install clears and a broken one (for example corrections taking 30 s) does not. The first answer after a start is slower (about 4.2-4.3/s) because the cache is cold.

Quality: the three prompts of the report's final runs are in `data\prompts\` (`code_ipv4`, `r22_task`, `agent_1`); the report's stored generations (token ids and decoded text, both passes) are in `data\reference_texts\`. `verify_install.py --speed` prints, per answer, whether it is identical to the report's pass 1 or pass 2 text and how many words match before the first difference (after one different word the rest of an answer takes a new path, so this count is not a quality score). Greedy decoding with corrections is not required to be identical across passes (the report's own pass 1 and pass 2 differ in places), so read the agreement figures against the report's logs rather than expecting 100 percent. The report's quality figures come from a separate word-by-word audit against the exact path: 98.33 percent agreement over 960 words across the three prompts (99.17 percent counting only the words the engine did not check).

To check that your **stores** are the release stores without running the model, `scripts\py\one_layer_test.py --outdir D:\p_test` rebuilds a 64-expert slice of one layer's b2 and plane records into a scratch folder and compares each record's hash with the shipped reference (about 1.3 GB read, 1-2 minutes, CPU only).

## 9. Known gaps (nothing is papered over)

| Gap | State in this bundle |
|---|---|
| **G1** the release exe links a frozen engine object made by an 11-patch chain on a scratch copy; the recipe is not proven to reproduce it | **Open, inputs preserved.** Prebuilt binaries are shipped (sha256 in `data\BINARIES.sha256`). The frozen link inputs are in `source\frozen\` with `MANIFEST.sha256` (engine object `deepseek41.cpp.obj`, BF core headers, `v41_serve2.cpp`, `build_serve2_ba.cmd`, `llama.lib`, ggml libs, `cpp-httplib.lib`), so the exe can be relinked with MSVC 14.51 (the link line is in `build_serve2_ba.cmd`; it expects the original `F:\llm\native\glm5hilo` include folders, so edit its paths). A rebuild will not match the exe's sha256 (inventory G1). Regenerating the object itself from source (the patch chain) is not reproducible yet. |
| **G2** the llama.cpp base of the engine libraries is unpinned (no `.git`, no recorded commit, the `llama.lib` build command is not recorded) | **Open.** Not fixable without the owner. |
| **G3** the draft pack's own script reported STOPPED (an awk backslash false negative), yet the pack exists | **Re-implemented, not yet run.** `build_stores.py` step `draft` compares the paths normalised, checks the 384 keys and compares every record's hash with the shipped reference; it does not reuse the 0.1 script. The shipped reference rows are the ones from the hand-finished pack. |
| **G4** stage B (packs) ran by hand; its command lines are reconstructed from the script and logs | **Open (marked).** The commands in `build_stores.py` are the inventory's reconstruction; thread counts for the plane packs (14) come from the script. The hash comparison against the shipped reference is what makes this safe to use: a wrong command shows up as a hash difference. |
| **G5** hard-coded `G:\`, `C:\`, `F:\` and old-model paths | **Fixed in the bundle's copies only** (`scripts\py\*_1017.py`, `v41_common.py`, the env template, the start script): every path comes from `config\promethevs.json`. The pack tools get `--expert-index` and `--dirs` explicitly (their compiled-in defaults point at the old model). The originals were not touched. |
| **G6** the expert lists are 8 Oct picks from the old model's routing | Shipped as data (`data\lists`). They decide placement only, not values. |
| **G7** the hot-route file needs finished stores and a model run | Shipped as data (`data\hot_routes`, copied to `indexes\`). Placement only. Running without it (`GPU_HOT=auto` fallback) is untested. |
| **G8** `exptable.bin` is read from a compiled-in `F:/llm/...` path | Shipped in `data\`. The strings `HV41C_EXPTABLE` (exe and DLL) and `V41_STATE_DIR` (exe) exist in the binaries, so the env template sets both. **They are not run-verified** (G19). |
| **G9** index files contain absolute drive-letter paths | The builder writes fresh indexes with your paths; the comparison ignores the path column. |
| **G10** no whole-file hash of the GGUF, mirrors, FP4 copies | Verified per tensor (GGUF), per layer (mirror sha256), per expert (copies) by the tools. |
| **G12** the converter's Python environment had no requirements file | Pins taken from the dist-info of the venv that built the release GGUF (section 4). |
| **G19** (new) `exptable.bin` and the state folder defaults are compiled in | See G8: confirm in the acceptance run that no `F:\llm\state` is created and the engine loads the bundle's `exptable.bin`. If the override variables are not honoured, the engine needs the bundle at the original paths. |
| **G20** weights revision not pinned | **Closed 22 Oct 2026.** Pinned to commit `1bae0e1088993b9b82de08f59de5be5a34e42170` (main, last modified 2026-10-17). One Hugging Face metadata lookup returned 83 files whose sizes and all 49 sha256 values equal `data\hf_manifest.json`, the manifest of the download that built the release stores. |
| **G21** which vendor code the GGUF converter uses | **Release inputs pinned (owner decision).** The release GGUF was built with the OLD checkpoint's `inference/model.py` and `chat_template_v41.jinja` (inventory G5). The bundle ships those exact files as data (`data\converter_vendor_old`, `data\chat_template_v41.jinja`; sha256 in `data\BINARIES.sha256`, checked by build step `programs`). **Two gaps stay open:** (a) whether converting with the 1017 `model.py` gives a byte-identical GGUF is untested (set `PROMETHEVS_CKPT` to try; a hardware item); (b) the template was never re-checked against the 1017 `encoding.py`. The conversion itself has not been run from this bundle. |
| **G22** (new) the build, store test, clean install and speed run have not been executed from this bundle | See "Status" at the top. Needs hardware time and a spare terabyte; the owner decides. |

## 10. Troubleshooting

* `REFUSED` from `check_hardware.ps1`: it prints each shortfall in numbers. A drive that reports bus type `Unknown` (virtual disks, some RAID) is not accepted as NVMe: name it with `-DriveA/-DriveB` and add `-AllowNonNvme` if you accept slower, unsupported behaviour. HDDs are not supported.
* `build_stores.ps1 --run` stops with `STEP x FAILED`: the message says why; fix it and run the same command again.
* `start_engine.ps1` exit 3: run with `-Check` to list the missing files; run `build_stores.ps1 --status`.
* Corrections take tens of seconds (`correction latency` in `logs\engine_*.log`) or code runs far below 4.5 words/s: close other programs that use the graphics card (browsers with hardware acceleration, overlays, games), restart the engine, and check the `gpuhot:` line in the log; the free memory after the pool should be about 480 MB or more.
* Nothing the bundle does deletes files. Moved-aside `*.partial_*`, `*.badsize_*`, `*.badhash_*` files are yours to remove.
