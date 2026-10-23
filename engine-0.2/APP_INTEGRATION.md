> Adapted for the 0.2 bundle: paths are relative to the bundle folder. Written 21 Oct 2026 on the development PC (Ryzen 9 3950X, 64 GB, RTX 3060 12 GB); the measured figures are from that PC.

# PROMETHEVS app: running the engine locally (for the laptop / app sessions)

Written 21 Oct 2026 by the engine owner session on the desktop PC (Ryzen 9 3950X, 64 GB, RTX 3060 12 GB).

**Goal (Ryan):** install the PROMETHEVS app (the Rust/Tauri app, formerly OPVS) on the desktop PC.
The app then runs the PROMETHEVS engine on that PC and talks to it as a local model provider.

## 1. What is already on the desktop PC

| Piece | Where | What it does |
|---|---|---|
| Engine program | `bin\v41_serve2_BF.exe` | Runs DeepSeek-V4.1 1017 with compress-then-correct (the release engine, BF). It serves on `127.0.0.1:8099`. |
| OpenAI-compatible proxy | `proxy\proxy.py` (Python 3.14, stdlib only) | Serves `127.0.0.1:8100/v1`. It uses DeepSeek's own prompt encoding, so tools, thinking mode and reasoning effort work. |
| Start script | `scripts\start_engine.ps1` | Sets the engine's settings (`config\engine_env.template.txt`), starts the engine and the proxy, and waits until both answer. No Git Bash needed. |
| Stop script | `scripts\stop_engine.ps1` | Stops the engine. Add `-KeepProxy` to leave the proxy running. |
| Model data | the drives chosen in `config\promethevs.json` (stores, GGUF, raw weights, indexes) | Stays where it is. **Never bundle it into the app installer.** |

The app does **not** need to contain the engine. It starts the engine with the start script and talks HTTP.

## 2. Provider settings in the app

| Setting | Value |
|---|---|
| Type | OpenAI-compatible (Chat Completions) |
| Base URL | `http://127.0.0.1:8100/v1` |
| Model id | `deepseek-v4.1-flash-1017-promethevs` (also returned by `GET /v1/models`) |
| API key | any non-empty text; it is not checked |
| Streaming | yes: SSE `chat.completion.chunk`, ending with `data: [DONE]` |
| Tools | yes: OpenAI `tools` in, `tool_calls` out, with `finish_reason: "tool_calls"` |
| Max tokens | send `max_tokens`. The default is 1024. |
| Context | 8,192 positions. Start the engine with `-Ctx` to change it, which costs VRAM/RAM. |
| Thinking / effort | See INSTALL.md section 8. Effort is 1-100 or low/high/max; the default is 75. |

The proxy and engine listen on 127.0.0.1 only. The app must run on the same PC.

## 3. Engine lifecycle from the app (Rust)

1. **Already up?** `GET http://127.0.0.1:8100/health` and `GET http://127.0.0.1:8099/health`. If both answer 200, use them.
2. **Start** it as a normal child process. Do not start it from a background, low-priority or job-limited context (see section 5):

   ```
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File <bundle>\scripts\start_engine.ps1
   ```

   - Exit codes: `0` ready, `2` already running, `3` missing file, `4` the engine exited while loading, `5` timeout.
   - Loading takes **55-75 s** [measured]. Show a "loading model" state. With `-NoWait` the script returns at once, and the app polls `/health` itself.
3. **Stop:**

   ```
   powershell.exe -NoProfile -ExecutionPolicy Bypass -File <bundle>\scripts\stop_engine.ps1
   ```

   Stop it when the user quits, or offer "keep the engine running": the cache stays warm only while it runs.
4. **Logs:** `<bundle>\logs\engine_<time>.log` (stderr) and `.log.out` (stdout).

## 4. Behaviour to design around

These are measured on 21 Oct: code prompt, 160 words, corrections on.

- **One request at a time.** The app must queue requests and never send two at once.
- **First answer after start:** about 4.2-4.3 words/s, because the model's parts load from disk.
- **Later answers:** about 7.0-7.7 words/s while the engine stays up, because the warm cache stays loaded. Keep it running between chats.
- **First word of a short follow-up:** about 2.1-2.6 s.
- **Long new prompts:** take much longer before the first word.
  - A 1,163-token agent prompt took about 80-130 s to prepare in earlier tests.
  - Set the HTTP client timeout to at least **15 minutes**, and always stream.
- **Memory:** the engine uses about **50 GB of RAM** (working set measured 49.6 GB) and nearly all **12 GB of VRAM**. Keep the app light, and do not run other GPU programs alongside it.
- **Exactness:** answers are corrected toward the full model, at about 99 percent strict agreement in the release tests. A correction is visible as a short pause.

## 5. Known pitfalls

- **Launch context matters.**
  - When an automation tool started the engine from a background PowerShell context, every correction took about **36 s** instead of about **1.4 s** [measured]. Warm answers fell to 0.7-1.8 words/s.
  - Started as a normal process, it was fine.
  - **Check:** after the first corrected answer, the engine log line `scf parallel: ... correction latency mean X s` should show about 1-3 s. If it shows tens of seconds, the process was started with low priority.
- **Close GPU apps** (screenshot tools, NVIDIA overlay, games) before use. They slow every GPU call.
- **Only one engine.** The start script refuses to start while any `v41_` process is alive.
- **Do not change** `config\engine_env.template.txt` or the GPU keep-warm setting without the engine owner.
  - Keep-warm must stay **off** for the long-running server. Left on, it made the server about 30 percent slower (21 Oct).

## 6. Acceptance checks for the app session

1. `GET /v1/models` lists `deepseek-v4.1-flash-1017-promethevs`.
2. A streamed chat answer arrives in the app, and its first word comes within about 3 s on a warm engine.
3. A tool request, e.g. a `get_weather` tool and "What's the weather in London?", returns a `get_weather` call with `{"city": "London"}`.
4. Two chats in a row: the second one is faster than the first.
5. Quitting the app stops the engine, or keeps it running, as the user chose.

## 7. Not in this version

- **Remote use from another machine.** It would need the proxy to listen on the Tailscale address with a token; that is Ryan's call.
- **Several users at once.**
- **An installer that bundles the engine program.** Possible later, once the app side works; the model data always stays on disk where it is.
