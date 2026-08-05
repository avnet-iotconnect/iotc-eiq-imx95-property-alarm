# eIQ AAF Connector on FRDM-IMX95 + Ara-240

NXP's REST server that keeps an LLM resident on the Ara-240 and exposes it at
`POST /v1/chat/completions`. It is how an agent reaches the DNPU without importing torch or the Ara
SDK — the whole ML stack stays on the far side of HTTP.

This is the demo's `ask` command, one process removed. koala reaches it over HTTP and therefore
needs no ML dependencies at all; everything heavy — torch, optimum-ara, the model itself — lives in
the venv this directory builds. Sources are at <https://github.com/nxp-imx-support/eiq-aaf-connector>.
The licence is proprietary, so only the config, the patch and these scripts are committed here.

Verified on BSP `LF6.18.2-1.0.0`, rt-sdk-ara2 **2.0.4**, Ara-240 firmware 131072.

## Files here

| | |
|---|---|
| `install.sh` | fetch the source, build the venv, install the wheel, apply the patch — one command |
| `run.sh` | start/restart it in the background; `--stop` |
| `apply-patch.sh` | re-applies the fix after a manual reinstall; a no-op if already applied |
| `connector-openai-fix.patch` | the tool-call conformance fix, 2 files |
| `config/server_config.json` | read from here; see the settings section |

`install.sh` adds `source/` and `venv/` beside them — delete those two to rebuild from scratch.

## Install

**This directory is the deployment.** It arrives with the rest of koala; run its installer on the
board, not on the host:

```bash
cd ~/koala/connector && ./install.sh     # ~1.2 GB, a few minutes, needs the network
./run.sh                                 # then ~225 s to load the 7B onto the Ara-240
tail -f server.log                       # ready when it says: Uvicorn running on http://0.0.0.0:3000
```

Prerequisite: the **rt-sdk-ara2 2.0.4 `.deb`** (top-level README.md), which supplies the Ara runtime
and the `optimum_ara-2.0.0.2` wheel. The deb drops that wheel in `/usr/share/python-wheels/` and
never installs it — something has to put it in a venv, and `install.sh` is that something.

Two things it gets right that are easy to get wrong by hand:

- **It installs connector `v2.0.0`, not HEAD.** v2.0.0 declares `optimum_ara-2.0.0.2`, the wheel
  this BSP already has, so nothing else is downloaded and it measures ~23 % faster (5.28 vs
  4.3 tok/s). The 2.1 line declares `optimum_ara-1.0.0`, which does not exist here — that is a
  different product line whose *higher* number is the *older* one.
- **It installs optimum-ara before the connector.** The connector declares it as a dependency but it
  is not on PyPI; `pyproject.toml` locates it through a `[tool.uv.sources]` path that pip ignores.
  Installing it first means the requirement is already satisfied.

The venv is deliberately built **without** `--system-site-packages`: its torch must not mix with the
BSP's `tflite_runtime`/`gi`/`cv2`, which is the whole reason this is a separate process.

Models are fetched separately into `/usr/share/llm/<name>/`:

```bash
fetch_models --repo-id nxp/Qwen2.5-7B-Instruct-Ara240
```

NXP also ships the connector as a `.deb` that installs into `/usr/share/eiq/aaf-connector/` with a
systemd unit. It works, but it needs the same wheel surgery and the same patch, and it puts the
install somewhere `scp -r` does not reach — hence `install.sh`.

## Why the patch is not optional

The connector emits tool calls with no `id` and no `type`, uses `choice.index` as a chunk counter,
and omits `finish_reason` on non-streaming responses. An OpenAI client cannot then tie a tool
*result* to the *call* that produced it, so it discards the call and the model reissues it forever —
strands-agents logs `incomplete tool use block, skipping`. The patch adds those fields. To undo it,
reinstall the connector into the venv. Send `work/NXP-eiq-aaf-connector-issues.md` upstream: if NXP
fixes this, the patch disappears.

## server_config.json — the settings that actually matter

- **`temperature` must not be 0.0.** NXP ships `temperature: 0.0, top_k: 0, top_p: 0.0`, which is
  their encoding of *greedy*; the Ara does the sampling and cannot resolve a token from it, so
  **every** completion returns HTTP 500 with a misleading transformers traceback. This file uses the
  models' own defaults, `0.7 / 20 / 0.8`.
- **`max_prompt_size`** is `3072` here, not NXP's `2047`. Every tool schema is part of the prompt, so
  a tool set of any size overruns the default and the request fails with *"Prompt cannot fit in
  context window"*. The model's compiled context is **4096 tokens total, prompt plus generation**, so
  3072 + `generate_max_tokens: 512` fits. That ceiling is fixed — the Ara's 16 GB holds weights, not
  context.
- **`enabled`** decides startup behaviour. `true` loads at startup and the port stays shut until the
  model is resident. All `false` binds quickly with no model, and you load one over REST — better
  while developing, and required to swap models without a restart.
- **`use_tool_guided_generation`** compiles tool schemas into a state machine (seconds on the first
  call with many tools, cached after). `POST /bench/guided_gen_perf` measures the cost.

The `*-3b-*` entries are NXP placeholders for models that were never published; only
`Qwen2.5-Coder-1.5B`, `Qwen2.5-7B-Instruct` and `Qwen2.5-VL-7B-Instruct` exist publicly.

## Loading a model over REST

```bash
curl -H 'Content-Type: application/json' -X POST http://localhost:3000/v1/models \
  -d '{"name":"Qwen2.5-7B-Instruct","description":"Qwen2.5 7B Instruct","type":"text",
       "tool_calling":"native","max_prompt_size":3072,"enabled":true}'
# -> {"detail":"Loading in progress."}   this body's max_prompt_size overrides the config file

curl http://localhost:3000/v1/models     # GET, not POST; poll for "ready": true
curl -X DELETE http://localhost:3000/v1/models/Qwen2.5-7B-Instruct
```

Two models can be resident at once despite what NXP's README says — but the VL model needs 12.3 GB
of the Ara's 16 GB, so it cannot share with a 7B. VL models want `type: "qwen_vl_video"`;
`qwen_vl_image` produces NaN logits and never terminates.

## When something breaks

| symptom | cause |
|---|---|
| `connection refused` on the port | still loading; the port binds only after startup finishes |
| HTTP 500 on every completion | `temperature: 0.0` — see above |
| streaming request dies mid-response | an exception after headers were sent. **Re-run with `"stream": false`** to see the real error |
| agent loops, reissuing one tool | the patch is not applied, or the connector was not restarted after it |
| `404 Model not found` | no model loaded, or the name does not match `GET /v1/models` |
| `ImportError` on a Qwen-VL class at startup | the wrong `optimum-ara` wheel for the connector release — see Install |
| koala's HUD says `ask: ready` but every question fails | the connector is not up; `./run.sh` and watch `server.log` |

`ss` is not installed on this board — use `netstat -lnt` or `/proc/net/tcp`. `pkill -f connector`
over ssh kills your own session, because the pattern matches the remote command line; `run.sh` uses
`pgrep -f "bin/conn[e]ctor"` to avoid exactly that.
