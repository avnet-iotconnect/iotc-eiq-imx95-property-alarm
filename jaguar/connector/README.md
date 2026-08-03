# Running NXP's eIQ AAF Connector on the board

The connector is a REST server that keeps an LLM resident on the Ara-240 and exposes it at
`POST /v1/chat/completions`. It is how `jaguar/tools-strands.py` reaches the DNPU without importing
torch or the Ara SDK — the whole ML stack stays on the far side of HTTP.

Source: <https://github.com/nxp-imx-support/eiq-aaf-connector> (cloned to `work-refs/eiq-aaf-connector`).
Licence is **Proprietary**, so only this directory's config and scripts are committed here — not
NXP's source.

Findings and measurements: `work/STATUS-jaguar.md`. Defects worth sending NXP:
`work/NXP-eiq-aaf-connector-issues.md`.

## Two venvs on the board, deliberately named differently

| | venv | contents |
|---|---|---|
| `~/jaguar/` | `venv` (with `--system-site-packages`) | strands-agents + openai. Pure Python; sees BSP `gi`/`cv2`. |
| `~/jaguar/connector/` | `connector-venv` (**no** system packages) | torch 2.9, transformers, optimum-ara. |

They must not be merged. optimum-ara hard-pins its own torch, which collides with the BSP
`tflite_runtime`/`gi`/`cv2` stack every other pilot needs. Keeping them apart is the entire reason
this is a REST service and not a library.

## Install

On the host, send the connector source over (the board has no `rsync`):

```bash
cd work-refs
tar czf - --exclude=.git eiq-aaf-connector \
  | ssh root@192.168.38.203 'mkdir -p /root/jaguar/connector && tar xzf - -C /root/jaguar/connector --strip-components=1'
scp jaguar/connector/{server_config.json,run.sh,requirements.txt} root@192.168.38.203:/root/jaguar/connector/
```

On the board:

```bash
cd /root/jaguar/connector

# 1. optimum-ara. THE ONE THING THAT IS EASY TO GET WRONG.
#    Use 1.0.1 specifically, from the Optimum-Ara GitHub releases. The connector's
#    own install.sh and pyproject.toml ask for 1.0.0, which predates the "use custom
#    logits processors if provided" hot-fix that 1.0.1 exists to ship -- it only
#    matters for guided generation on a host-MCP model, but there is no reason to
#    install the older one.
#    Do NOT use the board's /usr/share/python-wheels/optimum_ara-2.0.0.2 here: it
#    lacks the Qwen-VL image classes the connector imports at module load. (The SDK's
#    own examples/ need 2.0.0.2 instead -- the two are different product lines and the
#    bigger number is the older one.) 1.0.1 works fine against this board's SDK 2.0.4:
#    both wheels load the board's own /usr/share/rt-sdk-ara240/include/dvapi.py.
uv venv connector-venv --python 3.13
uv pip install -p connector-venv /root/test/optimum_ara-1.0.1-py3-none-any.whl
uv pip install -p connector-venv -r requirements.txt
uv pip install -p connector-venv .           # the connector package itself

# 2. Apply the OpenAI-conformance patch, or strands drops every tool call and loops.
#    Run this from the host: ./jaguar/apply-connector-patch.sh

# 3. Config, then run.
cp server_config.json config/server_config.json
chmod +x run.sh && ./run.sh
```

`outlines-core` has a prebuilt aarch64 wheel, so no Rust toolchain is needed. The venv lands at
about 1.4 GB and `uv sync` takes roughly six minutes.

## server_config.json — the settings that actually matter

- **`temperature` must not be 0.0.** NXP ships `temperature: 0.0, top_k: 0, top_p: 0.0`, which is
  their encoding of *greedy*; the Ara does the sampling and cannot resolve a token from it, so
  **every** completion returns HTTP 500. The committed file uses the models' own defaults,
  `0.7 / 20 / 0.8`.
- **`max_prompt_size`** is `3072` here, not NXP's `2047`. Every tool schema is part of the prompt, so
  MCP tool sets overrun the default and the request fails with *"Prompt cannot fit in context
  window"* — which, while streaming, appears only as a dropped connection. The model's compiled
  context is **4096 tokens total, prompt plus generation**, so 3072 + `generate_max_tokens: 512`
  fits. This is a hard ceiling; the Ara's 16 GB holds weights, not context.
- **`enabled`** decides startup behaviour. `true` loads at startup and the port stays shut until the
  model is resident. All `false` binds in ~80 s with no model, and you load one over REST — better
  while developing, and required to swap models without a restart.
- **`use_tool_guided_generation`** compiles tool schemas into a state machine (a few seconds on the
  first call with many tools, cached after). `POST /bench/guided_gen_perf` measures the cost.

Models live in `/usr/share/llm/<name>/`, fetched with `fetch_models --repo-id nxp/<repo>`. Only
`Qwen2.5-Coder-1.5B`, `Qwen2.5-7B-Instruct` and `Qwen2.5-VL-7B-Instruct` are installed; the
`*-3b-*` entries in the config are NXP placeholders for models that were never published.

## Loading a model over REST

```bash
curl -H 'Content-Type: application/json' -X POST http://localhost:3000/v1/models \
  -d '{"name":"Qwen2.5-7B-Instruct","description":"Qwen2.5 7B Instruct","type":"text",
       "tool_calling":"native","max_prompt_size":3072,"enabled":true}'
# -> {"detail":"Loading in progress."}   the POST body's max_prompt_size wins over the config file

curl http://localhost:3000/v1/models     # GET, not POST; poll for "ready": true
curl -X DELETE http://localhost:3000/v1/models/Qwen2.5-7B-Instruct
```

Two models can be resident at once despite what NXP's README says — but the VL model needs 12.3 GB
of the Ara's 16 GB, so it cannot share with the 7B.

## When something breaks

| symptom | cause |
|---|---|
| `connection refused` on the port | still loading; the port binds only after startup finishes |
| HTTP 500 on every completion | `temperature: 0.0` — see above |
| streaming request dies mid-response | an exception after headers were sent. **Re-run with `"stream": false`** to see the real error |
| strands loops, reissuing one tool | the conformance patch is not applied |
| `404 Model not found` | no model loaded, or the name does not match `GET /v1/models` |

`ss` is not installed on this board — use `netstat -lnt` or `/proc/net/tcp`.
