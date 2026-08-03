# eIQ AAF Connector on FRDM-IMX95 + Ara-240

NXP's REST server that keeps an LLM resident on the Ara-240 and exposes it at
`POST /v1/chat/completions`. It is how an agent reaches the DNPU without importing torch or the Ara
SDK — the whole ML stack stays on the far side of HTTP.

Installed from NXP's **`.deb`**, then patched in place. The `.deb` comes from the account-gated
[Ara SDK downloads](https://www.nxp.com/design/design-center/development-boards-and-designs/ara-software-development-kit:ARA-SDK);
sources are at <https://github.com/nxp-imx-support/eiq-aaf-connector>. Licence is Proprietary, so
only the config, the patch and these scripts are committed here.

Verified on BSP `LF6.18.2-1.0.0`, rt-sdk-ara2 **2.0.4**, Ara-240 firmware 131072.

## Files here

| | |
|---|---|
| `server_config.json` | replaces the one the `.deb` installs; see the settings section |
| `connector-openai-fix.patch` | tool-call conformance fix, 2 files |
| `apply-patch.sh` | applies it to the installed package (board-side) |
| `run.sh` | start/restart the connector by hand (board-side) |

## Install

The `.deb` installs to `/usr/share/eiq/aaf-connector/`, creates a venv there, and adds an
`eiq-aaf-connector` systemd service.

```bash
dpkg -i eiq-aaf-connector_*.deb            # runs install.sh via postinst
```

**On BSP LF6.18.2-1.0.0 that install will not complete unaided.** Its `install.sh` does:

```bash
uv pip install /usr/share/python-wheels/optimum_ara-1.0.0-py3-none-any.whl
```

and this BSP's Ara runtime ships `optimum_ara-2.0.0.2` instead — a *different product line* whose
number is higher but older, and which lacks the Qwen-VL image classes the connector imports at
module load. Install the right wheel into the connector's venv yourself:

```bash
# optimum-ara 1.0.1, from the Optimum-Ara GitHub releases. Use 1.0.1 specifically:
# 1.0.0 predates the "use custom logits processors if provided" hot-fix.
uv pip install -p /usr/share/eiq/aaf-connector/venv /path/to/optimum_ara-1.0.1-py3-none-any.whl
```

1.0.1 runs correctly against this board's SDK 2.0.4 — both wheels load the board's own
`/usr/share/rt-sdk-ara240/include/dvapi.py` and need the same symbols from it.

Then patch, configure and run:

```bash
./apply-patch.sh                                        # tool-call conformance
cp server_config.json /usr/share/eiq/aaf-connector/     # the settings below
./run.sh
```

Models are fetched separately into `/usr/share/llm/<name>/`:

```bash
fetch_models --repo-id nxp/Qwen2.5-7B-Instruct-Ara240
```

## Why the patch is not optional

The connector emits tool calls with no `id` and no `type`, uses `choice.index` as a chunk counter,
and omits `finish_reason` on non-streaming responses. An OpenAI client cannot then tie a tool
*result* to the *call* that produced it, so it discards the call and the model reissues it forever —
strands-agents logs `incomplete tool use block, skipping`. The patch adds those fields.
`--revert` restores the originals; both keep a `.orig` beside each file.

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
| `ImportError` on a Qwen-VL class at startup | wrong `optimum-ara`: needs 1.0.1, not the BSP's 2.0.0.2 |

`ss` is not installed on this board — use `netstat -lnt` or `/proc/net/tcp`.
