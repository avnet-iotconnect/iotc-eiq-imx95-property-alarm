"""Tool calling on the Ara-240, via strands-agents and the eIQ AAF Connector.

The model runs on the DNPU. This script only speaks HTTP, so it needs no torch,
no optimum-ara and no Ara SDK -- that whole stack lives on the far side of the
connector's OpenAI endpoint.

    .venv/bin/python jaguar/tools-strands.py
    .venv/bin/python jaguar/tools-strands.py "If time is past 8/2/26 midnight, trigger the alert"

Load the model first. With the connector already running, hot-load it over REST:

    curl -H 'Content-Type: application/json' -X POST http://192.168.38.203:3000/v1/models \
      -d '{"name":"Qwen2.5-7B-Instruct","description":"Qwen2.5 7B Instruct","type":"text",
           "tool_calling":"native","max_prompt_size":3072,"enabled":true}'
    # ~225 s for the 7B; poll until ready:
    curl http://192.168.38.203:3000/v1/models

max_prompt_size matters once MCP tools are in play. Every tool schema is part of
the prompt, so 21 IoTConnect tools alone run past NXP's 2047 default and the
request dies with "Prompt cannot fit in context window" -- as a clean HTTP 422
when not streaming, and as a silently dropped connection when streaming, because
the error happens after the response headers have gone out. The model's compiled
context is 4096 tokens total, prompt plus generation, so 3072 + 512 fits.

To start the connector from scratch instead, on the board:

    cd /root/jaguar/connector && ./run.sh     # restarts it if already running

Two things about that config are worth knowing, both explained in
work/NXP-eiq-aaf-connector-issues.md: temperature must not be 0.0 (the shipped
default makes every request fail), and the connector needs our tool-call patch
(jaguar/connector/apply-patch.sh) or strands drops every call and loops.
"""

import sys
from datetime import datetime

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.tools.mcp import MCPClient
from mcp.client.streamable_http import streamable_http_client


@tool
def get_time() -> str:
    """Return the current date and time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"    [tool] get_time -> {now}")
    return now


@tool
def alert(message: str) -> str:
    """Raise an alert.

    Args:
        message: the text to show.
    """
    print(f"    [tool] ALERT: {message}")
    return "alert raised"


prompt = sys.argv[1] if len(sys.argv) > 1 else "What day is today?"

model = OpenAIModel(
    client_args={"api_key": "unused", "base_url": "http://192.168.38.203:3000/v1"},
    model_id="Qwen2.5-7B-Instruct",
    params={"temperature": 0.7, "max_tokens": 256},
)


# MCPClient wants a callable that opens a transport. strands' own examples still
# use the older streamablehttp_client, which mcp 1.29 marks deprecated in favour
# of streamable_http_client -- same three-tuple, so it drops straight in. Headers
# and timeouts now go on an httpx.AsyncClient passed as http_client=.
mcp_client = MCPClient(
    lambda: streamable_http_client("http://localhost:8000/mcp")
)

# The MCP tools are only usable while the client session is open, so the agent
# runs inside the with-block too. list_tools_sync returns a PaginatedList, which
# subclasses list -- extending it with our own @tool functions is fine.
with mcp_client:
    # List the MCP tools but do not hand them to the agent yet. Every tool schema
    # is part of every prompt, and all 21 of IoTConnect's do not fit beside the
    # conversation in the model's 4096-token context -- the request then dies with
    # "Prompt cannot fit in context window", which while streaming shows up only as
    # a dropped connection. Next step is picking a handful of these by name.
    mcp_tools = mcp_client.list_tools_sync()
    print(f"MCP tools ({len(mcp_tools)}):")
    for t in mcp_tools:
        print(f"  {t.tool_name}")

    tools = [get_time, alert]

    # model= is not optional: without it strands quietly falls back to Bedrock.
    agent = Agent(model=model, tools=tools)
    print(f"Agent loaded with {len(tools)} tools")

    start = datetime.now()
    agent(prompt)
    print()  # prompt doesn't do newline
    print(int((datetime.now() - start).total_seconds() * 1000))
