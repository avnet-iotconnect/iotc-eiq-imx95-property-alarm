"""Drive the Ara-240 from strands-agents, through the eIQ AAF Connector's OpenAI endpoint.

The connector is a REST server on the board that keeps a model resident on the Ara-240.
strands-agents talks to it as if it were OpenAI, so nothing here imports optimum-ara,
torch or any Ara SDK piece -- the version conflict stays on the far side of HTTP.

Run this on the host (or the board) with the connector already serving:
    .venv/bin/python jaguar/strands-ara.py
"""

from strands import Agent, tool
from strands.models.openai import OpenAIModel

# The board, and the connector's port. The connector must have been started with
# --host 0.0.0.0 for this to be reachable from anywhere but the board itself.
BASE_URL = "http://192.168.38.203:3000/v1"


@tool
def get_alarm_state() -> str:
    """Return the current alarm state and whether a registered user is present."""
    return '{"state": "armed", "registered_user_present": false}'


@tool
def sound_alarm(reason: str) -> str:
    """Sound the theft alarm.

    Args:
        reason: why the alarm is sounding.
    """
    print(f"    [tool] ALARM SOUNDING: {reason}")
    return '{"status": "success"}'


model = OpenAIModel(
    client_args={"api_key": "unused", "base_url": BASE_URL},
    model_id="Qwen2.5-7B-Instruct",
    # temperature 0.0 makes the Ara's on-device sampler emit an invalid token id,
    # which surfaces as an OverflowError inside the connector. Keep it non-zero.
    params={"temperature": 0.7, "max_tokens": 256},
)

agent = Agent(model=model, tools=[get_alarm_state, sound_alarm])

result = agent("The laptop is missing from the stand. Check the alarm state, then sound the alarm.")
print("\n--- final ---")
print(result)
