API_KEY_PATTERN_NAME = "API_KEY"

PATH_SYS_PROMPT = "./src/orchestrator/data/sys_prompt.txt"

STOP_PROMPT = [
    "Observation:",
    "Observation\n",
    "```Thought",
    "```\nThought",
]

MAX_SERVER_RETRIES = 3
MCP_CMD = "python -m src.mcp_server --root . --writable"

LIMITS_ITERATION = 40
MAX_INPUT_TOKEN = 50000
MAX_OUTPUT_TOKEN = 3000
TIMEOUT_SEC = 600

# None if you want a default conf
PATH_SANDBOX_CONFIG = None
