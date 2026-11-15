"""Limits of the code MCP server.

Every tool output ends up in the context of a small local model, so each
tool caps what it returns and tells the model how to get the rest.
"""

from ..rag.config import EXCLUDED_DIRS

SERVER_NAME = "pythonrag-code"

# Folders hidden from list_files / grep (on top of the RAG exclusions):
# data/ holds the index cache, a huge JSON that would flood grep results.
IGNORED_DIRS = EXCLUDED_DIRS | {"data", "build", "dist", ".idea", ".vscode"}

# read_file
MAX_READ_LINES = 400
MAX_LINE_CHARS = 500

# list_files
MAX_LIST_ENTRIES = 200

# grep
MAX_GREP_MATCHES = 100
MAX_GREP_FILE_BYTES = 1_000_000

# search_code
DEFAULT_SEARCH_K = 5
MAX_SEARCH_K = 10
# How long search_code waits for the index before answering "still loading"
# (the client gives up after TIMEOUT_CALL_TOOL = 30 s).
INDEX_WAIT_TIMEOUT = 20.0

# edit_file: lines of diff shown after an edit
MAX_DIFF_LINES = 60

# Hard cap on any tool output, in characters.
MAX_OUTPUT_CHARS = 20_000
