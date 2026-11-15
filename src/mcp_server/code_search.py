"""Semantic code search on top of the RAG ``HybridIndex``.

Loading bge-m3 and updating the index takes from seconds to minutes, far
more than the MCP handshake allows (10 s). The index is therefore built in
a background thread started with the server; ``search_code`` waits for it
a bounded time and otherwise tells the model to retry.

Only retrieval is done here: the chunks are returned as-is and the agent's
own LLM reads them. ``RagGenerator`` (the second LLM) is not used.

When the agent edits a file, notify_changed() marks it "dirty" and
re-runs the incremental indexer in the background (only the changed files
are re-chunked and re-encoded). Until then, results from a dirty file are
not shown with stale byte offsets: the model is told to use read_file.
"""

import contextlib
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from model import ChunkModel
from ..rag.chunking.chunking_tools_python import read_chunk
from .config import INDEX_WAIT_TIMEOUT, MAX_SEARCH_K
from .workspace import Workspace, truncate

if TYPE_CHECKING:
    from ..rag.indexer.HybridIndex import HybridIndex


class IndexNotReady(RuntimeError):
    """The index is still being built."""


class CodeSearch:
    """Background-built hybrid index of the ``.py`` files of a workspace.

    Attributes:
        workspace: Workspace whose files are indexed.
        folder: Folder given to the indexer (as typed by the user, so that
            chunk paths match the on-disk cache).
        model_name: Embedding model name.
        use_embedding: False for a BM25-only index.
    """

    def __init__(
            self, workspace: Workspace, folder: str, model_name: str,
            use_embedding: bool) -> None:
        """Prepare the loader; nothing is loaded before start()."""
        self.workspace = workspace
        self.folder = folder
        self.model_name = model_name
        self.use_embedding = use_embedding
        self._index: "HybridIndex | None" = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        # Files edited since the index was built: resolved path -> edit
        # counter, so that an edit made during a refresh stays dirty.
        self._dirty: dict[str, int] = {}
        self._lock = threading.Lock()
        self._refreshing = False

    def start(self) -> None:
        """Start building the index in a daemon thread (idempotent)."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._build, name="index-loader", daemon=True)
            self._thread.start()

    def _build(self) -> None:
        """Thread body: import the heavy modules and build the index."""
        # The indexer prints progress: send it to stderr, stdout is the
        # MCP protocol channel in stdio mode.
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from ..rag.indexer.HybridIndex import HybridIndex
                self._index = HybridIndex.from_folder(
                    self.folder, self.model_name, self.use_embedding)
        except BaseException as e:
            self._error = e
        finally:
            self._done.set()

    def get_index(self, timeout: float = INDEX_WAIT_TIMEOUT) -> "HybridIndex":
        """Wait up to *timeout* seconds for the index.

        Raises:
            IndexNotReady: If it is still being built.
            RuntimeError: If building it failed.
        """
        self.start()
        if not self._done.wait(timeout):
            raise IndexNotReady(
                "the code index is still being built, retry search_code in "
                "a few seconds (grep and read_file already work)")
        if self._error is not None:
            raise RuntimeError(
                f"the code index could not be built: {self._error}")
        assert self._index is not None
        return self._index

    def notify_changed(self, path: str | Path) -> None:
        """Mark *path* as modified and refresh the index in the background.

        Only .py files are indexed; other files are ignored.
        """
        if Path(path).suffix != ".py":
            return
        with self._lock:
            key = str(Path(path).resolve())
            self._dirty[key] = self._dirty.get(key, 0) + 1
            if self._refreshing:
                return  # the running refresh will pick it up
            self._refreshing = True
        threading.Thread(target=self._refresh_loop, name="index-refresh",
                         daemon=True).start()

    def _refresh_loop(self) -> None:
        """Re-run the incremental indexer until no file is dirty."""
        self.start()
        self._done.wait()
        while True:
            with self._lock:
                pending = dict(self._dirty)
                if not pending or self._index is None:
                    self._refreshing = False
                    return
            old = self._index
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    from ..rag.indexer.HybridIndex import HybridIndex
                    from ..rag.indexer.indexer import update_index
                    chunks, vectors = update_index(
                        self.folder, old.model, self.model_name)
                    new = HybridIndex(chunks, vectors, old.model, self.folder)
            except Exception as e:
                print(f"[index] refresh failed: {e}", file=sys.stderr)
                with self._lock:
                    self._refreshing = False
                return
            with self._lock:
                self._index = new
                for key, version in pending.items():
                    if self._dirty.get(key) == version:
                        del self._dirty[key]

    def is_dirty(self, chunk: ChunkModel) -> bool:
        """True if the chunk's file changed since the index was built."""
        with self._lock:
            return str(Path(chunk.file).resolve()) in self._dirty

    def format_chunk(self, rank: int, chunk: ChunkModel) -> str:
        """Render one result with its location, as the model will cite it."""
        where = (f"{self.workspace.relative(chunk.file)}:"
                 f"{chunk.start_line}-{chunk.end_line}")
        scope = f" ({chunk.paths})" if chunk.paths else ""
        return f"--- [{rank}] {where}{scope} ---\n{read_chunk(chunk)}"

    def search(self, query: str, k: int) -> str:
        """Return the *k* chunks most relevant to *query*, formatted.

        Raises:
            ValueError: If the query is empty.
            IndexNotReady / RuntimeError: See get_index().
        """
        if not query.strip():
            raise ValueError("query is empty")
        k = max(1, min(k, MAX_SEARCH_K))
        chunks = self.get_index().search(query, k=k)
        if not chunks:
            return f"[no code found for '{query}']"
        parts = []
        for rank, chunk in enumerate(chunks, start=1):
            try:
                if self.is_dirty(chunk):
                    raise OSError("stale")
                parts.append(self.format_chunk(rank, chunk))
            except (OSError, UnicodeDecodeError):
                # File deleted or edited since indexing (byte offsets stale).
                parts.append(
                    f"--- [{rank}] {self.workspace.relative(chunk.file)} "
                    "(changed since indexing, use read_file) ---")
        return truncate("\n\n".join(parts))
