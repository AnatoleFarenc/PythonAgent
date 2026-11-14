"""How the worker is isolated from the host.

Used by the host only: unlike SandboxConfig, it is never sent to the
worker, so the code never learns e.g. the host path of the workspace.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Optional
from pathlib import Path


class IsolationConfig(BaseModel):
    """Isolation settings of the worker (container, limits, workspace)."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["docker", "local"] = Field(
        default="docker",
        description="docker: container; local: plain process, NOT isolated"
    )
    image: str = Field(default="pythonrag-sandbox:dev",
                       description="Docker image of the worker")
    runtime: Literal["runsc", "runc"] = Field(
        default="runsc",
        description="runsc: gVisor; runc: plain Docker (shares the kernel)"
    )
    cpus: float = Field(default=1.0, gt=0, description="CPU cores")
    pids_limit: int = Field(
        default=256, ge=64,
        description="Host threads of the container (gVisor itself uses ~36)"
    )
    memory_margin_mb: int = Field(
        default=128, ge=0,
        description="Added to max_memory_mb for the container limit, so "
                    "that MemoryError is raised before the OOM kill"
    )
    scratch_size_mb: int = Field(default=64, gt=0,
                                 description="Size of the tmpfs scratch dir")
    workspace: Optional[Path] = Field(
        default=None,
        description="Host directory mounted in the container"
    )
    workspace_writable: bool = Field(
        default=False,
        description="Mount the workspace read-write instead of read-only"
    )
