# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cosmos_framework.utils import checkpoint_db


def _disable_progress_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoint_db, "_log_hf_progress", lambda *_args: None)


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_download_file_uses_python_api(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps({"path": "/cache/result"}))

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("COSMOS_HF_LOCAL__gpt2", raising=False)
    monkeypatch.setattr(checkpoint_db.subprocess, "run", run)
    _disable_progress_monitor(monkeypatch)

    path = checkpoint_db._hf_download(
        ["gpt2", "--repo-type", "model", "--revision", "main", "config.json"]
    )

    assert path == "/cache/result"
    cmd, kwargs = calls[0]
    assert "hf_hub_download" in cmd[-1]
    assert "config.json" in cmd[-1]
    assert kwargs["env"]["HF_HUB_VERBOSITY"] == "error"


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_download_directory_maps_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps({"path": "/cache/result"}))

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("COSMOS_HF_LOCAL__gpt2", raising=False)
    monkeypatch.setattr(checkpoint_db.subprocess, "run", run)
    _disable_progress_monitor(monkeypatch)

    checkpoint_db._hf_download(
        [
            "gpt2",
            "--repo-type",
            "model",
            "--revision",
            "main",
            "--include",
            "*.json",
            "--exclude",
            "large/*",
        ]
    )

    snippet = calls[0][0][-1]
    assert "snapshot_download" in snippet
    assert "allow_patterns" in snippet
    assert "ignore_patterns" in snippet


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_download_peer_reads_rank0_sentinel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return SimpleNamespace(stdout=json.dumps({"path": "/cache/result"}))

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "test-run")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.delenv("COSMOS_HF_LOCAL__gpt2", raising=False)
    monkeypatch.setattr(checkpoint_db, "_HF_DOWNLOAD_SENTINEL_DIR", tmp_path)
    monkeypatch.setattr(checkpoint_db.subprocess, "run", run)
    _disable_progress_monitor(monkeypatch)

    args = ["gpt2", "--repo-type", "model", "--revision", "main", "config.json"]
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert checkpoint_db._hf_download(args) == "/cache/result"
    assert len(calls) == 1

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert checkpoint_db._hf_download(args) == "/cache/result"
    assert len(calls) == 1
