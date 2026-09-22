"""Unit and integration tests for Hugging Face Hub Remote Inspector."""

import json
import struct
import io
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aegis.hf_inspector import (
    parse_hf_uri,
    inspect_hf_header,
    download_hf_model,
    resolve_hf_model,
    HuggingFaceInspectorError,
)
from aegis.cli import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# URI Parsing Tests
# ---------------------------------------------------------------------------

def test_parse_hf_uri_standard():
    repo_id, filename, revision = parse_hf_uri("hf://google/gemma-2b")
    assert repo_id == "google/gemma-2b"
    assert filename is None
    assert revision is None


def test_parse_hf_uri_with_filename():
    repo_id, filename, revision = parse_hf_uri("hf://facebook/opt-125m/model.safetensors")
    assert repo_id == "facebook/opt-125m"
    assert filename == "model.safetensors"
    assert revision is None


def test_parse_hf_uri_with_revision():
    repo_id, filename, revision = parse_hf_uri("hf://meta-llama/Llama-2-7b-hf@v1.0.0")
    assert repo_id == "meta-llama/Llama-2-7b-hf"
    assert filename is None
    assert revision == "v1.0.0"


def test_parse_hf_uri_with_filename_and_revision():
    repo_id, filename, revision = parse_hf_uri("hf://mistralai/Mistral-7B-v0.1/model.safetensors@main")
    assert repo_id == "mistralai/Mistral-7B-v0.1"
    assert filename == "model.safetensors"
    assert revision == "main"


def test_parse_hf_uri_invalid_scheme():
    with pytest.raises(HuggingFaceInspectorError, match="must use the 'hf://' scheme"):
        parse_hf_uri("https://huggingface.co/google/gemma-2b")


def test_parse_hf_uri_empty_or_malformed():
    with pytest.raises(HuggingFaceInspectorError):
        parse_hf_uri("hf://")
    with pytest.raises(HuggingFaceInspectorError):
        parse_hf_uri("hf:///")


def test_parse_hf_uri_single_part_repo():
    repo_id, filename, revision = parse_hf_uri("hf://bert-base-uncased")
    assert repo_id == "bert-base-uncased"
    assert filename is None
    assert revision is None


# ---------------------------------------------------------------------------
# HTTP Range Request Header Inspection Tests
# ---------------------------------------------------------------------------

def _make_mock_safetensors_bytes(header_dict: dict) -> bytes:
    json_bytes = json.dumps(header_dict).encode("utf-8")
    prefix = struct.pack("<Q", len(json_bytes))
    return prefix + json_bytes


def test_inspect_hf_header_success():
    dummy_header = {
        "__metadata__": {"format": "pt"},
        "model.layers.0.weight": {
            "dtype": "F32",
            "shape": [64, 64],
            "data_offsets": [0, 16384],
        },
    }
    raw_payload = _make_mock_safetensors_bytes(dummy_header)
    len_bytes = raw_payload[:8]
    header_bytes = raw_payload[8:]

    def fake_urlopen(req, timeout=10.0):
        # Inspect range header
        range_header = req.headers.get("Range", "")
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Length": str(len(raw_payload) + 100000)}
        if range_header == "bytes=0-7":
            mock_resp.read.return_value = len_bytes
        else:
            mock_resp.read.return_value = header_bytes
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = inspect_hf_header("google/gemma-2b", "model.safetensors")

    assert result["repo_id"] == "google/gemma-2b"
    assert result["filename"] == "model.safetensors"
    assert result["header_size_bytes"] == len(header_bytes)
    assert "model.layers.0.weight" in result["header"]
    assert result["tensor_count"] == 1


def test_inspect_hf_header_http_error():
    err = urllib.error.HTTPError(
        url="https://huggingface.co/mock",
        code=404,
        msg="Not Found",
        hdrs=None,
        fp=io.BytesIO(b"Not Found"),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(HuggingFaceInspectorError, match="HTTP 404"):
            inspect_hf_header("google/nonexistent-model", "model.safetensors")


def test_download_hf_model_success(tmp_path):
    target_dir = tmp_path / "downloads"
    dummy_content = b"TEST_SAFETENSORS_MODEL_DATA"

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Length": str(len(dummy_content))}
    mock_resp.read.side_effect = [dummy_content, b""]
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        local_path = download_hf_model(
            repo_id="test/repo",
            filename="model.safetensors",
            dest_dir=target_dir,
        )

    assert local_path.exists()
    assert local_path.read_bytes() == dummy_content


# ---------------------------------------------------------------------------
# CLI Integration Tests for hf:// URI
# ---------------------------------------------------------------------------

def test_cli_scan_hf_remote_inspection():
    mock_inspection = {
        "repo_id": "bert-base-uncased",
        "filename": "model.safetensors",
        "revision": "main",
        "file_size_bytes": 440000000,
        "header_size_bytes": 1024,
        "tensor_count": 2,
        "header": {
            "embeddings.word_embeddings.weight": {
                "dtype": "F32",
                "shape": [30522, 768],
                "data_offsets": [0, 93763584],
            },
            "encoder.layer.0.attention.self.query.weight": {
                "dtype": "F32",
                "shape": [768, 768],
                "data_offsets": [93763584, 96122880],
            },
        },
    }

    with patch("aegis.cli.inspect_hf_header", return_value=mock_inspection):
        result = runner.invoke(app, ["scan", "hf://bert-base-uncased"])
        assert result.exit_code == 0
        assert "Hugging Face Remote Inspection" in result.output
        assert "bert-base-uncased" in result.output
        assert "embeddings.word_embeddings" in result.output
