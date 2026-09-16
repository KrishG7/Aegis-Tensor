"""Hugging Face Hub Remote Model Inspector & Resolver for Aegis-Tensor.

Enables inspecting remote .safetensors headers via zero-download HTTP Range Requests
and downloading remote model artifacts into a local security quarantine cache.
"""

from __future__ import annotations

import json
import os
import re
import struct
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

HF_HUB_BASE_URL = "https://huggingface.co"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aegis" / "hub"


class HuggingFaceInspectorError(ValueError):
    """Base exception for Hugging Face inspection failures."""
    pass


def parse_hf_uri(uri: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse a Hugging Face URI in the format `hf://username/repo[/filepath][@revision]`.
    
    Args:
        uri: The URI string (e.g. `hf://meta-llama/Llama-2-7b` or `hf://openai/clip-vit-base-patch32/model.safetensors@main`).
        
    Returns:
        Tuple of (repo_id, filename, revision). filename and revision may be None.
        
    Raises:
        HuggingFaceInspectorError: If URI does not begin with 'hf://' or lacks a valid repository identifier.
    """
    if not uri.startswith("hf://"):
        raise HuggingFaceInspectorError(f"Invalid URI '{uri}': must use the 'hf://' scheme (e.g., 'hf://owner/repo').")

    path_part = uri[len("hf://"):].strip("/")
    if not path_part:
        raise HuggingFaceInspectorError(f"Invalid Hugging Face URI '{uri}'. Expected 'hf://<owner>/<repo>'.")

    revision = None
    if "@" in path_part:
        path_part, revision = path_part.split("@", 1)

    parts = [p for p in path_part.split("/") if p]

    if not parts:
        raise HuggingFaceInspectorError(f"Invalid Hugging Face URI '{uri}'. Expected 'hf://<repo>' or 'hf://<owner>/<repo>'.")

    known_file_extensions = (".safetensors", ".bin", ".pt", ".pth", ".json", ".txt", ".onnx", ".msgpack")

    if len(parts) == 1:
        repo_id = parts[0]
        filename = None
    elif len(parts) == 2:
        if any(parts[1].endswith(ext) for ext in known_file_extensions):
            repo_id = parts[0]
            filename = parts[1]
        else:
            repo_id = f"{parts[0]}/{parts[1]}"
            filename = None
    else:
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])

    return repo_id, filename, revision


def get_hf_download_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """Generate the direct CDN download URL for a file in a Hugging Face repository."""
    return f"{HF_HUB_BASE_URL}/{repo_id}/resolve/{revision}/{filename}"


def inspect_hf_header(
    repo_id: str,
    filename: Optional[str] = "model.safetensors",
    revision: Optional[str] = "main",
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """Inspect the metadata and tensor structure of a remote .safetensors file without downloading weight payloads.
    
    Leverages HTTP Range Requests to:
    1. Read the first 8 bytes containing the little-endian uint64 header size (N).
    2. Read bytes 8 to 8 + N - 1 containing the JSON metadata string.
    
    Args:
        repo_id: Hugging Face repo ID (e.g. 'mistralai/Mistral-7B-v0.1').
        filename: Relative file path in the repo (defaults to 'model.safetensors').
        revision: Branch or commit SHA (defaults to 'main').
        timeout: HTTP request timeout in seconds.
        
    Returns:
        Dictionary containing parsed header metadata and tensor descriptors.
    """
    target_filename = filename or "model.safetensors"
    target_revision = revision or "main"
    url = get_hf_download_url(repo_id, target_filename, target_revision)
    user_agent = "Aegis-Tensor-HF-Inspector/0.1.0"

    # Step 1: Read the 8-byte uint64 header length
    req_len = urllib.request.Request(
        url,
        headers={
            "Range": "bytes=0-7",
            "User-Agent": user_agent,
        },
    )

    try:
        with urllib.request.urlopen(req_len, timeout=timeout) as resp:
            len_bytes = resp.read()
    except urllib.error.HTTPError as e:
        raise HuggingFaceInspectorError(f"Failed to access remote file '{url}': HTTP {e.code} {e.reason}") from e
    except Exception as e:
        raise HuggingFaceInspectorError(f"Network error connecting to '{url}': {e}") from e

    if len(len_bytes) < 8:
        raise HuggingFaceInspectorError(
            f"Invalid .safetensors file on remote: Expected 8-byte header size prefix, received {len(len_bytes)} bytes."
        )

    (header_size,) = struct.unpack("<Q", len_bytes)

    # Sanity limit: Safetensors headers should rarely exceed 100MB
    if header_size > 100 * 1024 * 1024:
        raise HuggingFaceInspectorError(
            f"Suspiciously large safetensors header size reported: {header_size} bytes."
        )

    # Step 2: Read bytes 8 to 8 + header_size - 1
    end_byte = 8 + header_size - 1
    req_header = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=8-{end_byte}",
            "User-Agent": user_agent,
        },
    )

    try:
        with urllib.request.urlopen(req_header, timeout=timeout) as resp:
            header_json_bytes = resp.read()
    except Exception as e:
        raise HuggingFaceInspectorError(f"Failed to read header content from '{url}': {e}") from e

    try:
        header_dict = json.loads(header_json_bytes.decode("utf-8"))
    except Exception as e:
        raise HuggingFaceInspectorError(f"Failed to parse safetensors JSON header: {e}") from e

    metadata = header_dict.get("__metadata__", {})
    tensor_descriptors = {k: v for k, v in header_dict.items() if k != "__metadata__"}

    return {
        "repo_id": repo_id,
        "filename": target_filename,
        "revision": target_revision,
        "header_size_bytes": header_size,
        "metadata": metadata,
        "num_tensors": len(tensor_descriptors),
        "tensor_count": len(tensor_descriptors),
        "tensors": tensor_descriptors,
        "header": tensor_descriptors,
    }


def download_hf_model(
    repo_id: str,
    filename: Optional[str] = "model.safetensors",
    revision: Optional[str] = "main",
    destination_dir: Optional[Path] = None,
    dest_dir: Optional[Path] = None,
    chunk_size: int = 1024 * 1024,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    timeout: float = 60.0,
) -> Path:
    """Download a remote .safetensors model from Hugging Face into a local quarantine cache.
    
    Args:
        repo_id: Hugging Face repo ID.
        filename: Relative path of file in repo.
        revision: Branch or commit SHA.
        destination_dir: Local destination directory.
        dest_dir: Alias for destination_dir.
        chunk_size: Streaming chunk size in bytes.
        progress_callback: Optional callback receiving (bytes_downloaded, total_bytes).
        timeout: Request timeout in seconds.
        
    Returns:
        Path to the downloaded local file.
    """
    target_filename = filename or "model.safetensors"
    target_revision = revision or "main"
    url = get_hf_download_url(repo_id, target_filename, target_revision)
    user_agent = "Aegis-Tensor-HF-Downloader/0.1.0"

    effective_dir = dest_dir or destination_dir or (DEFAULT_CACHE_DIR / Path(repo_id))
    effective_dir.mkdir(parents=True, exist_ok=True)
    target_file = effective_dir / Path(target_filename).name

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with open(target_file, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total_size)
    except urllib.error.HTTPError as e:
        raise HuggingFaceInspectorError(f"HTTP {e.code} downloading '{url}': {e.reason}") from e
    except Exception as e:
        raise HuggingFaceInspectorError(f"Download failed for '{url}': {e}") from e

    return target_file


def resolve_hf_model(
    uri: str,
    download: bool = False,
    cache_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, str, Optional[Path]]:
    """Parse and resolve a Hugging Face model URI.
    
    Args:
        uri: Hugging Face URI string ('hf://owner/repo[/file]').
        download: Whether to download the model file locally if not already cached.
        cache_dir: Custom cache directory.
        progress_callback: Download progress hook.
        
    Returns:
        Tuple of (repo_id, filename, local_path_or_none).
    """
    repo_id, filename, revision = parse_hf_uri(uri)
    target_filename = filename or "model.safetensors"
    target_revision = revision or "main"
    dest_dir = cache_dir or (DEFAULT_CACHE_DIR / Path(repo_id))
    cached_path = dest_dir / Path(target_filename).name

    if cached_path.exists():
        return repo_id, target_filename, cached_path

    if download:
        downloaded_path = download_hf_model(
            repo_id,
            target_filename,
            revision=target_revision,
            destination_dir=dest_dir,
            progress_callback=progress_callback,
        )
        return repo_id, target_filename, downloaded_path

    return repo_id, target_filename, None
