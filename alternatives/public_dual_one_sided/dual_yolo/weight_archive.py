"""Store checkpoint bytes in small, hash-verified files for a Git repository.

This module uses only the standard library. It never loads a checkpoint,
downloads anything, initializes CUDA, or publishes files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def basename(value):
    if (not isinstance(value, str) or value in ("", ".", "..")
            or "/" in value or "\\" in value or "\x00" in value):
        raise ValueError("Archive filenames must be plain local basenames")
    return value


def pack(source, directory, metadata=None, chunk_bytes=64 * 1024**2):
    source, directory = Path(source), Path(directory)
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 64 * 1024**2:
        raise ValueError("Chunk size must be between one byte and 64 MiB")
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("Expected a nonempty checkpoint file")
    if directory.exists():
        raise ValueError("Packaging requires a fresh output directory")
    filename = basename(source.name)
    directory.mkdir(parents=True)
    whole = hashlib.sha256()
    pieces, size = [], 0
    with source.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            name = f"{filename}.part{len(pieces):04d}"
            (directory / name).write_bytes(block)
            whole.update(block)
            size += len(block)
            pieces.append({"file": name, "bytes": len(block),
                           "sha256": hashlib.sha256(block).hexdigest()})
    if size != source.stat().st_size or whole.hexdigest() != digest(source):
        raise ValueError("Source checkpoint changed during packaging")
    manifest = {"schema": 1, "filename": filename, "bytes": size,
                "sha256": whole.hexdigest(), "pieces": pieces,
                "metadata": metadata or {}}
    (directory / "archive.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def restore(manifest_path, output_directory=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != 1:
        raise ValueError("Unsupported checkpoint archive schema")
    filename = basename(manifest["filename"])
    pieces = manifest["pieces"]
    if (type(manifest["bytes"]) is not int or manifest["bytes"] <= 0
            or not isinstance(pieces, list) or not pieces):
        raise ValueError("Invalid checkpoint archive size or piece list")
    names = [basename(piece["file"]) for piece in pieces]
    if len(set(names)) != len(names) or filename in names:
        raise ValueError("Repeated or conflicting checkpoint archive names")
    if any(type(piece["bytes"]) is not int or not 0 < piece["bytes"] <= 64 * 1024**2
           for piece in pieces) or sum(piece["bytes"] for piece in pieces) != manifest["bytes"]:
        raise ValueError("Piece sizes do not match checkpoint archive size")
    destination = Path(output_directory) if output_directory is not None else manifest_path.parent
    target = destination / filename
    if target.exists():
        if not target.is_file() or target.stat().st_size != manifest["bytes"] or digest(target) != manifest["sha256"]:
            raise ValueError("Refusing to replace an existing file with different bytes")
        return target
    destination.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".checkpoint-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            whole = hashlib.sha256()
            for piece in pieces:
                path = manifest_path.parent / piece["file"]
                if path.is_symlink() or not path.is_file() or path.stat().st_size != piece["bytes"]:
                    raise ValueError(f"Missing or invalid checkpoint piece: {piece['file']}")
                part = hashlib.sha256()
                with path.open("rb") as source:
                    while block := source.read(1024**2):
                        stream.write(block)
                        whole.update(block)
                        part.update(block)
                if part.hexdigest() != piece["sha256"]:
                    raise ValueError(f"Checkpoint piece hash differs: {piece['file']}")
            if stream.tell() != manifest["bytes"] or whole.hexdigest() != manifest["sha256"]:
                raise ValueError("Reconstructed checkpoint hash or size differs")
            stream.flush()
            os.fsync(stream.fileno())
        # Publish without replacing a file another notebook may have created.
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("pack")
    package.add_argument("source", type=Path)
    package.add_argument("directory", type=Path)
    package.add_argument("--metadata", type=Path)
    unpack = commands.add_parser("restore")
    unpack.add_argument("manifest", type=Path)
    unpack.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        metadata = json.loads(args.metadata.read_text()) if args.metadata else None
        result = pack(args.source, args.directory, metadata)
        print(json.dumps({"filename": result["filename"], "bytes": result["bytes"],
                          "sha256": result["sha256"], "pieces": len(result["pieces"])}))
    else:
        path = restore(args.manifest, args.output_directory)
        print(json.dumps({"restored": str(path), "sha256": digest(path)}))


if __name__ == "__main__":
    main()
