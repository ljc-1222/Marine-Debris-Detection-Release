"""Build and verify a commit-pinned source archive including all upstream sources."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tarfile
import tempfile

if __package__:
    from .environment import ROOT, SOURCE_REVISIONS
else:
    from environment import ROOT, SOURCE_REVISIONS


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def safe_name(value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or any(c in value for c in "\r\n\0"):
        raise ValueError("Unsafe source member path")
    return path


def member_identity(member, stream):
    safe_name(member.name)
    if member.isfile():
        value = stream.extractfile(member).read()
        return {"kind": "file", "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value), "mode": member.mode & 0o777}
    if member.issym():
        target = posixpath.normpath(posixpath.join(posixpath.dirname(member.name), member.linkname))
        if member.linkname.startswith("/") or not target.startswith("marine-debris/"):
            raise ValueError("Source symlink escapes the archive")
        return {"kind": "symlink", "target": member.linkname}
    raise ValueError("Unsupported source archive member type")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def build(directory, revision="HEAD", root=ROOT):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "source-manifest.json").exists():
        raise FileExistsError("Use a new release output directory")
    commit = git(root, "rev-parse", "--verify", revision + "^{commit}")
    sources = [(Path(root), commit, "marine-debris/")]
    upstreams = {}
    for name, pinned in SOURCE_REVISIONS.items():
        row = git(root, "ls-tree", commit, "--", name).split()
        if len(row) < 3 or row[:3] != ["160000", "commit", pinned]:
            raise ValueError("Release commit does not pin the expected upstream: " + name)
        upstream = Path(root) / name
        sources.append((upstream, pinned, "marine-debris/" + name + "/"))
        upstreams[name] = {"revision": pinned, "tree": git(upstream, "rev-parse", pinned + "^{tree}")}
    archive = directory / "marine-debris-source.tar.gz"
    inventory = {}
    with archive.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target:
            for repository, pinned, prefix in sources:
                with tempfile.TemporaryFile(dir=directory) as temporary:
                    subprocess.run(["git", "archive", "--format=tar", "--prefix=" + prefix, pinned], cwd=repository, stdout=temporary, check=True)
                    temporary.seek(0)
                    with tarfile.open(fileobj=temporary) as original:
                        for member in original:
                            if member.isdir():
                                continue
                            if member.name in inventory:
                                raise ValueError("Duplicate source member")
                            inventory[member.name] = member_identity(member, original)
                            member.uid = member.gid = member.mtime = 0
                            member.uname = member.gname = ""
                            member.pax_headers = {}
                            data = original.extractfile(member) if member.isfile() else None
                            target.addfile(member, data)
    manifest = {"schema_version": 1, "status": "local_rc_not_published", "project_commit": commit,
                "submodules": upstreams, "archive": {"filename": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size},
                "files": inventory, "github_url": None, "hf_repo_id": None, "hf_revision": None}
    (directory / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify(directory)
    return manifest


def verify(directory, source_root=None):
    directory = Path(directory).resolve()
    document = json.loads((directory / "source-manifest.json").read_text())
    if document.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", document.get("project_commit", "")):
        raise ValueError("Invalid source manifest")
    info = document["archive"]
    name = safe_name(info["filename"])
    if len(name.parts) != 1:
        raise ValueError("Archive must be directly inside its release directory")
    archive = directory / name
    if archive.stat().st_size != info["bytes"] or digest(archive) != info["sha256"]:
        raise ValueError("Source archive SHA256/byte mismatch")
    actual = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            if member.isdir():
                continue
            path = safe_name(member.name)
            if path.parts[0] != "marine-debris" or ".git" in path.parts or member.name in actual:
                raise ValueError("Unexpected/duplicate source member")
            actual[member.name] = member_identity(member, stream)
    if actual != document["files"]:
        raise ValueError("Source inventory differs from the archive")
    if source_root is not None:
        source_root = Path(source_root).resolve()
        for name, info in actual.items():
            relative = PurePosixPath(name).relative_to("marine-debris")
            path = source_root / relative
            if info["kind"] == "symlink":
                if not path.is_symlink() or path.readlink().as_posix() != info["target"]:
                    raise ValueError("Extracted source symlink mismatch: " + str(relative))
            elif path.is_symlink() or not path.is_file() or digest(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
                raise ValueError("Extracted source file mismatch: " + str(relative))
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    create = subcommands.add_parser("build")
    create.add_argument("--revision", default="HEAD")
    create.add_argument("--output-dir", type=Path, required=True)
    check = subcommands.add_parser("verify")
    check.add_argument("--directory", type=Path, required=True)
    check.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    document = build(args.output_dir, args.revision) if args.command == "build" else verify(args.directory, args.source_root)
    print(json.dumps({"project_commit": document["project_commit"], "archive": document["archive"], "files": len(document["files"])}))


if __name__ == "__main__":
    main()
