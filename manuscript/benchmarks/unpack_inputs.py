"""Verify and unpack the fixed benchmark inputs beside this script."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack():
    manifest = json.loads((ROOT / "inputs.json").read_text())
    for name, archive_info in manifest["archives"].items():
        archive = ROOT / name
        assert sha(archive) == archive_info["sha256"], "Input archive changed"
        with tarfile.open(archive, "r:gz") as handle:
            assert {member.name for member in handle} == set(archive_info["files"])
            for member in handle.getmembers():
                destination = ROOT / member.name
                assert member.isfile() and destination.resolve().is_relative_to(ROOT)
                expected = manifest["files"][member.name]
                if destination.exists():
                    assert sha(destination) == expected["sha256"], destination
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as source, destination.open("wb") as output:
                    while block := source.read(8 * 1024 * 1024):
                        output.write(block)
                assert destination.stat().st_size == expected["bytes"]
                assert sha(destination) == expected["sha256"], destination
    print(f"Verified {len(manifest['files'])} fixed input files.")


if __name__ == "__main__":
    unpack()
