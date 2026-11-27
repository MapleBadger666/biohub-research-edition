"""Verify the exact Research Edition file set and public release identity."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

MANIFEST = 'PUBLIC_RELEASE_MANIFEST.json'
VERSION = '1.0.0rc1'


def verify(root, expected_manifest_sha256=None):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("release root must be a regular directory")
    manifest = root/MANIFEST
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("release manifest is missing")
    raw = manifest.read_bytes()
    identity = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != identity:
        raise ValueError("release manifest digest differs")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result

    record = json.loads(raw, object_pairs_hook=unique)
    if record.get("schema") != 1 or record.get("version") != VERSION:
        raise ValueError("unsupported release manifest")
    actual = {}
    seen = set()
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if name == '.git' or name.startswith('.git/'):
            continue
        if name.casefold() in seen or path.is_symlink():
            raise ValueError("path collision or symlink")
        seen.add(name.casefold())
        if path.is_dir():
            continue
        if not path.is_file() or name == MANIFEST:
            if name == MANIFEST:
                continue
            raise ValueError("unsupported entry type")
        with path.open('rb') as file:
            digest = hashlib.file_digest(file, 'sha256').hexdigest()
        actual[name] = {"sha256":digest,"bytes":path.stat().st_size}
    expected = record["files"]
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or path.as_posix() != name or '\\' in name:
            raise ValueError("unsafe manifest path")
    if expected != actual:
        raise ValueError("release inventory missing, changed or unexpected file")
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    print(verify(Path(__file__).resolve().parent,args.expected_manifest_sha256))


if __name__ == '__main__':
    main()
