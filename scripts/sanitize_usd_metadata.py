"""Export a USD layer after removing local provenance metadata, preserving asset data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

LOCAL_PATH = re.compile(r"(?:/(?:home|Users|mnt|cfs|tmp)/|file:/+|[A-Za-z]:\\)")
PERSONAL_KEYS = frozenset({"author", "authorname", "createdby", "owner", "username"})
NOTICE_KEYS = frozenset({"copyright", "license", "licenses", "notice", "attribution"})
METADATA_FIELDS = frozenset(
    {"documentation", "comment", "customLayerData", "customData"}
)


def clean_dictionary(value: dict) -> tuple[dict, list[str]]:
    """Keep notices and numeric values; remove personal keys or local path strings."""
    result = {}
    removed = []
    for key, item in value.items():
        if key.lower() in NOTICE_KEYS:
            result[key] = item
        elif key.lower() in PERSONAL_KEYS:
            removed.append(key)
        elif isinstance(item, str) and LOCAL_PATH.search(item):
            removed.append(key)
        elif isinstance(item, dict):
            result[key], nested = clean_dictionary(item)
            removed.extend(f"{key}.{name}" for name in nested)
        else:
            result[key] = item
    return result, removed


def sanitize_usd_metadata(source: Path, output: Path) -> list[dict[str, str]]:
    """Copy a layer to a fresh output; never overwrite its source or edit references."""
    from pxr import Sdf

    source = source.resolve()
    output = output.resolve()
    if output == source or output.exists():
        raise FileExistsError("use a fresh output path distinct from the source")
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None:
        raise ValueError(f"cannot read USD layer: {source}")
    copy = Sdf.Layer.CreateAnonymous("sanitized.usda")
    copy.TransferContent(layer)
    paths = []
    copy.Traverse(Sdf.Path.absoluteRootPath, paths.append)
    removed = []
    for path in paths:
        spec = copy.GetObjectAtPath(path)
        if spec is None:
            continue
        for key in spec.ListInfoKeys():
            if key not in METADATA_FIELDS:
                continue
            value = spec.GetInfo(key)
            if isinstance(value, str) and LOCAL_PATH.search(value):
                spec.ClearInfo(key)
                removed.append({"spec": str(path), "field": key})
            elif isinstance(value, dict):
                clean, names = clean_dictionary(value)
                if names:
                    spec.SetInfo(key, clean)
                    removed.extend(
                        {"spec": str(path), "field": f"{key}.{name}"} for name in names
                    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not copy.Export(str(output)):
        raise RuntimeError("USD export failed")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    changes = sanitize_usd_metadata(args.source, args.output)
    print(f"Exported a fresh layer; removed {len(changes)} provenance fields.")


if __name__ == "__main__":
    main()
