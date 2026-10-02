#!/usr/bin/env python3
"""Consolidate fragmented trajectory data into a compact, portable layout.

Old layout (fragmented, ~100K files):
    trajectories/case_XXX/sample.json   (absolute image paths)
    trajectories/case_XXX/meta.json
    trajectories/case_XXX/images/0001.jpg ...

New layout (self-contained, all relative paths):
    dataset.jsonl          # all samples, images use relative paths
    metadata.jsonl         # all metadata, one line per case
    images/                # flat: images/{case_id}/{seq}.jpg
        case_XXX/0001.jpg

Benefits:
  - Self-contained: the entire output dir can be cp/rsync/tar'd as a unit
  - ALL paths are relative to the output dir → portable across machines
  - Compatible with ms-swift: just set `--dataset /path/to/dataset.jsonl`
    and ms-swift resolves relative image paths from the jsonl's parent dir
  - Default: copies images into output dir (full self-containment)

Usage:
    # Default: copy images + relative paths (self-contained, portable)
    python scripts/consolidate_trajectories.py \\
        --input-dir  data/results/distill_stage4_tasks \\
        --output-dir data/sft/distill_v1

    # Skip image copy (fast, for testing; images must be placed later)
    python scripts/consolidate_trajectories.py \\
        --input-dir  data/results/distill_stage4_tasks \\
        --output-dir data/sft/distill_v1 \\
        --no-copy-images
"""

import argparse
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed, progress bars disabled. Run: pip install tqdm")

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else iter([])


def sanitize_case_id(raw: str) -> str:
    """Make a filesystem-safe case id from raw row_id."""
    s = re.sub(r'[^\w\-.]', '_', raw)
    return s.strip('_')[:120]


def rewrite_image_paths(
    sample: Dict[str, Any],
    old_images_dir: Optional[Path],
    new_images_dir: Path,
    case_id: str,
    use_relative: bool = True,
    path_rewrites: Optional[List[tuple]] = None,
    do_copy: bool = True,
) -> Tuple[Dict[str, Any], int, int]:
    """Rewrite image paths to relative form, optionally copying files.

    Returns:
        (updated_sample, n_copied, n_missing)

    Always outputs relative paths like ``images/{case_id}/0001.jpg``.
    """
    old_images = sample.get("images", [])
    if not old_images:
        return sample, 0, 0

    out = dict(sample)
    new_paths = []
    n_copied = 0
    n_missing = 0

    if do_copy:
        case_img_dir = new_images_dir / case_id
        case_img_dir.mkdir(parents=True, exist_ok=True)

    for img_path in old_images:
        src = Path(img_path)
        fname = src.name  # e.g. 0001.jpg

        if do_copy:
            dst = new_images_dir / case_id / fname
            if dst.exists():
                pass  # already copied (e.g. resuming)
            else:
                real_src = _resolve_image_source(
                    img_path, src, old_images_dir, path_rewrites,
                )
                if real_src:
                    shutil.copy2(str(real_src), str(dst))
                    n_copied += 1
                else:
                    n_missing += 1

        new_paths.append(f"images/{case_id}/{fname}")

    out["images"] = new_paths
    return out, n_copied, n_missing


def _resolve_image_source(
    img_path: str,
    src: Path,
    old_images_dir: Optional[Path],
    path_rewrites: Optional[List[tuple]],
) -> Optional[Path]:
    """Find the actual file on disk for an image path.

    Tries, in order:
      1. The path as-is (covers existing absolute paths).
      2. The path treated as relative to old_images_dir (covers new-format
         all.jsonl whose paths are relative to the input dir).
      3. Apply each path_rewrites prefix substitution and re-check.
      4. Fallback: look up filename only inside old_images_dir.
    """
    if src.exists():
        return src
    if old_images_dir is not None:
        joined = old_images_dir / img_path
        if joined.exists():
            return joined
    for old_prefix, new_prefix in (path_rewrites or []):
        alt = Path(img_path.replace(old_prefix, new_prefix))
        if alt.exists():
            return alt
    if old_images_dir and (old_images_dir / src.name).exists():
        return old_images_dir / src.name
    return None


def consolidate(
    input_dir: Path,
    output_dir: Path,
    tar_images: bool = False,
    filter_valid: bool = True,
    path_rewrites: Optional[List[tuple]] = None,
    copy_images: bool = True,
    append: bool = False,
    dataset_name: str = "dataset.jsonl",
    metadata_name: str = "metadata.jsonl",
    skip_existing_ids: Optional[set] = None,
) -> Dict[str, int]:
    """Main consolidation logic.

    Args:
        append: if True, append to dataset/metadata files instead of overwriting.
        dataset_name: output jsonl filename (default 'dataset.jsonl';
                      use 'all.jsonl' to merge into a swift_exporter-style dir).
        skip_existing_ids: a set of case_id (or row_id) to skip — useful when
                           merging old data into a dir that already has some
                           cases from a fresh run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    dataset_path = output_dir / dataset_name
    metadata_path = output_dir / metadata_name
    file_mode = "a" if append else "w"

    stats = {
        "total": 0, "written": 0, "skipped": 0,
        "images_total": 0, "images_copied": 0, "images_missing": 0,
    }

    print(f"[1/3] Scanning input: {input_dir}")
    cases = _collect_cases(input_dir)
    if not cases:
        return stats
    print(f"      Found {len(cases)} cases")

    print(f"\n[2/3] Processing trajectories "
          f"({'copying images' if copy_images else 'paths only, no copy'}"
          f"{', append mode' if append else ''})")
    t0 = time.time()
    with open(dataset_path, file_mode) as f_data, open(metadata_path, file_mode) as f_meta:
        pbar = tqdm(
            cases, total=len(cases), unit="case", dynamic_ncols=True,
            desc="consolidate", mininterval=0.5, miniters=1,
        )
        for case in pbar:
            stats["total"] += 1
            sample = case["sample"]
            meta = case["meta"]
            case_id = case["case_id"]
            row_id = (meta or {}).get("row_id", case_id)

            if skip_existing_ids and (
                case_id in skip_existing_ids or row_id in skip_existing_ids
            ):
                stats.setdefault("dedup_skipped", 0)
                stats["dedup_skipped"] += 1
                pbar.set_postfix(
                    ok=stats["written"], skip=stats["skipped"],
                    dup=stats["dedup_skipped"],
                    imgs=stats["images_copied"], miss=stats["images_missing"],
                )
                continue

            if filter_valid:
                msgs = sample.get("messages", [])
                if not msgs or msgs[-1].get("role", "") != "assistant":
                    stats["skipped"] += 1
                    pbar.set_postfix(
                        ok=stats["written"], skip=stats["skipped"],
                        imgs=stats["images_copied"], miss=stats["images_missing"],
                    )
                    continue

            old_imgs_dir = case["images_dir"] or Path("/nonexistent")
            updated, n_copied, n_missing = rewrite_image_paths(
                sample, old_imgs_dir, images_dir, case_id,
                use_relative=True,
                path_rewrites=path_rewrites,
                do_copy=copy_images,
            )

            f_data.write(json.dumps(updated, ensure_ascii=False) + "\n")
            stats["written"] += 1
            stats["images_total"] += len(updated.get("images", []))
            stats["images_copied"] += n_copied
            stats["images_missing"] += n_missing

            if meta:
                meta_out = dict(meta)
                meta_out["_case_id"] = case_id
                f_meta.write(json.dumps(meta_out, ensure_ascii=False) + "\n")

            pbar.set_postfix(
                ok=stats["written"], skip=stats["skipped"],
                imgs=stats["images_copied"], miss=stats["images_missing"],
            )

    elapsed = time.time() - t0
    print(f"\n[3/3] Done in {elapsed:.1f}s")
    print(f"  Total cases: {stats['total']}")
    print(f"  Written:     {stats['written']}")
    print(f"  Skipped:     {stats['skipped']}  (invalid trajectories)")
    if stats.get("dedup_skipped"):
        print(f"  Dedup-skipped: {stats['dedup_skipped']}  "
              f"(already exist in target dir)")
    print(f"  Images referenced: {stats['images_total']}")
    if copy_images:
        print(f"  Images copied:     {stats['images_copied']}")
        if stats["images_missing"]:
            print(f"  Images MISSING:    {stats['images_missing']}  "
                  f"(source not found — try --path-rewrite)")
    print(f"\nOutput:")
    print(f"  {dataset_path}  ({stats['written']} samples)")
    print(f"  {metadata_path}")
    if copy_images:
        print(f"  {images_dir}/  (images copied)")
    else:
        print(f"  (images NOT copied — place at {images_dir}/ before training)")

    if tar_images and images_dir.exists():
        _pack_tar_with_progress(images_dir, output_dir / "images.tar")

    return stats


def _collect_cases(input_dir: Path) -> List[Dict[str, Any]]:
    """Walk input_dir and load all case sample.json + meta.json files."""
    traj_dir = input_dir / "trajectories"
    all_jsonl = input_dir / "all.jsonl"

    cases: List[Dict[str, Any]] = []
    if traj_dir.exists():
        case_dirs = sorted(d for d in traj_dir.iterdir() if d.is_dir())
        for case_dir in tqdm(case_dirs, unit="dir", desc="  scan",
                             dynamic_ncols=True, leave=False):
            sample_file = case_dir / "sample.json"
            if not sample_file.exists():
                continue
            try:
                with open(sample_file, "r") as f:
                    sample = json.load(f)
            except json.JSONDecodeError:
                continue
            meta = None
            meta_file = case_dir / "meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                except json.JSONDecodeError:
                    pass
            case_id = sanitize_case_id(case_dir.name.replace("case_", "", 1))
            cases.append({
                "case_id": case_id, "sample": sample, "meta": meta,
                "images_dir": case_dir / "images",
            })
    elif all_jsonl.exists():
        # Optional aligned metadata.jsonl (i-th line ↔ i-th sample)
        meta_jsonl = input_dir / "metadata.jsonl"
        metas: List[Optional[Dict[str, Any]]] = []
        if meta_jsonl.exists():
            with open(meta_jsonl, "r") as mf:
                for ml in mf:
                    ml = ml.strip()
                    if not ml:
                        continue
                    try:
                        metas.append(json.loads(ml))
                    except json.JSONDecodeError:
                        metas.append(None)

        with open(all_jsonl, "r") as f:
            for idx, line in enumerate(tqdm(f, unit="line", desc="  scan",
                                            dynamic_ncols=True, leave=False)):
                if not line.strip():
                    continue
                meta = metas[idx] if idx < len(metas) else None
                # Prefer real ids from metadata over generic sample_NNNNN
                if meta:
                    raw_id = meta.get("_case_id") or meta.get("row_id") or f"sample_{idx:05d}"
                    case_id = sanitize_case_id(str(raw_id))
                else:
                    case_id = f"sample_{idx:05d}"
                cases.append({
                    "case_id": case_id,
                    "sample": json.loads(line.strip()),
                    "meta": meta,
                    # Images are relative to the input dir in this layout
                    "images_dir": input_dir,
                })
    else:
        print(f"Error: neither {traj_dir} nor {all_jsonl} found in {input_dir}")
    return cases


def _pack_tar_with_progress(images_dir: Path, tar_path: Path) -> None:
    """Pack images_dir into a tar archive with a tqdm progress bar."""
    print(f"\n[+] Packing images into tar archive...")
    files = [p for p in images_dir.rglob("*") if p.is_file()]
    with tarfile.open(tar_path, "w") as tar:
        for f in tqdm(files, unit="file", desc="  tar",
                      dynamic_ncols=True):
            tar.add(str(f), arcname=str(f.relative_to(images_dir.parent)))
    tar_size = tar_path.stat().st_size / (1024 ** 3)
    print(f"  {tar_path}  ({tar_size:.2f} GB, {len(files)} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate fragmented trajectory data into portable layout."
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Input directory with trajectories/ or all.jsonl")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for consolidated data")
    parser.add_argument("--tar-images", action="store_true",
                        help="Also pack images into a tar archive for archival")
    parser.add_argument("--no-filter", action="store_true",
                        help="Don't filter out invalid trajectories (keep all)")
    parser.add_argument("--path-rewrite", type=str, nargs=2, action="append",
                        metavar=("OLD_PREFIX", "NEW_PREFIX"),
                        help="Path rewrite rule for broken image paths. "
                             "E.g. --path-rewrite /old/path /new/path")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Don't copy images, only rewrite paths to relative form. "
                             "Default: copy images into output dir for self-contained dataset.")
    args = parser.parse_args()

    path_rewrites = args.path_rewrite or []

    consolidate(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        tar_images=args.tar_images,
        filter_valid=not args.no_filter,
        path_rewrites=path_rewrites,
        copy_images=not args.no_copy_images,
    )


if __name__ == "__main__":
    main()
