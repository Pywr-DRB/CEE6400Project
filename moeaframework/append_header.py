#!/usr/bin/env python3
"""
Append_header.py
Step 9: Replace first two lines of each .set file with the policy header and append a hash.
- Input layout:
  outputs/
    Policy_PWL/
      1-header-file.txt
      refsets/<reservoir>/*.set
    Policy_RBF/
      1-header-file.txt
      refsets/<reservoir>/*.set
    Policy_STARFIT/
      1-header-file.txt
      refsets/<reservoir>/*.set
- Output: creates *_header.set next to each input .set
"""

import argparse
import hashlib
import sys
from pathlib import Path

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main(outputs_root: Path, suffix: str = "_header.set"):
    if not outputs_root.exists():
        print(f"[ERROR] Outputs root not found: {outputs_root}", file=sys.stderr)
        sys.exit(1)

    policy_dirs = sorted(p for p in outputs_root.glob("Policy_*") if p.is_dir())
    if not policy_dirs:
        print(f"[WARN] No Policy_* directories under {outputs_root}")
        return

    total_processed = 0
    for policy_dir in policy_dirs:
        header_file = policy_dir / "1-header-file.txt"
        refsets_root = policy_dir / "refsets"

        if not header_file.exists():
            print(f"[WARN] Missing header file for {policy_dir.name}: {header_file}")
            continue
        if not refsets_root.exists():
            print(f"[WARN] No refsets/ found for {policy_dir.name}: {refsets_root}")
            continue

        header_lines = header_file.read_text(encoding="utf-8").strip().splitlines()
        reservoirs = sorted(d for d in refsets_root.glob("*") if d.is_dir())
        if not reservoirs:
            print(f"[WARN] No reservoir subfolders in {refsets_root}")
            continue

        print(f">> Policy: {policy_dir.name} | Using header: {header_file.name}")

        for rdir in reservoirs:
            set_files = sorted(rdir.glob("*.set"))
            if not set_files:
                print(f"   - {rdir.name}: (no .set files)")
                continue

            print(f"   - Reservoir: {rdir.name} ({len(set_files)} files)")
            for set_path in set_files:
                try:
                    set_lines = set_path.read_text(encoding="utf-8").strip().splitlines()
                    # Replace first two lines with header (if file shorter than 2 lines, keep whatever exists)
                    tail = set_lines[2:] if len(set_lines) >= 2 else []
                    modified_lines = header_lines + tail
                    content = "\n".join(modified_lines) + "\n"
                    # Append hash comment
                    digest = sha256_str(content)
                    content += f"# sha256: {digest}\n"

                    out_path = set_path.with_name(set_path.stem + suffix)
                    out_path.write_text(content, encoding="utf-8")
                    total_processed += 1
                except Exception as e:
                    print(f"     ! ERROR processing {set_path}: {e}", file=sys.stderr)

    print(f">> Done. Wrote {total_processed} updated .set files with headers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace .set headers and append hash.")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"),
                        help="Root path to outputs/ (default: ./outputs)")
    parser.add_argument("--suffix", type=str, default="_header.set",
                        help="Suffix for output files (default: _header.set)")
    args = parser.parse_args()
    main(args.outputs_root, suffix=args.suffix)
