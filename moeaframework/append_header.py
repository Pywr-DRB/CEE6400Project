#!/usr/bin/env python3
"""
Step 2: Write *_header.set (tutorial-style), filtering by seed [seed_from..seed_to].
"""
from __future__ import annotations
import argparse, sys, hashlib, re
from pathlib import Path

SEED_RX = re.compile(r"_seed(\d+)_")

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def in_seed_window(name: str, s_from: int, s_to: int) -> bool:
    m = SEED_RX.search(name)
    if not m:
        return False  # no seed tag; skip
    seed = int(m.group(1))
    return s_from <= seed <= s_to

def main(outputs_root: Path, suffix: str, seed_from: int, seed_to: int) -> None:
    if not outputs_root.exists():
        print(f"[ERROR] missing {outputs_root}", file=sys.stderr); sys.exit(1)

    for policy_dir in sorted(p for p in outputs_root.glob("Policy_*") if p.is_dir()):
        header_file = policy_dir / "1-header-file.txt"
        refsets_root = policy_dir / "refsets"
        if not header_file.exists():
            print(f"[WARN] {policy_dir.name}: no header file -> skip"); continue
        if not refsets_root.exists():
            print(f"[WARN] {policy_dir.name}: no refsets/ -> skip"); continue
        header_lines = header_file.read_text(encoding="utf-8").strip().splitlines()
        print(f">> Policy: {policy_dir.name} | Header: {header_file.name}")
        for rdir in sorted(d for d in refsets_root.glob("*") if d.is_dir()):
            files = sorted(p for p in rdir.glob("*.set") if in_seed_window(p.name, seed_from, seed_to))
            if not files:
                print(f"   - {rdir.name}: (no .set in seed window)"); continue
            print(f"   - Reservoir: {rdir.name} ({len(files)} files)")
            for set_path in files:
                set_lines = set_path.read_text(encoding="utf-8").strip().splitlines()
                tail = set_lines[2:] if len(set_lines) >= 2 else []
                content = "\n".join(header_lines + tail) + "\n"
                content += f"# sha256: {sha256_str(content)}\n"
                out_path = set_path.with_name(set_path.stem + suffix)
                out_path.write_text(content, encoding="utf-8")
    print(">> Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    ap.add_argument("--suffix", type=str, default="_header.set")
    ap.add_argument("--seed-from", type=int, default=1)
    ap.add_argument("--seed-to", type=int, default=10)
    args = ap.parse_args()
    main(args.outputs_root, suffix=args.suffix, seed_from=args.seed_from, seed_to=args.seed_to)
