import os
import re
from collections import Counter
from config import BASE_DIR

DATASET_DIR = os.path.join(BASE_DIR, "dataset_preparation")
SOURCES = [
    os.path.join(BASE_DIR, "main_project"),
    os.path.join(BASE_DIR, "writing")
]
OLD_OUTPUT_DIR = os.path.join(DATASET_DIR, "output")
NEW_OUTPUT_DIR = os.path.join(DATASET_DIR, "output_lreal_auto")

os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

def find_incar(directory):
    if not directory or not os.path.isdir(directory):
        return None
    cur = os.path.abspath(directory)
    for _ in range(5):
        p = os.path.join(cur, "INCAR")
        if os.path.isfile(p):
            return p
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None

def is_lreal_auto(directory):
    p = find_incar(directory)
    if not p:
        return False
    try:
        with open(p, "r", errors="ignore") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip().upper() == "LREAL":
                        val = v.split("#")[0].strip().upper()
                        return val in ["AUTO", "A"]
    except Exception:
        pass
    return False

dir_cache = {}
pattern = re.compile(r'source_dir="?([^"\s]+)"?')

print("=" * 65)
print("Creating New Dataset Filtering for LREAL = Auto Only")
print(f"Source: {OLD_OUTPUT_DIR}")
print(f"Destination: {NEW_OUTPUT_DIR}")
print("=" * 65)

summary_counts = {}

for split in ["train", "val", "test"]:
    src_path = os.path.join(OLD_OUTPUT_DIR, f"{split}.xyz")
    dst_path = os.path.join(NEW_OUTPUT_DIR, f"{split}.xyz")

    total_in = 0
    total_kept = 0
    total_skipped = 0

    print(f"\nProcessing {split}.xyz...")
    with open(src_path, "r", errors="ignore") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        while True:
            line1 = fin.readline()
            if not line1:
                break
            line1_strip = line1.strip()
            if not line1_strip:
                continue
            natoms = int(line1_strip)
            line2 = fin.readline()

            # Read atom lines
            atom_lines = [fin.readline() for _ in range(natoms)]
            total_in += 1

            # Check source_dir
            m = pattern.search(line2)
            if not m:
                total_skipped += 1
                continue

            s_dir = m.group(1)
            if s_dir not in dir_cache:
                found_calc = None
                for src in SOURCES:
                    cand = os.path.join(src, s_dir)
                    if os.path.isdir(cand):
                        found_calc = cand
                        break
                dir_cache[s_dir] = is_lreal_auto(found_calc)

            if dir_cache[s_dir]:
                fout.write(line1)
                fout.write(line2)
                for al in atom_lines:
                    fout.write(al)
                total_kept += 1
            else:
                total_skipped += 1

    pct = (total_kept / total_in * 100) if total_in else 0
    summary_counts[split] = (total_in, total_kept, total_skipped, pct)
    print(f"  {split}: Input={total_in:,} | Kept (LREAL Auto)={total_kept:,} ({pct:.1f}%) | Skipped (LREAL False)={total_skipped:,}")
    print(f"  Wrote: {dst_path}")

print("\n" + "=" * 65)
print("Dataset Creation Complete!")
print(f"{'Split':<8} | {'Total In':<10} | {'Kept (Auto)':<12} | {'Percentage':<10}")
print("-" * 50)
total_in_all = sum(v[0] for v in summary_counts.values())
total_kept_all = sum(v[1] for v in summary_counts.values())
for s, (tin, kept, skip, pct) in summary_counts.items():
    print(f"{s:<8} | {tin:<10,d} | {kept:<12,d} | {pct:>8.1f}%")
print("-" * 50)
print(f"{'Total':<8} | {total_in_all:<10,d} | {total_kept_all:<12,d} | {total_kept_all/total_in_all*100:>8.1f}%")