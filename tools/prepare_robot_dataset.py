"""Create a lightweight ROBOT dataset index for TL-Fault-Diagnosis-Library.

The generated files are small UTF-8 pointers to the original Excel files. The
ROBOT loader in data_loader/load_methods.py reads those pointers and then loads
the 18 acceleration channels from the real workbook.
"""

import argparse
import hashlib
import re
from pathlib import Path


CLASS_DIRS = {
    "normal": "00_normal",
    "4": "01_axis4",
    "5": "02_axis5",
    "6": "03_axis6",
    "45": "04_axis45",
    "46": "05_axis46",
    "56": "06_axis56",
    "456": "07_axis456",
}
TRAJECTORY_TO_CONDITION = {
    "old": 0,
    "new1": 1,
    "new2": 2,
    "new3": 3,
}


def infer_class(folder: Path, filename: str) -> str:
    lower_name = filename.lower()
    if lower_name.startswith("normal"):
        return CLASS_DIRS["normal"]

    match = re.search(r"axis(456|45|46|56|4|5|6)", lower_name)
    if match is None:
        match = re.search(r"abnormal_(456|45|46|56|4|5|6)axis", lower_name)
    if match is None:
        match = re.search(r"(456|45|46|56|4|5|6)", folder.name)
    if match is not None:
        return CLASS_DIRS[match.group(1)]
    raise ValueError(f"Cannot infer robot fault class for {folder / filename}")


def infer_trajectory(filename: str) -> str:
    match = re.search(r"(old|new\d+)", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer trajectory for {filename}")
    return match.group(1).lower()


def pointer_name(source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem)
    return f"{stem}_{digest}.robot"


def build_index(raw_root: Path, output_root: Path, max_files_per_class_condition: int = 0) -> int:
    output_dataset_root = output_root / "ROBOT"
    seen = {}
    count = 0
    for folder in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        if folder.name in {"external", "work"}:
            continue
        for excel_path in sorted(folder.glob("*.xlsx")):
            class_dir = infer_class(folder, excel_path.name)
            trajectory = infer_trajectory(excel_path.name)
            if trajectory not in TRAJECTORY_TO_CONDITION:
                continue
            condition = TRAJECTORY_TO_CONDITION[trajectory]
            key = (condition, class_dir)
            if max_files_per_class_condition and seen.get(key, 0) >= max_files_per_class_condition:
                continue
            target_dir = output_dataset_root / f"condition_{condition}" / class_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / pointer_name(excel_path)
            target_file.write_text(str(excel_path.resolve()), encoding="utf-8")
            seen[key] = seen.get(key, 0) + 1
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare ROBOT pointer dataset")
    parser.add_argument("--raw_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--max_files_per_class_condition", type=int, default=0)
    args = parser.parse_args()

    count = build_index(
        args.raw_root,
        args.output_root,
        max_files_per_class_condition=args.max_files_per_class_condition,
    )
    print(f"Indexed {count} Excel files under {args.output_root / 'ROBOT'}")


if __name__ == "__main__":
    main()
