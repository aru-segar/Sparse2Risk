import csv
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------
# Project and dataset configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = Path(
    r"C:\FYP\dataset"
)

SOURCE_SPLITS = (
    "train",
    "val",
)

CAMERA = "image_02"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "manifests"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "kitti_full_depth_manifest.csv"
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def collect_drive_samples(
    source_split: str,
    drive_path: Path,
):
    """
    Collect matched current-LiDAR and reference-depth frames
    for one KITTI drive.

    Only image_02 is used in the primary Sparse2Risk study.
    """

    lidar_dir = (
        drive_path
        / "proj_depth"
        / "velodyne_raw"
        / CAMERA
    )

    reference_dir = (
        drive_path
        / "proj_depth"
        / "groundtruth"
        / CAMERA
    )

    result = {
        "rows": [],
        "missing_lidar_dir": False,
        "missing_reference_dir": False,
        "lidar_without_reference": [],
        "reference_without_lidar": [],
    }

    if not lidar_dir.exists():
        result["missing_lidar_dir"] = True
        return result

    if not reference_dir.exists():
        result["missing_reference_dir"] = True
        return result

    lidar_files = {
        path.name: path
        for path in lidar_dir.glob("*.png")
    }

    reference_files = {
        path.name: path
        for path in reference_dir.glob("*.png")
    }

    lidar_names = set(lidar_files)
    reference_names = set(reference_files)

    matched_names = sorted(
        lidar_names & reference_names
    )

    result["lidar_without_reference"] = sorted(
        lidar_names - reference_names
    )

    result["reference_without_lidar"] = sorted(
        reference_names - lidar_names
    )

    for filename in matched_names:

        frame = Path(filename).stem

        result["rows"].append(
            {
                "source_split": source_split,
                "drive": drive_path.name,
                "frame": frame,
                "camera": CAMERA,
                "lidar_path": str(
                    lidar_files[filename].resolve()
                ),
                "reference_path": str(
                    reference_files[filename].resolve()
                ),
            }
        )

    return result

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 72)
    print("Sparse2Risk - Full KITTI Depth Manifest Builder")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_rows = []

    drive_counts = Counter()
    frame_counts = Counter()

    missing_lidar_directories = []
    missing_reference_directories = []

    lidar_without_reference = []
    reference_without_lidar = []

    all_drive_ids = []

    # -----------------------------------------------------
    # Scan train and val
    # -----------------------------------------------------

    for source_split in SOURCE_SPLITS:

        split_root = (
            DATASET_ROOT
            / source_split
        )

        print()
        print(
            f"Scanning source split: "
            f"{source_split}"
        )

        print(
            f"Path: {split_root}"
        )

        if not split_root.exists():
            raise FileNotFoundError(
                f"Dataset folder does not exist: "
                f"{split_root}"
            )

        drive_paths = sorted(
            path
            for path in split_root.iterdir()
            if path.is_dir()
        )

        drive_counts[source_split] = len(
            drive_paths
        )

        print(
            f"Drive folders discovered: "
            f"{len(drive_paths):,}"
        )

        split_frame_total = 0

        for drive_index, drive_path in enumerate(
            drive_paths,
            start=1,
        ):

            all_drive_ids.append(
                (
                    source_split,
                    drive_path.name,
                )
            )

            result = collect_drive_samples(
                source_split=source_split,
                drive_path=drive_path,
            )

            if result["missing_lidar_dir"]:

                missing_lidar_directories.append(
                    (
                        source_split,
                        drive_path.name,
                    )
                )

                continue

            if result["missing_reference_dir"]:

                missing_reference_directories.append(
                    (
                        source_split,
                        drive_path.name,
                    )
                )

                continue

            for filename in result[
                "lidar_without_reference"
            ]:

                lidar_without_reference.append(
                    (
                        source_split,
                        drive_path.name,
                        filename,
                    )
                )

            for filename in result[
                "reference_without_lidar"
            ]:

                reference_without_lidar.append(
                    (
                        source_split,
                        drive_path.name,
                        filename,
                    )
                )

            rows = result["rows"]

            manifest_rows.extend(
                rows
            )

            split_frame_total += len(
                rows
            )

            if (
                drive_index % 20 == 0
                or drive_index == len(drive_paths)
            ):

                print(
                    f"  Processed drives: "
                    f"{drive_index:3d} / "
                    f"{len(drive_paths)}"
                )

        frame_counts[source_split] = (
            split_frame_total
        )

        print(
            f"Matched image_02 frames: "
            f"{split_frame_total:,}"
        )

    # -----------------------------------------------------
    # Check whether any drive ID appears in both
    # original source folders
    # -----------------------------------------------------

    train_drives = {
        drive
        for split, drive in all_drive_ids
        if split == "train"
    }

    val_drives = {
        drive
        for split, drive in all_drive_ids
        if split == "val"
    }

    overlapping_drives = sorted(
        train_drives & val_drives
    )

    # -----------------------------------------------------
    # Check uniqueness of sample identities
    # -----------------------------------------------------

    sample_ids = [
        (
            row["drive"],
            row["frame"],
            row["camera"],
        )
        for row in manifest_rows
    ]

    unique_sample_ids = set(
        sample_ids
    )

    duplicate_sample_count = (
        len(sample_ids)
        - len(unique_sample_ids)
    )

    # -----------------------------------------------------
    # Write CSV
    # -----------------------------------------------------

    fieldnames = [
        "source_split",
        "drive",
        "frame",
        "camera",
        "lidar_path",
        "reference_path",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Manifest summary")
    print("=" * 72)

    print(
        f"Train drive folders        : "
        f"{drive_counts['train']:,}"
    )

    print(
        f"Val drive folders          : "
        f"{drive_counts['val']:,}"
    )

    print(
        f"Total drive folders        : "
        f"{drive_counts['train'] + drive_counts['val']:,}"
    )

    print()

    print(
        f"Train matched frames       : "
        f"{frame_counts['train']:,}"
    )

    print(
        f"Val matched frames         : "
        f"{frame_counts['val']:,}"
    )

    print(
        f"Total matched frames       : "
        f"{len(manifest_rows):,}"
    )

    print()

    print(
        f"Missing LiDAR directories  : "
        f"{len(missing_lidar_directories):,}"
    )

    print(
        f"Missing reference dirs     : "
        f"{len(missing_reference_directories):,}"
    )

    print(
        f"LiDAR without reference    : "
        f"{len(lidar_without_reference):,}"
    )

    print(
        f"Reference without LiDAR    : "
        f"{len(reference_without_lidar):,}"
    )

    print()

    print(
        f"Duplicate sample IDs       : "
        f"{duplicate_sample_count:,}"
    )

    print(
        f"Drive IDs in train AND val : "
        f"{len(overlapping_drives):,}"
    )

    if overlapping_drives:

        print()
        print(
            "Overlapping drive IDs:"
        )

        for drive in overlapping_drives:

            print(
                f"  {drive}"
            )

    print()
    print(
        f"Manifest saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # -----------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Integrity checks")
    print("=" * 72)

    has_frames = (
        len(manifest_rows) > 0
    )

    no_missing_directories = (
        len(missing_lidar_directories) == 0
        and len(missing_reference_directories) == 0
    )

    no_unmatched_frames = (
        len(lidar_without_reference) == 0
        and len(reference_without_lidar) == 0
    )

    no_duplicate_samples = (
        duplicate_sample_count == 0
    )

    print(
        f"Manifest contains frames        : "
        f"{has_frames}"
    )

    print(
        f"No missing depth directories    : "
        f"{no_missing_directories}"
    )

    print(
        f"No unmatched depth frames       : "
        f"{no_unmatched_frames}"
    )

    print(
        f"No duplicate sample identities  : "
        f"{no_duplicate_samples}"
    )

    print()

    if (
        has_frames
        and no_missing_directories
        and no_unmatched_frames
        and no_duplicate_samples
    ):

        print(
            "PASS: Full KITTI depth manifest "
            "built successfully."
        )

    else:

        print(
            "WARNING: Manifest requires further "
            "investigation before splitting."
        )

    print("=" * 72)

if __name__ == "__main__":
    main()