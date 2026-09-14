import csv
import hashlib
from collections import Counter
from pathlib import Path


# =========================================================
# Sparse2Risk - Frozen Split Validator
# =========================================================
#
# This script independently validates the frozen drive-level
# split against:
#
#   1. the full frame-level KITTI manifest;
#   2. the original pilot RGB folder;
#   3. the expected SHA-256 fingerprint;
#   4. the frozen partition counts and source-split policy.
#
# IMPORTANT:
# If any check fails, do not use the frozen split until the
# cause has been investigated.
# =========================================================


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manifests"
    / "kitti_full_depth_manifest.csv"
)

FROZEN_SPLIT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "splits"
    / "kitti_sparse2risk_v1.csv"
)

PILOT_RGB_DIR = Path(
    r"C:\FYP\dataset\depth_selection"
    r"\val_selection_cropped\image"
)

EXPECTED_SHA256 = (
    "6ece3eca566bfe1b888efc0f97c8bc73"
    "c523774c61f0979d260b6459394c2337"
)

EXPECTED_TOTAL_DRIVES = 151
EXPECTED_TOTAL_FRAMES = 46_375

EXPECTED_DRIVE_COUNTS = {
    "TRAIN": 101,
    "DEV": 13,
    "CALIBRATION": 22,
    "LOCKED_TEST": 15,
}

EXPECTED_FRAME_COUNTS = {
    "TRAIN": 34_359,
    "DEV": 3_426,
    "CALIBRATION": 4_295,
    "LOCKED_TEST": 4_295,
}

EXPECTED_ALLOWED_SPLITS = set(
    EXPECTED_DRIVE_COUNTS
)


# =========================================================
# Helpers
# =========================================================

def calculate_sha256(path: Path) -> str:
    """
    Calculate SHA-256 using binary file contents.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_csv(path: Path):
    """
    Read a CSV file into a list of dictionaries.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def extract_pilot_identity(filename: str):
    """
    Extract:
        drive, frame, camera

    from pilot RGB filenames such as:

    2011_09_26_drive_0002_sync_image_0000000005_image_02.png
    """

    marker = "_image_"

    if marker not in filename:
        return None

    drive, remaining = filename.split(
        marker,
        maxsplit=1,
    )

    parts = remaining.rsplit(
        "_image_",
        maxsplit=1,
    )

    if len(parts) != 2:
        return None

    frame = parts[0]

    camera = (
        "image_"
        + Path(parts[1]).stem
    )

    return (
        drive,
        frame,
        camera,
    )


def print_check(
    label: str,
    passed: bool,
):
    """
    Print one validation result.
    """

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{label:<54}: "
        f"{status}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 82)
    print(
        "Sparse2Risk - Frozen KITTI Split Validation"
    )
    print("=" * 82)

    # -----------------------------------------------------
    # Load files
    # -----------------------------------------------------

    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    split_rows = read_csv(
        FROZEN_SPLIT_PATH
    )

    # -----------------------------------------------------
    # SHA-256
    # -----------------------------------------------------

    actual_sha256 = calculate_sha256(
        FROZEN_SPLIT_PATH
    )

    hash_matches = (
        actual_sha256
        == EXPECTED_SHA256
    )

    print()
    print("Frozen split fingerprint")
    print("-" * 82)

    print(
        f"Expected SHA-256 : {EXPECTED_SHA256}"
    )

    print(
        f"Actual SHA-256   : {actual_sha256}"
    )

    print_check(
        "Frozen split SHA-256 unchanged",
        hash_matches,
    )

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_split_columns = {
        "drive",
        "date",
        "frames",
        "source_split",
        "proposed_split",
    }

    split_columns = (
        set(split_rows[0].keys())
        if split_rows
        else set()
    )

    required_columns_present = (
        required_split_columns
        .issubset(split_columns)
    )

    # -----------------------------------------------------
    # Frozen split row-level structure
    # -----------------------------------------------------

    frozen_drives = [
        row["drive"]
        for row in split_rows
    ]

    unique_frozen_drives = set(
        frozen_drives
    )

    duplicate_frozen_drive_count = (
        len(frozen_drives)
        - len(unique_frozen_drives)
    )

    split_names = {
        row["proposed_split"]
        for row in split_rows
    }

    only_expected_split_names = (
        split_names
        == EXPECTED_ALLOWED_SPLITS
    )

    # -----------------------------------------------------
    # Parse numerical fields in frozen split
    # -----------------------------------------------------

    split_frame_count_field_valid = True

    for row in split_rows:
        try:
            row["frames"] = int(
                row["frames"]
            )
        except (
            TypeError,
            ValueError,
        ):
            split_frame_count_field_valid = False
            break

    # -----------------------------------------------------
    # Frozen split counts
    # -----------------------------------------------------

    frozen_drive_counts = Counter()

    frozen_frame_counts = Counter()

    if split_frame_count_field_valid:

        for row in split_rows:

            split_name = (
                row["proposed_split"]
            )

            frozen_drive_counts[
                split_name
            ] += 1

            frozen_frame_counts[
                split_name
            ] += row["frames"]

    drive_counts_match = (
        dict(frozen_drive_counts)
        == EXPECTED_DRIVE_COUNTS
    )

    frame_counts_match = (
        dict(frozen_frame_counts)
        == EXPECTED_FRAME_COUNTS
    )

    total_frozen_drives_match = (
        len(split_rows)
        == EXPECTED_TOTAL_DRIVES
    )

    total_frozen_frames_match = (
        sum(
            frozen_frame_counts.values()
        )
        == EXPECTED_TOTAL_FRAMES
    )

    # -----------------------------------------------------
    # Full manifest structure
    # -----------------------------------------------------

    manifest_sample_ids = [
        (
            row["drive"],
            row["frame"],
            row["camera"],
        )
        for row in manifest_rows
    ]

    manifest_unique_sample_ids = set(
        manifest_sample_ids
    )

    manifest_has_unique_samples = (
        len(manifest_sample_ids)
        == len(manifest_unique_sample_ids)
    )

    manifest_total_frames_match = (
        len(manifest_rows)
        == EXPECTED_TOTAL_FRAMES
    )

    # -----------------------------------------------------
    # Recompute drive statistics from manifest
    # -----------------------------------------------------

    manifest_frames_per_drive = Counter()

    manifest_source_by_drive = {}

    source_conflict = False

    for row in manifest_rows:

        drive = row["drive"]

        manifest_frames_per_drive[
            drive
        ] += 1

        source_split = (
            row["source_split"]
        )

        if drive in manifest_source_by_drive:

            if (
                manifest_source_by_drive[drive]
                != source_split
            ):

                source_conflict = True

        manifest_source_by_drive[
            drive
        ] = source_split

    manifest_drives = set(
        manifest_frames_per_drive
    )

    manifest_drive_count_match = (
        len(manifest_drives)
        == EXPECTED_TOTAL_DRIVES
    )

    every_manifest_drive_is_frozen = (
        manifest_drives
        == unique_frozen_drives
    )

    # -----------------------------------------------------
    # Validate every frozen drive against manifest
    # -----------------------------------------------------

    frozen_frames_match_manifest = True
    frozen_sources_match_manifest = True
    frozen_dates_match_drive_names = True

    split_by_drive = {}

    for row in split_rows:

        drive = row["drive"]

        split_by_drive[
            drive
        ] = row["proposed_split"]

        if split_frame_count_field_valid:

            if (
                manifest_frames_per_drive.get(
                    drive
                )
                != row["frames"]
            ):

                frozen_frames_match_manifest = False

        if (
            manifest_source_by_drive.get(
                drive
            )
            != row["source_split"]
        ):

            frozen_sources_match_manifest = False

        if (
            row["date"]
            != drive[:10]
        ):

            frozen_dates_match_drive_names = False

    # -----------------------------------------------------
    # Source-split policy
    # -----------------------------------------------------

    source_policy_valid = True

    original_val_drives = {
        drive
        for (
            drive,
            source_split,
        ) in manifest_source_by_drive.items()
        if source_split == "val"
    }

    original_train_drives = {
        drive
        for (
            drive,
            source_split,
        ) in manifest_source_by_drive.items()
        if source_split == "train"
    }

    dev_drives = {
        row["drive"]
        for row in split_rows
        if row["proposed_split"] == "DEV"
    }

    train_drives = {
        row["drive"]
        for row in split_rows
        if row["proposed_split"] == "TRAIN"
    }

    calibration_drives = {
        row["drive"]
        for row in split_rows
        if (
            row["proposed_split"]
            == "CALIBRATION"
        )
    }

    locked_test_drives = {
        row["drive"]
        for row in split_rows
        if (
            row["proposed_split"]
            == "LOCKED_TEST"
        )
    }

    if (
        dev_drives
        != original_val_drives
    ):
        source_policy_valid = False

    non_dev_drives = (
        train_drives
        | calibration_drives
        | locked_test_drives
    )

    if (
        non_dev_drives
        != original_train_drives
    ):
        source_policy_valid = False

    # -----------------------------------------------------
    # Partition overlap
    # -----------------------------------------------------

    partition_sets = {
        "TRAIN": train_drives,
        "DEV": dev_drives,
        "CALIBRATION": calibration_drives,
        "LOCKED_TEST": locked_test_drives,
    }

    partition_names = list(
        partition_sets
    )

    no_drive_overlap = True

    for first_index in range(
        len(partition_names)
    ):

        for second_index in range(
            first_index + 1,
            len(partition_names),
        ):

            first_name = (
                partition_names[
                    first_index
                ]
            )

            second_name = (
                partition_names[
                    second_index
                ]
            )

            if not (
                partition_sets[
                    first_name
                ].isdisjoint(
                    partition_sets[
                        second_name
                    ]
                )
            ):

                no_drive_overlap = False

    # -----------------------------------------------------
    # Pilot isolation
    # -----------------------------------------------------

    pilot_files = sorted(
        PILOT_RGB_DIR.glob("*.png")
    )

    pilot_identities = set()

    unparsed_pilot_files = []

    for path in pilot_files:

        identity = extract_pilot_identity(
            path.name
        )

        if identity is None:

            unparsed_pilot_files.append(
                path.name
            )

            continue

        pilot_identities.add(
            identity
        )

    pilot_drives = {
        drive
        for (
            drive,
            _,
            _,
        ) in pilot_identities
    }

    exactly_1000_pilot_files = (
        len(pilot_files) == 1000
    )

    exactly_1000_pilot_identities = (
        len(pilot_identities) == 1000
    )

    no_unparsed_pilot_names = (
        len(unparsed_pilot_files) == 0
    )

    pilot_drives_exactly_dev = (
        pilot_drives
        == dev_drives
    )

    pilot_drives_outside_dev = (
        pilot_drives
        & non_dev_drives
    )

    no_pilot_drive_outside_dev = (
        len(
            pilot_drives_outside_dev
        )
        == 0
    )

    # -----------------------------------------------------
    # Recompute partition frame totals from full manifest
    # rather than trusting the frozen CSV's frame column.
    # -----------------------------------------------------

    manifest_partition_frame_counts = Counter()

    unknown_manifest_drive_count = 0

    for row in manifest_rows:

        split_name = split_by_drive.get(
            row["drive"]
        )

        if split_name is None:

            unknown_manifest_drive_count += 1

            continue

        manifest_partition_frame_counts[
            split_name
        ] += 1

    manifest_partition_counts_match = (
        dict(
            manifest_partition_frame_counts
        )
        == EXPECTED_FRAME_COUNTS
    )

    no_unknown_manifest_drives = (
        unknown_manifest_drive_count == 0
    )

    # -----------------------------------------------------
    # Print summaries
    # -----------------------------------------------------

    print()
    print("=" * 82)
    print("Frozen partition summary")
    print("=" * 82)

    for split_name in (
        "TRAIN",
        "DEV",
        "CALIBRATION",
        "LOCKED_TEST",
    ):

        print(
            f"{split_name:<13}"
            f"drives="
            f"{frozen_drive_counts[split_name]:3,d}  "
            f"frames="
            f"{frozen_frame_counts[split_name]:6,d}"
        )

    print()
    print("=" * 82)
    print("Validation checks")
    print("=" * 82)

    checks = {
        "Frozen SHA-256 matches recorded fingerprint":
            hash_matches,

        "Required frozen-split columns are present":
            required_columns_present,

        "Frozen frame-count column is numeric":
            split_frame_count_field_valid,

        "Frozen split has no duplicate drive rows":
            duplicate_frozen_drive_count == 0,

        "Frozen split contains only expected partition names":
            only_expected_split_names,

        "Frozen split has exactly 151 drives":
            total_frozen_drives_match,

        "Frozen drive counts match recorded counts":
            drive_counts_match,

        "Frozen frame totals match recorded counts":
            frame_counts_match,

        "Frozen total frame count is 46,375":
            total_frozen_frames_match,

        "Manifest contains exactly 46,375 frame rows":
            manifest_total_frames_match,

        "Manifest frame identities are unique":
            manifest_has_unique_samples,

        "Manifest has exactly 151 unique drives":
            manifest_drive_count_match,

        "Manifest has no drive source-split conflicts":
            not source_conflict,

        "Frozen and manifest drive sets are identical":
            every_manifest_drive_is_frozen,

        "Frozen per-drive frame counts match manifest":
            frozen_frames_match_manifest,

        "Frozen source_split values match manifest":
            frozen_sources_match_manifest,

        "Frozen date values match drive names":
            frozen_dates_match_drive_names,

        "DEV exactly equals original KITTI val drives":
            dev_drives == original_val_drives,

        "TRAIN/CAL/TEST exactly equal original train drives":
            non_dev_drives == original_train_drives,

        "Source-split partition policy is valid":
            source_policy_valid,

        "No drive overlaps between final partitions":
            no_drive_overlap,

        "Pilot folder contains exactly 1,000 RGB files":
            exactly_1000_pilot_files,

        "All pilot filenames parse successfully":
            no_unparsed_pilot_names,

        "Pilot contains exactly 1,000 unique identities":
            exactly_1000_pilot_identities,

        "Pilot-seen drives exactly equal DEV drives":
            pilot_drives_exactly_dev,

        "No pilot-seen drive occurs outside DEV":
            no_pilot_drive_outside_dev,

        "Every manifest frame maps to a frozen drive":
            no_unknown_manifest_drives,

        "Recomputed partition frame totals match frozen policy":
            manifest_partition_counts_match,
    }

    for label, passed in checks.items():

        print_check(
            label,
            passed,
        )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    all_checks_pass = all(
        checks.values()
    )

    print()
    print("=" * 82)

    if all_checks_pass:

        print(
            "PASS: The Sparse2Risk KITTI v1 split is "
            "internally consistent and unchanged."
        )

        print(
            "The frozen split can now be committed "
            "to version control."
        )

    else:

        print(
            "FAIL: One or more frozen-split checks failed."
        )

        print(
            "Do NOT use or commit the split until "
            "the failure has been investigated."
        )

    print("=" * 82)


if __name__ == "__main__":
    main()
