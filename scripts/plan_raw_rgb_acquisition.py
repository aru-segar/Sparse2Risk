import csv
from collections import Counter, defaultdict
from pathlib import Path


# =========================================================
# Sparse2Risk - KITTI Raw RGB Acquisition Audit
# =========================================================
#
# Purpose
# -------
# Read the frozen Sparse2Risk study split and the full
# image_02 depth manifest, then determine exactly which
# KITTI raw RGB frames are required.
#
# This script DOES NOT download anything.
#
# Expected raw RGB layout:
#
# C:\FYP\dataset\kitti_raw\
#   2011_09_26\
#     2011_09_26_drive_0001_sync\
#       image_02\
#         data\
#           0000000005.png
#           ...
#
# The raw RGB data remain separate from the depth-completion
# folders. Matching is performed by:
#
#   drive + frame + camera
#
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

RAW_RGB_ROOT = Path(
    r"C:\FYP\dataset\kitti_raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "raw_rgb_audit"
)

DRIVE_REPORT_PATH = (
    OUTPUT_DIR
    / "required_raw_rgb_drives.csv"
)

MISSING_FRAME_REPORT_PATH = (
    OUTPUT_DIR
    / "missing_raw_rgb_frames.csv"
)

EXPECTED_CAMERA = "image_02"

EXPECTED_TOTAL_DRIVES = 151
EXPECTED_TOTAL_FRAMES = 46_375


# =========================================================
# Helpers
# =========================================================

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


def print_check(
    label: str,
    passed: bool,
):
    """
    Print one PASS/FAIL validation line.
    """

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{label:<56}: {status}"
    )


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 82)
    print(
        "Sparse2Risk - KITTI Raw RGB Acquisition Audit"
    )
    print("=" * 82)

    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    frozen_split_rows = read_csv(
        FROZEN_SPLIT_PATH
    )

    # -----------------------------------------------------
    # Validate and index frozen split
    # -----------------------------------------------------

    split_by_drive = {}

    source_by_drive = {}

    expected_frames_by_drive = {}

    duplicate_split_drive = False

    for row in frozen_split_rows:

        drive = row["drive"]

        if drive in split_by_drive:
            duplicate_split_drive = True

        split_by_drive[drive] = (
            row["proposed_split"]
        )

        source_by_drive[drive] = (
            row["source_split"]
        )

        expected_frames_by_drive[drive] = int(
            row["frames"]
        )

    # -----------------------------------------------------
    # Build required RGB frame lists from full manifest
    # -----------------------------------------------------

    required_frames_by_drive = defaultdict(
        set
    )

    manifest_source_by_drive = {}

    manifest_camera_values = set()

    duplicate_manifest_identity = False

    manifest_identities = set()

    for row in manifest_rows:

        drive = row["drive"]
        frame = row["frame"]
        camera = row["camera"]

        identity = (
            drive,
            frame,
            camera,
        )

        if identity in manifest_identities:
            duplicate_manifest_identity = True

        manifest_identities.add(
            identity
        )

        manifest_camera_values.add(
            camera
        )

        required_frames_by_drive[
            drive
        ].add(
            frame
        )

        manifest_source_by_drive[
            drive
        ] = row["source_split"]

    manifest_drives = set(
        required_frames_by_drive
    )

    frozen_drives = set(
        split_by_drive
    )

    # -----------------------------------------------------
    # Overall manifest checks
    # -----------------------------------------------------

    manifest_count_correct = (
        len(manifest_rows)
        == EXPECTED_TOTAL_FRAMES
    )

    drive_count_correct = (
        len(manifest_drives)
        == EXPECTED_TOTAL_DRIVES
    )

    frozen_and_manifest_drives_match = (
        manifest_drives
        == frozen_drives
    )

    only_expected_camera = (
        manifest_camera_values
        == {EXPECTED_CAMERA}
    )

    no_duplicate_manifest_identity = (
        not duplicate_manifest_identity
    )

    no_duplicate_split_drive = (
        not duplicate_split_drive
    )

    # -----------------------------------------------------
    # Verify per-drive frame counts against frozen split
    # -----------------------------------------------------

    per_drive_counts_match = True

    for drive in manifest_drives:

        actual_count = len(
            required_frames_by_drive[
                drive
            ]
        )

        expected_count = (
            expected_frames_by_drive.get(
                drive
            )
        )

        if actual_count != expected_count:
            per_drive_counts_match = False
            break

    # -----------------------------------------------------
    # Audit raw RGB folders
    # -----------------------------------------------------

    drive_report_rows = []

    missing_frame_rows = []

    split_required_counts = Counter()
    split_matched_counts = Counter()
    split_missing_counts = Counter()

    total_existing_png_files = 0
    total_matched_required = 0
    total_missing_required = 0

    for drive in sorted(
        manifest_drives
    ):

        date = drive[:10]

        split_name = split_by_drive[
            drive
        ]

        source_split = source_by_drive[
            drive
        ]

        required_frames = (
            required_frames_by_drive[
                drive
            ]
        )

        rgb_dir = (
            RAW_RGB_ROOT
            / date
            / drive
            / EXPECTED_CAMERA
            / "data"
        )

        existing_frame_names = set()

        if rgb_dir.exists():

            for path in rgb_dir.glob(
                "*.png"
            ):

                existing_frame_names.add(
                    path.stem
                )

        matched_frames = (
            required_frames
            & existing_frame_names
        )

        missing_frames = (
            required_frames
            - existing_frame_names
        )

        extra_existing_frames = (
            existing_frame_names
            - required_frames
        )

        required_count = len(
            required_frames
        )

        existing_count = len(
            existing_frame_names
        )

        matched_count = len(
            matched_frames
        )

        missing_count = len(
            missing_frames
        )

        extra_count = len(
            extra_existing_frames
        )

        total_existing_png_files += (
            existing_count
        )

        total_matched_required += (
            matched_count
        )

        total_missing_required += (
            missing_count
        )

        split_required_counts[
            split_name
        ] += required_count

        split_matched_counts[
            split_name
        ] += matched_count

        split_missing_counts[
            split_name
        ] += missing_count

        drive_report_rows.append(
            {
                "proposed_split": split_name,
                "source_split": source_split,
                "date": date,
                "drive": drive,
                "camera": EXPECTED_CAMERA,
                "required_frames": required_count,
                "raw_rgb_directory": str(
                    rgb_dir
                ),
                "raw_rgb_directory_exists": (
                    rgb_dir.exists()
                ),
                "existing_png_files": existing_count,
                "matched_required_frames": matched_count,
                "missing_required_frames": missing_count,
                "extra_existing_frames": extra_count,
                "complete_for_study": (
                    missing_count == 0
                ),
            }
        )

        for frame in sorted(
            missing_frames
        ):

            missing_frame_rows.append(
                {
                    "proposed_split": split_name,
                    "source_split": source_split,
                    "date": date,
                    "drive": drive,
                    "frame": frame,
                    "camera": EXPECTED_CAMERA,
                    "expected_rgb_path": str(
                        rgb_dir
                        / f"{frame}.png"
                    ),
                }
            )

    # -----------------------------------------------------
    # Save reports
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    drive_report_fields = [
        "proposed_split",
        "source_split",
        "date",
        "drive",
        "camera",
        "required_frames",
        "raw_rgb_directory",
        "raw_rgb_directory_exists",
        "existing_png_files",
        "matched_required_frames",
        "missing_required_frames",
        "extra_existing_frames",
        "complete_for_study",
    ]

    with DRIVE_REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=drive_report_fields,
        )

        writer.writeheader()

        writer.writerows(
            drive_report_rows
        )

    missing_frame_fields = [
        "proposed_split",
        "source_split",
        "date",
        "drive",
        "frame",
        "camera",
        "expected_rgb_path",
    ]

    with MISSING_FRAME_REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=missing_frame_fields,
        )

        writer.writeheader()

        writer.writerows(
            missing_frame_rows
        )

    # -----------------------------------------------------
    # Print corpus requirements
    # -----------------------------------------------------

    dates = sorted(
        {
            drive[:10]
            for drive in manifest_drives
        }
    )

    print()
    print("Frozen study RGB requirement")
    print("-" * 82)

    print(
        f"Required camera                  : "
        f"{EXPECTED_CAMERA}"
    )

    print(
        f"Required recording dates         : "
        f"{len(dates)}"
    )

    for date in dates:

        date_drives = [
            drive
            for drive in manifest_drives
            if drive.startswith(
                date
            )
        ]

        date_frames = sum(
            len(
                required_frames_by_drive[
                    drive
                ]
            )
            for drive in date_drives
        )

        print(
            f"  {date}: "
            f"{len(date_drives):3d} drives, "
            f"{date_frames:6,d} required frames"
        )

    print()
    print(
        f"Required drives                  : "
        f"{len(manifest_drives):,}"
    )

    print(
        f"Required RGB frames              : "
        f"{len(manifest_rows):,}"
    )

    print(
        f"Planned raw RGB root             : "
        f"{RAW_RGB_ROOT}"
    )

    print(
        f"Raw RGB root currently exists    : "
        f"{RAW_RGB_ROOT.exists()}"
    )

    # -----------------------------------------------------
    # Split-specific audit
    # -----------------------------------------------------

    print()
    print("=" * 82)
    print("RGB availability by frozen partition")
    print("=" * 82)

    for split_name in (
        "TRAIN",
        "DEV",
        "CALIBRATION",
        "LOCKED_TEST",
    ):

        required = (
            split_required_counts[
                split_name
            ]
        )

        matched = (
            split_matched_counts[
                split_name
            ]
        )

        missing = (
            split_missing_counts[
                split_name
            ]
        )

        coverage = (
            100.0
            * matched
            / required
            if required > 0
            else 0.0
        )

        print(
            f"{split_name:<13}"
            f"required={required:6,d}  "
            f"matched={matched:6,d}  "
            f"missing={missing:6,d}  "
            f"coverage={coverage:6.2f}%"
        )

    # -----------------------------------------------------
    # Overall audit
    # -----------------------------------------------------

    complete_drive_count = sum(
        bool(
            row[
                "complete_for_study"
            ]
        )
        for row in drive_report_rows
    )

    incomplete_drive_count = (
        len(drive_report_rows)
        - complete_drive_count
    )

    overall_coverage = (
        100.0
        * total_matched_required
        / EXPECTED_TOTAL_FRAMES
    )

    print()
    print("=" * 82)
    print("Overall raw RGB audit")
    print("=" * 82)

    print(
        f"Existing PNG files in expected dirs : "
        f"{total_existing_png_files:,}"
    )

    print(
        f"Matched required RGB frames          : "
        f"{total_matched_required:,}"
    )

    print(
        f"Missing required RGB frames          : "
        f"{total_missing_required:,}"
    )

    print(
        f"Complete required drives             : "
        f"{complete_drive_count:,}"
    )

    print(
        f"Incomplete required drives           : "
        f"{incomplete_drive_count:,}"
    )

    print(
        f"Required-frame RGB coverage           : "
        f"{overall_coverage:.2f}%"
    )

    # -----------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------

    print()
    print("=" * 82)
    print("Integrity checks")
    print("=" * 82)

    checks = {
        "Manifest contains exactly 46,375 rows":
            manifest_count_correct,

        "Manifest contains exactly 151 drives":
            drive_count_correct,

        "Frozen and manifest drive sets are identical":
            frozen_and_manifest_drives_match,

        "Manifest contains image_02 only":
            only_expected_camera,

        "Manifest has no duplicate sample identity":
            no_duplicate_manifest_identity,

        "Frozen split has no duplicate drive row":
            no_duplicate_split_drive,

        "Per-drive required counts match frozen split":
            per_drive_counts_match,
    }

    for label, passed in checks.items():

        print_check(
            label,
            passed,
        )

    all_integrity_checks_pass = all(
        checks.values()
    )

    print()
    print("Reports")
    print("-" * 82)

    print(
        f"Drive audit   : {DRIVE_REPORT_PATH}"
    )

    print(
        f"Missing frames: {MISSING_FRAME_REPORT_PATH}"
    )

    print()
    print("=" * 82)

    if all_integrity_checks_pass:

        print(
            "PASS: The exact raw RGB acquisition requirement "
            "has been derived successfully."
        )

        if total_missing_required == 0:

            print(
                "All required image_02 RGB frames are already "
                "available."
            )

        else:

            print(
                f"{total_missing_required:,} required RGB frames "
                "are still missing."
            )

            print(
                "The next step is to plan/download only the raw "
                "KITTI sequences needed by these 151 drives."
            )

    else:

        print(
            "FAIL: Do not download or match RGB until the "
            "integrity failure has been investigated."
        )

    print("=" * 82)


if __name__ == "__main__":
    main()
