import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manifests"
    / "kitti_full_depth_manifest.csv"
)

PILOT_RGB_DIR = Path(
    r"C:\FYP\dataset\depth_selection"
    r"\val_selection_cropped\image"
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_manifest():
    """
    Load the generated full-depth manifest.
    """

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )

def extract_pilot_identity(filename: str):
    """
    Extract drive, frame and camera from a pilot RGB filename.

    Example:
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

    return drive, frame, camera

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 72)
    print("Sparse2Risk - Full KITTI Manifest Analysis")
    print("=" * 72)

    rows = load_manifest()

    print()
    print(f"Manifest rows: {len(rows):,}")

    # -----------------------------------------------------
    # Frames per drive
    # -----------------------------------------------------

    frames_per_drive = Counter(
        row["drive"]
        for row in rows
    )

    source_by_drive = {}

    for row in rows:
        source_by_drive[row["drive"]] = (
            row["source_split"]
        )

    unique_drives = sorted(
        frames_per_drive
    )

    frame_counts = list(
        frames_per_drive.values()
    )

    print()
    print("Drive statistics")
    print("-" * 72)

    print(
        f"Unique drives             : "
        f"{len(unique_drives):,}"
    )

    print(
        f"Minimum frames per drive  : "
        f"{min(frame_counts):,}"
    )

    print(
        f"Maximum frames per drive  : "
        f"{max(frame_counts):,}"
    )

    print(
        f"Mean frames per drive     : "
        f"{mean(frame_counts):.2f}"
    )

    print(
        f"Median frames per drive   : "
        f"{median(frame_counts):.2f}"
    )

    # -----------------------------------------------------
    # Date distribution
    # -----------------------------------------------------

    frames_per_date = Counter()
    drives_per_date = defaultdict(set)

    for row in rows:

        date = row["drive"][:10]

        frames_per_date[date] += 1
        drives_per_date[date].add(
            row["drive"]
        )

    print()
    print("Date distribution")
    print("-" * 72)

    for date in sorted(frames_per_date):

        print(
            f"{date}: "
            f"{len(drives_per_date[date]):3d} drives, "
            f"{frames_per_date[date]:6,d} frames"
        )

    # -----------------------------------------------------
    # Largest and smallest drives
    # -----------------------------------------------------

    sorted_drives = sorted(
        frames_per_drive.items(),
        key=lambda item: item[1],
    )

    print()
    print("10 smallest drives")
    print("-" * 72)

    for drive, count in sorted_drives[:10]:
        print(
            f"{count:5d}  {drive}"
        )

    print()
    print("10 largest drives")
    print("-" * 72)

    for drive, count in reversed(
        sorted_drives[-10:]
    ):
        print(
            f"{count:5d}  {drive}"
        )

    # -----------------------------------------------------
    # Pilot identities
    # -----------------------------------------------------

    pilot_files = sorted(
        PILOT_RGB_DIR.glob("*.png")
    )

    pilot_ids_all = set()
    pilot_ids_image02 = set()
    bad_pilot_names = []

    for path in pilot_files:

        identity = extract_pilot_identity(
            path.name
        )

        if identity is None:
            bad_pilot_names.append(
                path.name
            )
            continue

        pilot_ids_all.add(
            identity
        )

        if identity[2] == "image_02":
            pilot_ids_image02.add(
                identity
            )

    pilot_drives = {
        drive
        for drive, _, _ in pilot_ids_all
    }

    print()
    print("Pilot overlap")
    print("-" * 72)

    print(
        f"Pilot files                 : "
        f"{len(pilot_files):,}"
    )

    print(
        f"Parsed pilot identities     : "
        f"{len(pilot_ids_all):,}"
    )

    print(
        f"Pilot image_02 identities   : "
        f"{len(pilot_ids_image02):,}"
    )

    print(
        f"Unique pilot drives         : "
        f"{len(pilot_drives):,}"
    )

    print(
        f"Unparsed pilot filenames    : "
        f"{len(bad_pilot_names):,}"
    )

    # -----------------------------------------------------
    # Compare pilot with full manifest
    # -----------------------------------------------------

    full_ids = {
        (
            row["drive"],
            row["frame"],
            row["camera"],
        )
        for row in rows
    }

    exact_pilot_overlap = (
        pilot_ids_image02
        & full_ids
    )

    full_drives = set(
        unique_drives
    )

    pilot_drives_in_full = (
        pilot_drives
        & full_drives
    )

    print()
    print(
        f"Exact pilot image_02 samples "
        f"in full manifest : "
        f"{len(exact_pilot_overlap):,}"
    )

    print(
        f"Pilot drives represented "
        f"in full corpus     : "
        f"{len(pilot_drives_in_full):,}"
    )

    # -----------------------------------------------------
    # How much of the full corpus belongs to pilot-seen
    # drives?
    # -----------------------------------------------------

    frames_on_pilot_drives = sum(
        frames_per_drive[drive]
        for drive in pilot_drives_in_full
    )

    frames_on_unseen_drives = (
        len(rows)
        - frames_on_pilot_drives
    )

    print()
    print(
        f"Full frames on pilot-seen drives : "
        f"{frames_on_pilot_drives:,}"
    )

    print(
        f"Full frames on unseen drives     : "
        f"{frames_on_unseen_drives:,}"
    )

    print()
    print("Pilot drives")
    print("-" * 72)

    for drive in sorted(
        pilot_drives_in_full
    ):

        print(
            f"{frames_per_drive[drive]:5d}  "
            f"{drive}  "
            f"[source={source_by_drive[drive]}]"
        )

    # -----------------------------------------------------
    # Check whether all original val drives were pilot-seen
    # -----------------------------------------------------

    original_val_drives = {
        drive
        for drive, source
        in source_by_drive.items()
        if source == "val"
    }

    pilot_equals_original_val = (
        pilot_drives_in_full
        == original_val_drives
    )

    print()
    print("Important split check")
    print("-" * 72)

    print(
        f"Original val drives          : "
        f"{len(original_val_drives):,}"
    )

    print(
        f"Pilot-seen full drives       : "
        f"{len(pilot_drives_in_full):,}"
    )

    print(
        "Pilot drives exactly equal "
        "original val drives : "
        f"{pilot_equals_original_val}"
    )

    # -----------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Integrity checks")
    print("=" * 72)

    all_pilot_names_parsed = (
        len(bad_pilot_names) == 0
    )

    expected_pilot_count = (
        len(pilot_ids_all) == 1000
    )

    expected_image02_count = (
        len(pilot_ids_image02) == 500
    )

    print(
        f"All pilot filenames parsed      : "
        f"{all_pilot_names_parsed}"
    )

    print(
        f"Exactly 1000 pilot identities   : "
        f"{expected_pilot_count}"
    )

    print(
        f"Exactly 500 pilot image_02 IDs  : "
        f"{expected_image02_count}"
    )

    print()
    print(
        "PASS: Full manifest analysis completed."
    )

    print("=" * 72)

if __name__ == "__main__":
    main()