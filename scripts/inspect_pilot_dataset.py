from pathlib import Path
from collections import Counter
import re


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PILOT_ROOT = Path(
    r"C:\FYP\dataset\depth_selection\val_selection_cropped"
)

RGB_DIR = PILOT_ROOT / "image"
LIDAR_DIR = PILOT_ROOT / "velodyne_raw"
REFERENCE_DIR = PILOT_ROOT / "groundtruth_depth"
INTRINSICS_DIR = PILOT_ROOT / "intrinsics"


# ---------------------------------------------------------
# KITTI filename pattern
# ---------------------------------------------------------

# Examples:
#
# 2011_09_26_drive_0002_sync_image_0000000005_image_02.png
# 2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png
# 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png
#
# We care about:
#   drive  = 2011_09_26_drive_0002_sync
#   frame  = 0000000005
#   camera = image_02

FILE_PATTERN = re.compile(
    r"^(?P<drive>\d{4}_\d{2}_\d{2}_drive_\d{4}_sync)"
    r"_(?:image|velodyne_raw|groundtruth_depth)"
    r"_(?P<frame>\d{10})"
    r"_(?P<camera>image_0[23])"
    r"\.png$"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def list_files(folder: Path, suffix: str):
    """Return sorted files with a particular suffix."""
    if not folder.exists():
        return []

    return sorted(folder.glob(f"*{suffix}"))


def print_folder_status(name: str, folder: Path):
    """Print whether an expected folder exists."""
    status = "FOUND" if folder.exists() else "MISSING"

    print(f"{name:<20}: {status}")
    print(f"  Path: {folder}")


def extract_sample_id(filename: str):
    """
    Convert a KITTI filename into a common sample identity.

    Returns:
        (drive, frame, camera)

    Example:
        (
            "2011_09_26_drive_0002_sync",
            "0000000005",
            "image_02"
        )
    """
    match = FILE_PATTERN.match(filename)

    if match is None:
        return None

    drive = match.group("drive")
    frame = match.group("frame")
    camera = match.group("camera")

    return drive, frame, camera


def build_sample_map(files):
    """
    Build a dictionary:

        sample_id -> file path
    """
    sample_map = {}
    unparsed_files = []

    for file in files:
        sample_id = extract_sample_id(file.name)

        if sample_id is None:
            unparsed_files.append(file)
            continue

        if sample_id in sample_map:
            raise ValueError(
                f"Duplicate sample ID detected: {sample_id}"
            )

        sample_map[sample_id] = file

    return sample_map, unparsed_files


def camera_counts(sample_ids):
    """Count samples belonging to image_02 and image_03."""
    cameras = [sample_id[2] for sample_id in sample_ids]

    return Counter(cameras)


# ---------------------------------------------------------
# Main inspection
# ---------------------------------------------------------

def main():

    print("=" * 72)
    print("Sparse2Risk - KITTI Pilot Dataset Inspection")
    print("=" * 72)

    print()
    print("Pilot dataset root:")
    print(PILOT_ROOT)

    print()
    print(f"Dataset root exists: {PILOT_ROOT.exists()}")

    print()
    print("-" * 72)
    print("Required folders")
    print("-" * 72)

    print_folder_status("RGB", RGB_DIR)
    print_folder_status("Current LiDAR", LIDAR_DIR)
    print_folder_status("Reference depth", REFERENCE_DIR)
    print_folder_status("Intrinsics", INTRINSICS_DIR)

    rgb_files = list_files(RGB_DIR, ".png")
    lidar_files = list_files(LIDAR_DIR, ".png")
    reference_files = list_files(REFERENCE_DIR, ".png")
    intrinsics_files = list_files(INTRINSICS_DIR, ".txt")

    print()
    print("-" * 72)
    print("File counts")
    print("-" * 72)

    print(f"RGB images             : {len(rgb_files)}")
    print(f"Current LiDAR maps     : {len(lidar_files)}")
    print(f"Reference depth maps   : {len(reference_files)}")
    print(f"Intrinsics files       : {len(intrinsics_files)}")

    # -----------------------------------------------------
    # Parse RGB, LiDAR and reference files
    # -----------------------------------------------------

    rgb_map, bad_rgb = build_sample_map(rgb_files)
    lidar_map, bad_lidar = build_sample_map(lidar_files)
    reference_map, bad_reference = build_sample_map(
        reference_files
    )

    rgb_ids = set(rgb_map)
    lidar_ids = set(lidar_map)
    reference_ids = set(reference_map)

    matched_ids = (
        rgb_ids
        & lidar_ids
        & reference_ids
    )

    print()
    print("-" * 72)
    print("Filename parsing")
    print("-" * 72)

    print(f"Parsed RGB files           : {len(rgb_map)}")
    print(f"Parsed LiDAR files         : {len(lidar_map)}")
    print(f"Parsed reference files     : {len(reference_map)}")

    print(f"Unparsed RGB files         : {len(bad_rgb)}")
    print(f"Unparsed LiDAR files       : {len(bad_lidar)}")
    print(
        f"Unparsed reference files   : "
        f"{len(bad_reference)}"
    )

    # -----------------------------------------------------
    # Matching
    # -----------------------------------------------------

    print()
    print("-" * 72)
    print("RGB / LiDAR / reference matching")
    print("-" * 72)

    print(
        f"Samples present in all three sets : "
        f"{len(matched_ids)}"
    )

    print(
        f"RGB samples missing LiDAR         : "
        f"{len(rgb_ids - lidar_ids)}"
    )

    print(
        f"RGB samples missing reference     : "
        f"{len(rgb_ids - reference_ids)}"
    )

    print(
        f"LiDAR samples missing RGB         : "
        f"{len(lidar_ids - rgb_ids)}"
    )

    print(
        f"Reference samples missing RGB     : "
        f"{len(reference_ids - rgb_ids)}"
    )

    # -----------------------------------------------------
    # Camera distribution
    # -----------------------------------------------------

    counts = camera_counts(matched_ids)

    print()
    print("-" * 72)
    print("Camera distribution")
    print("-" * 72)

    print(
        f"image_02 matched samples : "
        f"{counts.get('image_02', 0)}"
    )

    print(
        f"image_03 matched samples : "
        f"{counts.get('image_03', 0)}"
    )

    # -----------------------------------------------------
    # Example matches
    # -----------------------------------------------------

    print()
    print("-" * 72)
    print("Example matched samples")
    print("-" * 72)

    for sample_id in sorted(matched_ids)[:5]:

        drive, frame, camera = sample_id

        print()
        print(f"Drive  : {drive}")
        print(f"Frame  : {frame}")
        print(f"Camera : {camera}")

        print(f"RGB    : {rgb_map[sample_id].name}")
        print(f"LiDAR  : {lidar_map[sample_id].name}")
        print(
            f"Ref    : "
            f"{reference_map[sample_id].name}"
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("-" * 72)
    print("Final integrity result")
    print("-" * 72)

    expected_file_count = 1000

    basic_structure_valid = (
        PILOT_ROOT.exists()
        and RGB_DIR.exists()
        and LIDAR_DIR.exists()
        and REFERENCE_DIR.exists()
    )

    counts_valid = (
        len(rgb_files) == expected_file_count
        and len(lidar_files) == expected_file_count
        and len(reference_files) == expected_file_count
    )

    parsing_valid = (
        len(bad_rgb) == 0
        and len(bad_lidar) == 0
        and len(bad_reference) == 0
    )

    matching_valid = (
        len(matched_ids) == expected_file_count
        and rgb_ids == lidar_ids
        and rgb_ids == reference_ids
    )

    if (
        basic_structure_valid
        and counts_valid
        and parsing_valid
        and matching_valid
    ):
        print(
            "PASS: All 1000 pilot RGB, LiDAR and "
            "reference samples match correctly."
        )
    else:
        print(
            "FAIL: The pilot dataset requires "
            "further investigation."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()