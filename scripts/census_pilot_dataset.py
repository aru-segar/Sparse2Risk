from collections import defaultdict
from pathlib import Path
import sys


# ---------------------------------------------------------
# Allow this script to import our local src package
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))


from sparse2risk.data.depth_io import load_kitti_depth  # noqa: E402
from sparse2risk.data.masks import build_basic_masks  # noqa: E402


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

PILOT_ROOT = Path(
    r"C:\FYP\dataset\depth_selection\val_selection_cropped"
)

LIDAR_DIR = PILOT_ROOT / "velodyne_raw"
REFERENCE_DIR = PILOT_ROOT / "groundtruth_depth"

MAX_DEPTH = 80.0


# ---------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------

def create_empty_stats():
    """
    Create an empty statistics dictionary.

    This dictionary stores totals across many frames.
    """

    return {
        "frames": 0,
        "total_pixels": 0,
        "current_lidar_valid": 0,
        "reference_valid": 0,
        "current_and_reference": 0,
        "non_current_reference": 0,
        "current_lidar_only": 0,
    }


def update_stats(stats: dict, masks: dict):
    """
    Add statistics from one frame to the running totals.
    """

    stats["frames"] += 1

    # Every mask has the same image size.
    total_pixels_in_frame = masks["current_lidar_valid"].size

    stats["total_pixels"] += total_pixels_in_frame

    stats["current_lidar_valid"] += int(
        masks["current_lidar_valid"].sum()
    )

    stats["reference_valid"] += int(
        masks["reference_valid"].sum()
    )

    stats["current_and_reference"] += int(
        masks["current_and_reference"].sum()
    )

    stats["non_current_reference"] += int(
        masks["non_current_reference"].sum()
    )

    stats["current_lidar_only"] += int(
        masks["current_lidar_only"].sum()
    )


def calculate_percentage(
    numerator: int,
    denominator: int,
) -> float:
    """
    Safely calculate a percentage.
    """

    if denominator == 0:
        return float("nan")

    return 100.0 * numerator / denominator


def print_stats(
    title: str,
    stats: dict,
):
    """
    Print statistics for one group of frames.
    """

    total_pixels = stats["total_pixels"]
    reference_pixels = stats["reference_valid"]

    current_lidar_percent = calculate_percentage(
        stats["current_lidar_valid"],
        total_pixels,
    )

    reference_percent = calculate_percentage(
        stats["reference_valid"],
        total_pixels,
    )

    current_and_reference_percent = calculate_percentage(
        stats["current_and_reference"],
        total_pixels,
    )

    non_current_reference_percent = calculate_percentage(
        stats["non_current_reference"],
        total_pixels,
    )

    current_lidar_only_percent = calculate_percentage(
        stats["current_lidar_only"],
        total_pixels,
    )

    reference_also_current_percent = calculate_percentage(
        stats["current_and_reference"],
        reference_pixels,
    )

    reference_non_current_percent = calculate_percentage(
        stats["non_current_reference"],
        reference_pixels,
    )

    print()
    print(title)
    print("-" * 72)

    print(
        f"Frames                         : "
        f"{stats['frames']:,}"
    )

    print(
        f"Total image pixels             : "
        f"{total_pixels:,}"
    )

    print()
    print("Percentage of all image pixels")
    print()

    print(
        f"Current LiDAR valid             : "
        f"{current_lidar_percent:.2f}%"
    )

    print(
        f"Reference valid                 : "
        f"{reference_percent:.2f}%"
    )

    print(
        f"Current + reference             : "
        f"{current_and_reference_percent:.2f}%"
    )

    print(
        f"Non-current reference           : "
        f"{non_current_reference_percent:.2f}%"
    )

    print(
        f"Current LiDAR only              : "
        f"{current_lidar_only_percent:.4f}%"
    )

    print()
    print("Percentage of reference-valid pixels")
    print()

    print(
        f"Reference pixels also current   : "
        f"{reference_also_current_percent:.2f}%"
    )

    print(
        f"Reference pixels non-current    : "
        f"{reference_non_current_percent:.2f}%"
    )


# ---------------------------------------------------------
# Main census
# ---------------------------------------------------------

def main():

    print("=" * 72)
    print("Sparse2Risk - KITTI Pilot Dataset Census")
    print("=" * 72)

    lidar_files = sorted(
        LIDAR_DIR.glob("*.png")
    )

    if len(lidar_files) == 0:
        raise RuntimeError(
            f"No LiDAR files found in {LIDAR_DIR}"
        )

    print()
    print(
        f"LiDAR files discovered : "
        f"{len(lidar_files):,}"
    )

    print(
        f"Maximum experiment depth: "
        f"{MAX_DEPTH:.1f} m"
    )

    # -----------------------------------------------------
    # Statistics for all frames
    # -----------------------------------------------------

    all_stats = create_empty_stats()

    # Separate statistics for image_02 and image_03.
    camera_stats = defaultdict(
        create_empty_stats
    )

    missing_reference_files = []

    processed_frames = 0

    # -----------------------------------------------------
    # Process every pilot frame
    # -----------------------------------------------------

    for lidar_path in lidar_files:

        # Create the matching reference-depth filename.
        reference_name = lidar_path.name.replace(
            "_velodyne_raw_",
            "_groundtruth_depth_",
        )

        reference_path = (
            REFERENCE_DIR / reference_name
        )

        if not reference_path.exists():
            missing_reference_files.append(
                lidar_path.name
            )
            continue

        # -------------------------------------------------
        # Identify which camera produced this frame
        # -------------------------------------------------

        if "_image_02.png" in lidar_path.name:
            camera = "image_02"

        elif "_image_03.png" in lidar_path.name:
            camera = "image_03"

        else:
            raise ValueError(
                "Unable to determine camera from filename: "
                f"{lidar_path.name}"
            )

        # -------------------------------------------------
        # Load the depth maps
        # -------------------------------------------------

        lidar_depth = load_kitti_depth(
            lidar_path
        )

        reference_depth = load_kitti_depth(
            reference_path
        )

        # -------------------------------------------------
        # Build validity masks
        # -------------------------------------------------

        masks = build_basic_masks(
            lidar_depth=lidar_depth,
            reference_depth=reference_depth,
            max_depth=MAX_DEPTH,
        )

        # -------------------------------------------------
        # Update statistics
        # -------------------------------------------------

        update_stats(
            all_stats,
            masks,
        )

        update_stats(
            camera_stats[camera],
            masks,
        )

        processed_frames += 1

        # Print progress every 100 frames.
        if processed_frames % 100 == 0:
            print(
                f"Processed "
                f"{processed_frames:4d} / "
                f"{len(lidar_files)} frames"
            )

    # -----------------------------------------------------
    # Processing summary
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Processing summary")
    print("=" * 72)

    print(
        f"LiDAR files discovered : "
        f"{len(lidar_files):,}"
    )

    print(
        f"Frames processed       : "
        f"{processed_frames:,}"
    )

    print(
        f"Missing references     : "
        f"{len(missing_reference_files):,}"
    )

    # -----------------------------------------------------
    # Overall results
    # -----------------------------------------------------

    print_stats(
        "ALL PILOT CAMERAS",
        all_stats,
    )

    # -----------------------------------------------------
    # Camera-specific results
    # -----------------------------------------------------

    for camera in sorted(camera_stats):

        print_stats(
            camera.upper(),
            camera_stats[camera],
        )

    # -----------------------------------------------------
    # Logical integrity checks
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Logical integrity checks")
    print("=" * 72)

    # Every reference-valid pixel must be either:
    #
    # 1. also current-LiDAR valid
    #
    # OR
    #
    # 2. not current-LiDAR valid.
    #
    reference_partition_is_valid = (
        all_stats["current_and_reference"]
        + all_stats["non_current_reference"]
        == all_stats["reference_valid"]
    )

    print(
        "Current+reference + non-current "
        "reference = all reference pixels : "
        f"{reference_partition_is_valid}"
    )

    # -----------------------------------------------------
    # Frame-count checks
    # -----------------------------------------------------

    expected_frames = 1000

    correct_frame_count = (
        processed_frames == expected_frames
        and len(missing_reference_files) == 0
    )

    print(
        f"Exactly {expected_frames} frames processed   : "
        f"{correct_frame_count}"
    )

    correct_camera_count = (
        camera_stats["image_02"]["frames"] == 500
        and camera_stats["image_03"]["frames"] == 500
    )

    print(
        "500 image_02 + 500 image_03       : "
        f"{correct_camera_count}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()

    if (
        reference_partition_is_valid
        and correct_frame_count
        and correct_camera_count
    ):
        print(
            "PASS: Pilot dataset census and "
            "basic validity masks passed."
        )

    else:
        print(
            "FAIL: Pilot dataset census "
            "requires investigation."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()