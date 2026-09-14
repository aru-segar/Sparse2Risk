from collections import defaultdict
from pathlib import Path
import sys

# ---------------------------------------------------------
# Import local Sparse2Risk package
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from sparse2risk.data.depth_io import load_kitti_depth  
from sparse2risk.data.masks import (  
    build_basic_masks,
    build_strict_non_current_reference,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PILOT_ROOT = Path(
    r"C:\FYP\dataset\depth_selection\val_selection_cropped"
)

LIDAR_DIR = PILOT_ROOT / "velodyne_raw"
REFERENCE_DIR = PILOT_ROOT / "groundtruth_depth"

MAX_DEPTH = 80.0

EXCLUSION_RADII = (0, 1, 2, 3)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def create_empty_stats():
    return {
        "frames": 0,
        "total_pixels": 0,
        "reference_valid": 0,
        "non_current_reference": 0,
        "radius_0": 0,
        "radius_1": 0,
        "radius_2": 0,
        "radius_3": 0,
    }

def percentage(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return float("nan")

    return 100.0 * numerator / denominator

def update_stats(
    stats: dict,
    masks: dict,
    radius_masks: dict,
):
    
    stats["frames"] += 1

    stats["total_pixels"] += (
        masks["reference_valid"].size
    )

    stats["reference_valid"] += int(
        masks["reference_valid"].sum()
    )

    stats["non_current_reference"] += int(
        masks["non_current_reference"].sum()
    )

    for radius in EXCLUSION_RADII:
        stats[f"radius_{radius}"] += int(
            radius_masks[radius].sum()
        )

def print_stats(
    title: str,
    stats: dict,
):
    """
    Print exclusion-radius statistics for one group of frames.
    """

    reference_pixels = stats["reference_valid"]
    all_pixels = stats["total_pixels"]

    reference_percent_all = percentage(
        reference_pixels,
        all_pixels,
    )

    print()
    print(title)
    print("-" * 72)

    print(
        f"Frames                              : "
        f"{stats['frames']:,}"
    )

    print(
        f"Reference-valid pixels              : "
        f"{reference_pixels:,}"
    )

    print()

    print(
        f"Reference valid (% all pixels)      : "
        f"{reference_percent_all:.2f}%"
    )

    print()

    for radius in EXCLUSION_RADII:

        count = stats[f"radius_{radius}"]

        percent_all = percentage(
            count,
            all_pixels,
        )

        percent_reference = percentage(
            count,
            reference_pixels,
        )

        print(
            f"Non-current reference radius {radius}     : "
            f"{count:,}"
        )

        print(
            f"  % of all image pixels             : "
            f"{percent_all:.2f}%"
        )

        print(
            f"  % of reference-valid pixels       : "
            f"{percent_reference:.2f}%"
        )

        print()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 72)
    print("Sparse2Risk - Non-Current Reference Radius Census")
    print("=" * 72)

    lidar_files = sorted(
        LIDAR_DIR.glob("*.png")
    )

    if not lidar_files:
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

    print(
        f"Exclusion radii: "
        f"{EXCLUSION_RADII}"
    )

    all_stats = create_empty_stats()

    camera_stats = defaultdict(
        create_empty_stats
    )

    processed_frames = 0

    radius_zero_matches_basic = True
    masks_are_nested = True

    for lidar_path in lidar_files:

        reference_name = lidar_path.name.replace(
            "_velodyne_raw_",
            "_groundtruth_depth_",
        )

        reference_path = (
            REFERENCE_DIR / reference_name
        )

        if not reference_path.exists():
            raise FileNotFoundError(
                f"Missing reference file: {reference_path}"
            )

        if "_image_02.png" in lidar_path.name:
            camera = "image_02"

        elif "_image_03.png" in lidar_path.name:
            camera = "image_03"

        else:
            raise ValueError(
                "Unable to identify camera from filename: "
                f"{lidar_path.name}"
            )

        lidar_depth = load_kitti_depth(
            lidar_path
        )

        reference_depth = load_kitti_depth(
            reference_path
        )

        masks = build_basic_masks(
            lidar_depth=lidar_depth,
            reference_depth=reference_depth,
            max_depth=MAX_DEPTH,
        )

        radius_masks = {}

        for radius in EXCLUSION_RADII:

            radius_masks[radius] = (
                build_strict_non_current_reference(
                    current_lidar_valid=(
                        masks["current_lidar_valid"]
                    ),
                    reference_valid=(
                        masks["reference_valid"]
                    ),
                    exclusion_radius=radius,
                )
            )

        # Radius 0 must be exactly equivalent to the
        # ordinary non-current reference mask.
        if not (
            radius_masks[0]
            == masks["non_current_reference"]
        ).all():
            radius_zero_matches_basic = False

        # Increasing the radius must never ADD pixels.
        if not (
            (
                radius_masks[1]
                <= radius_masks[0]
            ).all()
            and (
                radius_masks[2]
                <= radius_masks[1]
            ).all()
            and (
                radius_masks[3]
                <= radius_masks[2]
            ).all()
        ):
            masks_are_nested = False

        update_stats(
            all_stats,
            masks,
            radius_masks,
        )

        update_stats(
            camera_stats[camera],
            masks,
            radius_masks,
        )

        processed_frames += 1

        if processed_frames % 100 == 0:

            print(
                f"Processed "
                f"{processed_frames:4d} / "
                f"{len(lidar_files)} frames"
            )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    print_stats(
        "ALL PILOT CAMERAS",
        all_stats,
    )

    for camera in sorted(camera_stats):

        print_stats(
            camera.upper(),
            camera_stats[camera],
        )

    # -----------------------------------------------------
    # Integrity checks
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("Integrity checks")
    print("=" * 72)

    print(
        "Radius 0 equals ordinary non-current mask : "
        f"{radius_zero_matches_basic}"
    )

    print(
        "Masks shrink as radius increases          : "
        f"{masks_are_nested}"
    )

    correct_frame_count = (
        processed_frames == 1000
    )

    print(
        "Exactly 1000 frames processed             : "
        f"{correct_frame_count}"
    )

    main_mask_has_pixels = (
        all_stats["radius_2"] > 0
    )

    print(
        "Primary radius-2 mask contains pixels      : "
        f"{main_mask_has_pixels}"
    )

    print()

    if (
        radius_zero_matches_basic
        and masks_are_nested
        and correct_frame_count
        and main_mask_has_pixels
    ):
        print(
            "PASS: Strict non-current reference "
            "masks passed all checks."
        )

    else:
        print(
            "FAIL: Radius masks require investigation."
        )

    print("=" * 72)

if __name__ == "__main__":
    main()