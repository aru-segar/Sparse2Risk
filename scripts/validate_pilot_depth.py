from pathlib import Path
import sys

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Allow this script to import our local src package
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))


from sparse2risk.data.depth_io import (  # noqa: E402
    KITTI_DEPTH_SCALE,
    load_kitti_depth,
    valid_depth_mask,
)


# ---------------------------------------------------------
# Pilot dataset
# ---------------------------------------------------------

PILOT_ROOT = Path(
    r"C:\FYP\dataset\depth_selection\val_selection_cropped"
)

RGB_DIR = PILOT_ROOT / "image"
LIDAR_DIR = PILOT_ROOT / "velodyne_raw"
REFERENCE_DIR = PILOT_ROOT / "groundtruth_depth"


def calculate_statistics(depth: np.ndarray) -> dict:
    """
    Calculate statistics over valid positive depth pixels.
    """

    mask = valid_depth_mask(depth)
    values = depth[mask]

    total_pixels = depth.size
    valid_pixels = values.size

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "coverage_percent": 0.0,
            "min_depth": float("nan"),
            "max_depth": float("nan"),
            "mean_depth": float("nan"),
        }

    return {
        "valid_pixels": int(valid_pixels),
        "coverage_percent": float(
            100.0 * valid_pixels / total_pixels
        ),
        "min_depth": float(values.min()),
        "max_depth": float(values.max()),
        "mean_depth": float(values.mean()),
    }


def inspect_raw_png(path: Path) -> dict:
    """
    Inspect the PNG before converting stored values to metres.
    """

    with Image.open(path) as image:
        mode = image.mode
        raw = np.array(image)

    return {
        "mode": mode,
        "dtype": str(raw.dtype),
        "shape": raw.shape,
        "min_raw": int(raw.min()),
        "max_raw": int(raw.max()),
    }


def main():

    print("=" * 72)
    print("Sparse2Risk - KITTI Pilot Depth Validation")
    print("=" * 72)

    lidar_files = sorted(LIDAR_DIR.glob("*.png"))

    if len(lidar_files) == 0:
        raise RuntimeError(
            f"No LiDAR PNG files found in {LIDAR_DIR}"
        )

    # -----------------------------------------------------
    # Select one pilot sample
    # -----------------------------------------------------

    lidar_path = lidar_files[0]

    rgb_name = lidar_path.name.replace(
        "_velodyne_raw_",
        "_image_",
    )

    reference_name = lidar_path.name.replace(
        "_velodyne_raw_",
        "_groundtruth_depth_",
    )

    rgb_path = RGB_DIR / rgb_name
    reference_path = REFERENCE_DIR / reference_name

    if not rgb_path.exists():
        raise FileNotFoundError(
            f"Matching RGB file was not found: {rgb_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            "Matching reference depth file was not found: "
            f"{reference_path}"
        )

    print()
    print("Matched sample")
    print("-" * 72)

    print(f"RGB       : {rgb_path.name}")
    print(f"LiDAR     : {lidar_path.name}")
    print(f"Reference : {reference_path.name}")

    # -----------------------------------------------------
    # Inspect raw PNG encoding
    # -----------------------------------------------------

    lidar_raw = inspect_raw_png(lidar_path)
    reference_raw = inspect_raw_png(reference_path)

    print()
    print("Raw PNG inspection")
    print("-" * 72)

    print("Current LiDAR:")
    print(f"  Pillow mode : {lidar_raw['mode']}")
    print(f"  NumPy dtype : {lidar_raw['dtype']}")
    print(f"  Raw shape   : {lidar_raw['shape']}")
    print(f"  Raw minimum : {lidar_raw['min_raw']}")
    print(f"  Raw maximum : {lidar_raw['max_raw']}")

    print()
    print("Reference:")
    print(f"  Pillow mode : {reference_raw['mode']}")
    print(f"  NumPy dtype : {reference_raw['dtype']}")
    print(f"  Raw shape   : {reference_raw['shape']}")
    print(f"  Raw minimum : {reference_raw['min_raw']}")
    print(f"  Raw maximum : {reference_raw['max_raw']}")

    print()
    print(
        f"KITTI conversion scale: "
        f"stored value / {KITTI_DEPTH_SCALE}"
    )

    # -----------------------------------------------------
    # Load RGB and depths
    # -----------------------------------------------------

    with Image.open(rgb_path) as image:
        rgb = np.array(image)

    lidar_depth = load_kitti_depth(lidar_path)
    reference_depth = load_kitti_depth(reference_path)

    print()
    print("Spatial dimensions")
    print("-" * 72)

    print(f"RGB       : {rgb.shape}")
    print(f"LiDAR     : {lidar_depth.shape}")
    print(f"Reference : {reference_depth.shape}")

    spatial_shapes_match = (
        rgb.shape[:2]
        == lidar_depth.shape
        == reference_depth.shape
    )

    print(
        f"Spatial dimensions match: "
        f"{spatial_shapes_match}"
    )

    # -----------------------------------------------------
    # Depth statistics
    # -----------------------------------------------------

    lidar_stats = calculate_statistics(lidar_depth)
    reference_stats = calculate_statistics(reference_depth)

    print()
    print("Current LiDAR D_L")
    print("-" * 72)

    print(
        f"Valid pixels     : "
        f"{lidar_stats['valid_pixels']:,}"
    )
    print(
        f"Valid coverage   : "
        f"{lidar_stats['coverage_percent']:.2f}%"
    )
    print(
        f"Minimum depth    : "
        f"{lidar_stats['min_depth']:.3f} m"
    )
    print(
        f"Maximum depth    : "
        f"{lidar_stats['max_depth']:.3f} m"
    )
    print(
        f"Mean valid depth : "
        f"{lidar_stats['mean_depth']:.3f} m"
    )

    print()
    print("Accumulated reference D_R")
    print("-" * 72)

    print(
        f"Valid pixels     : "
        f"{reference_stats['valid_pixels']:,}"
    )
    print(
        f"Valid coverage   : "
        f"{reference_stats['coverage_percent']:.2f}%"
    )
    print(
        f"Minimum depth    : "
        f"{reference_stats['min_depth']:.3f} m"
    )
    print(
        f"Maximum depth    : "
        f"{reference_stats['max_depth']:.3f} m"
    )
    print(
        f"Mean valid depth : "
        f"{reference_stats['mean_depth']:.3f} m"
    )

    # -----------------------------------------------------
    # 80 m experimental masks
    # -----------------------------------------------------

    lidar_mask_80 = valid_depth_mask(
        lidar_depth,
        max_depth=80.0,
    )

    reference_mask_80 = valid_depth_mask(
        reference_depth,
        max_depth=80.0,
    )

    print()
    print("0-80 m experimental range")
    print("-" * 72)

    print(
        f"LiDAR valid <= 80 m     : "
        f"{lidar_mask_80.sum():,}"
    )

    print(
        f"Reference valid <= 80 m : "
        f"{reference_mask_80.sum():,}"
    )

    # -----------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------

    lidar_has_values = (
        lidar_stats["valid_pixels"] > 0
    )

    reference_has_values = (
        reference_stats["valid_pixels"] > 0
    )

    reference_is_denser = (
        reference_stats["valid_pixels"]
        > lidar_stats["valid_pixels"]
    )

    sensible_depth_scale = (
        lidar_stats["max_depth"] < 200.0
        and reference_stats["max_depth"] < 200.0
    )

    print()
    print("Sanity checks")
    print("-" * 72)

    print(
        f"LiDAR has valid depth       : "
        f"{lidar_has_values}"
    )

    print(
        f"Reference has valid depth   : "
        f"{reference_has_values}"
    )

    print(
        f"Reference denser than LiDAR : "
        f"{reference_is_denser}"
    )

    print(
        f"Depth scale looks sensible  : "
        f"{sensible_depth_scale}"
    )

    all_checks_pass = (
        spatial_shapes_match
        and lidar_has_values
        and reference_has_values
        and reference_is_denser
        and sensible_depth_scale
    )

    print()
    print("-" * 72)

    if all_checks_pass:
        print(
            "PASS: First KITTI RGB/LiDAR/reference "
            "sample passed depth validation."
        )
    else:
        print(
            "FAIL: Sample requires further investigation."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()