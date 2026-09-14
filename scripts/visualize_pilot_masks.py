from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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

RGB_DIR = PILOT_ROOT / "image"
LIDAR_DIR = PILOT_ROOT / "velodyne_raw"
REFERENCE_DIR = PILOT_ROOT / "groundtruth_depth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

MAX_DEPTH = 80.0
EXCLUSION_RADIUS = 2

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_rgb(path: Path) -> np.ndarray:
    """
    Load an RGB image as a NumPy array.
    """

    with Image.open(path) as image:
        return np.array(
            image.convert("RGB")
        )

def masked_depth(
    depth: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ma.MaskedArray:
    """
    Hide invalid depth pixels for visualization.
    """

    return np.ma.masked_where(
        ~valid_mask,
        depth,
    )

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 72)
    print("Sparse2Risk - Pilot Mask Visual Validation")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Select the first image_02 sample
    # -----------------------------------------------------

    lidar_files = sorted(
        LIDAR_DIR.glob("*_image_02.png")
    )

    if not lidar_files:
        raise RuntimeError(
            "No image_02 LiDAR files were found."
        )

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
            f"Missing RGB file: {rgb_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing reference file: {reference_path}"
        )

    # -----------------------------------------------------
    # Load sample
    # -----------------------------------------------------

    rgb = load_rgb(
        rgb_path
    )

    lidar_depth = load_kitti_depth(
        lidar_path
    )

    reference_depth = load_kitti_depth(
        reference_path
    )

    # -----------------------------------------------------
    # Build masks
    # -----------------------------------------------------

    masks = build_basic_masks(
        lidar_depth=lidar_depth,
        reference_depth=reference_depth,
        max_depth=MAX_DEPTH,
    )

    strict_non_current_reference = (
        build_strict_non_current_reference(
            current_lidar_valid=(
                masks["current_lidar_valid"]
            ),
            reference_valid=(
                masks["reference_valid"]
            ),
            exclusion_radius=EXCLUSION_RADIUS,
        )
    )

    # -----------------------------------------------------
    # Basic integrity check
    # -----------------------------------------------------

    if not (
        rgb.shape[:2]
        == lidar_depth.shape
        == reference_depth.shape
    ):
        raise ValueError(
            "RGB, LiDAR and reference spatial "
            "dimensions do not match."
        )

    # -----------------------------------------------------
    # Print sample information
    # -----------------------------------------------------

    print()
    print("Sample")
    print("-" * 72)

    print(f"RGB       : {rgb_path.name}")
    print(f"LiDAR     : {lidar_path.name}")
    print(f"Reference : {reference_path.name}")

    print()
    print("Mask counts")
    print("-" * 72)

    print(
        f"Current LiDAR valid       : "
        f"{masks['current_lidar_valid'].sum():,}"
    )

    print(
        f"Reference valid           : "
        f"{masks['reference_valid'].sum():,}"
    )

    print(
        f"Non-current reference     : "
        f"{masks['non_current_reference'].sum():,}"
    )

    print(
        f"Strict radius-2 reference : "
        f"{strict_non_current_reference.sum():,}"
    )

    # -----------------------------------------------------
    # Prepare depth visualizations
    # -----------------------------------------------------

    lidar_display = masked_depth(
        lidar_depth,
        masks["current_lidar_valid"],
    )

    reference_display = masked_depth(
        reference_depth,
        masks["reference_valid"],
    )

    # -----------------------------------------------------
    # Create figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 8),
    )

    # -----------------------------------------------------
    # Panel 1 - RGB
    # -----------------------------------------------------

    axes[0, 0].imshow(
        rgb
    )

    axes[0, 0].set_title(
        "RGB image"
    )

    # -----------------------------------------------------
    # Panel 2 - Current LiDAR depth
    # -----------------------------------------------------

    lidar_plot = axes[0, 1].imshow(
        lidar_display,
        vmin=0,
        vmax=MAX_DEPTH,
        cmap="viridis",
    )

    axes[0, 1].set_title(
        "Current sparse LiDAR depth"
    )

    fig.colorbar(
        lidar_plot,
        ax=axes[0, 1],
        fraction=0.046,
        pad=0.04,
        label="Depth (m)",
    )

    # -----------------------------------------------------
    # Panel 3 - Accumulated reference depth
    # -----------------------------------------------------

    reference_plot = axes[0, 2].imshow(
        reference_display,
        vmin=0,
        vmax=MAX_DEPTH,
        cmap="viridis",
    )

    axes[0, 2].set_title(
        "Accumulated reference depth"
    )

    fig.colorbar(
        reference_plot,
        ax=axes[0, 2],
        fraction=0.046,
        pad=0.04,
        label="Depth (m)",
    )

    # -----------------------------------------------------
    # Panel 4 - Current LiDAR mask
    # -----------------------------------------------------

    axes[1, 0].imshow(
        masks["current_lidar_valid"],
        cmap="gray",
    )

    axes[1, 0].set_title(
        "Current LiDAR valid pixels"
    )

    # -----------------------------------------------------
    # Panel 5 - Ordinary non-current reference
    # -----------------------------------------------------

    axes[1, 1].imshow(
        rgb
    )

    axes[1, 1].imshow(
        masks["non_current_reference"],
        cmap="autumn",
        alpha=0.45,
    )

    axes[1, 1].set_title(
        "Non-current reference (radius 0)"
    )

    # -----------------------------------------------------
    # Panel 6 - Strict radius-2 reference
    # -----------------------------------------------------

    axes[1, 2].imshow(
        rgb
    )

    axes[1, 2].imshow(
        strict_non_current_reference,
        cmap="autumn",
        alpha=0.45,
    )

    axes[1, 2].set_title(
        "Strict non-current reference (radius 2)"
    )

    # -----------------------------------------------------
    # Remove axes
    # -----------------------------------------------------

    for axis in axes.flat:
        axis.axis("off")

    fig.suptitle(
        "Sparse2Risk KITTI Pilot Mask Validation",
        fontsize=16,
    )

    fig.tight_layout()

    # -----------------------------------------------------
    # Save figure
    # -----------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "pilot_mask_validation.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("-" * 72)

    print(
        f"Saved figure: {output_path}"
    )

    print(
        "PASS: Pilot mask validation figure "
        "generated successfully."
    )

    print("=" * 72)

if __name__ == "__main__":
    main()