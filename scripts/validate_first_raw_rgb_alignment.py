import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# =========================================================
# Sparse2Risk - First Raw RGB / Depth Alignment Validation
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_ROOT),
)

from sparse2risk.data.depth_io import load_kitti_depth  # noqa: E402
from sparse2risk.data.masks import build_basic_masks  # noqa: E402


MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manifests"
    / "kitti_full_depth_manifest.csv"
)

RAW_RGB_ROOT = Path(
    r"C:\FYP\dataset\kitti_raw"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
)

TEST_DRIVE = "2011_09_26_drive_0001_sync"
EXPECTED_CAMERA = "image_02"
MAX_DEPTH = 80.0


def read_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def rgb_path_for(
    drive: str,
    frame: str,
):
    return (
        RAW_RGB_ROOT
        / drive[:10]
        / drive
        / EXPECTED_CAMERA
        / "data"
        / f"{frame}.png"
    )


def load_rgb(path: Path):
    with Image.open(path) as image:
        return np.array(
            image.convert("RGB")
        )


def main():

    print("=" * 82)
    print(
        "Sparse2Risk - First Raw RGB / Depth Alignment Validation"
    )
    print("=" * 82)

    rows = [
        row
        for row in read_manifest()
        if (
            row["drive"] == TEST_DRIVE
            and row["camera"] == EXPECTED_CAMERA
        )
    ]

    rows.sort(
        key=lambda row: row["frame"]
    )

    if not rows:
        raise RuntimeError(
            f"No manifest rows found for {TEST_DRIVE}."
        )

    print()
    print(
        f"Test drive          : {TEST_DRIVE}"
    )

    print(
        f"Manifest frames     : {len(rows):,}"
    )

    missing_rgb = []
    shape_mismatches = []

    rgb_shape_counts = Counter()
    lidar_shape_counts = Counter()
    reference_shape_counts = Counter()

    checked = 0

    for row in rows:

        frame = row["frame"]

        rgb_path = rgb_path_for(
            TEST_DRIVE,
            frame,
        )

        if not rgb_path.exists():
            missing_rgb.append(
                frame
            )
            continue

        rgb = load_rgb(
            rgb_path
        )

        lidar = load_kitti_depth(
            Path(row["lidar_path"])
        )

        reference = load_kitti_depth(
            Path(row["reference_path"])
        )

        rgb_hw = (
            rgb.shape[0],
            rgb.shape[1],
        )

        lidar_hw = (
            lidar.shape[0],
            lidar.shape[1],
        )

        reference_hw = (
            reference.shape[0],
            reference.shape[1],
        )

        rgb_shape_counts[
            rgb_hw
        ] += 1

        lidar_shape_counts[
            lidar_hw
        ] += 1

        reference_shape_counts[
            reference_hw
        ] += 1

        if not (
            rgb_hw
            == lidar_hw
            == reference_hw
        ):
            shape_mismatches.append(
                (
                    frame,
                    rgb_hw,
                    lidar_hw,
                    reference_hw,
                )
            )

        checked += 1

    print()
    print("Drive-wide shape validation")
    print("-" * 82)

    print(
        f"RGB frames found    : {checked:,}/{len(rows):,}"
    )

    print(
        f"Missing RGB frames  : {len(missing_rgb):,}"
    )

    print(
        f"Shape mismatches    : {len(shape_mismatches):,}"
    )

    print()
    print(
        f"RGB shapes          : {dict(rgb_shape_counts)}"
    )

    print(
        f"LiDAR shapes        : {dict(lidar_shape_counts)}"
    )

    print(
        f"Reference shapes    : {dict(reference_shape_counts)}"
    )

    if missing_rgb:
        print()
        print(
            "First missing RGB frames:"
        )

        for frame in missing_rgb[:10]:
            print(
                f"  {frame}"
            )

    if shape_mismatches:
        print()
        print(
            "First shape mismatches:"
        )

        for mismatch in shape_mismatches[:10]:
            print(
                f"  frame={mismatch[0]} "
                f"rgb={mismatch[1]} "
                f"lidar={mismatch[2]} "
                f"reference={mismatch[3]}"
            )

    all_rgb_present = (
        len(missing_rgb) == 0
    )

    all_shapes_match = (
        len(shape_mismatches) == 0
    )

    # -----------------------------------------------------
    # Visual check using the first available sample
    # -----------------------------------------------------

    visual_row = None

    for row in rows:

        if rgb_path_for(
            TEST_DRIVE,
            row["frame"],
        ).exists():
            visual_row = row
            break

    if visual_row is None:
        raise RuntimeError(
            "No RGB sample is available for visual validation."
        )

    frame = visual_row["frame"]

    rgb_path = rgb_path_for(
        TEST_DRIVE,
        frame,
    )

    rgb = load_rgb(
        rgb_path
    )

    lidar = load_kitti_depth(
        Path(
            visual_row["lidar_path"]
        )
    )

    reference = load_kitti_depth(
        Path(
            visual_row["reference_path"]
        )
    )

    masks = build_basic_masks(
        lidar_depth=lidar,
        reference_depth=reference,
        max_depth=MAX_DEPTH,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "full_raw_rgb_depth_alignment.png"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    axes[0].imshow(
        rgb
    )

    axes[0].set_title(
        f"Raw RGB\n{TEST_DRIVE} / {frame}"
    )

    axes[1].imshow(
        rgb
    )

    lidar_overlay = np.ma.masked_where(
        ~masks["current_lidar_valid"],
        lidar,
    )

    axes[1].imshow(
        lidar_overlay,
        cmap="viridis",
        alpha=0.75,
        vmin=0,
        vmax=MAX_DEPTH,
    )

    axes[1].set_title(
        "RGB + current LiDAR"
    )

    axes[2].imshow(
        rgb
    )

    reference_overlay = np.ma.masked_where(
        ~masks["reference_valid"],
        reference,
    )

    axes[2].imshow(
        reference_overlay,
        cmap="viridis",
        alpha=0.60,
        vmin=0,
        vmax=MAX_DEPTH,
    )

    axes[2].set_title(
        "RGB + accumulated reference"
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        "Sparse2Risk Full-Corpus Raw RGB Alignment Check",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Visual validation")
    print("-" * 82)

    print(
        f"Sample frame        : {frame}"
    )

    print(
        f"Figure saved to     : {output_path}"
    )

    print()
    print("=" * 82)
    print("Integrity checks")
    print("=" * 82)

    print(
        f"All 98 RGB frames present          : "
        f"{all_rgb_present}"
    )

    print(
        f"RGB/LiDAR/reference shapes match   : "
        f"{all_shapes_match}"
    )

    print()

    if (
        all_rgb_present
        and all_shapes_match
    ):
        print(
            "PASS: First raw KITTI drive is structurally "
            "aligned with the full depth corpus."
        )

        print(
            "Inspect the saved overlay before enabling "
            "the full RGB acquisition."
        )

    else:
        print(
            "FAIL: Do not start the full RGB acquisition "
            "until the mismatch is investigated."
        )

        sys.exit(1)

    print("=" * 82)


if __name__ == "__main__":
    main()
