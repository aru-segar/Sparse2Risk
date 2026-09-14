from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from sparse2risk.data.masks import ( 
    calculate_distance_to_current_lidar,
    build_strict_non_current_reference,
)

def main():

    print("=" * 72)
    print("Sparse2Risk - Strict Mask Unit Test")
    print("=" * 72)

    # -----------------------------------------------------
    # Create a simple 7 x 7 artificial image.
    #
    # One current-LiDAR anchor is placed exactly
    # in the centre.
    # -----------------------------------------------------

    current_lidar_valid = np.zeros(
        (7, 7),
        dtype=bool,
    )

    current_lidar_valid[3, 3] = True

    # Pretend that reference depth exists everywhere.
    reference_valid = np.ones(
        (7, 7),
        dtype=bool,
    )

    distance = calculate_distance_to_current_lidar(
        current_lidar_valid
    )

    print()
    print("Distance to centre LiDAR anchor")
    print("-" * 72)

    print(
        np.round(
            distance,
            decimals=2,
        )
    )

    # -----------------------------------------------------
    # Check several known distances.
    # -----------------------------------------------------

    assert distance[3, 3] == 0.0

    # Horizontal/vertical neighbour.
    assert np.isclose(
        distance[3, 4],
        1.0,
    )

    # Diagonal neighbour.
    assert np.isclose(
        distance[4, 4],
        np.sqrt(2),
    )

    # Exactly two pixels away.
    assert np.isclose(
        distance[3, 5],
        2.0,
    )

    # sqrt(5) pixels away.
    assert np.isclose(
        distance[4, 5],
        np.sqrt(5),
    )

    # -----------------------------------------------------
    # Primary radius-2 mask
    # -----------------------------------------------------

    strict_radius_2 = build_strict_non_current_reference(
        current_lidar_valid=current_lidar_valid,
        reference_valid=reference_valid,
        exclusion_radius=2,
    )

    # LiDAR anchor itself must be excluded.
    assert not strict_radius_2[3, 3]

    # Distance 1 must be excluded.
    assert not strict_radius_2[3, 4]

    # Distance sqrt(2) must be excluded.
    assert not strict_radius_2[4, 4]

    # Distance exactly 2 must be excluded.
    assert not strict_radius_2[3, 5]

    # Distance sqrt(5) > 2 must remain.
    assert strict_radius_2[4, 5]

    print()
    print("Radius-2 retained-mask representation")
    print("-" * 72)

    print(
        strict_radius_2.astype(int)
    )

    print()
    print(
        "PASS: Euclidean radius-2 exclusion "
        "behaves correctly."
    )

    print("=" * 72)

if __name__ == "__main__":
    main()