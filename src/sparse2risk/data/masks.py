import numpy as np


DEFAULT_MAX_DEPTH = 80.0


def build_basic_masks(
    lidar_depth: np.ndarray,
    reference_depth: np.ndarray,
    max_depth: float = DEFAULT_MAX_DEPTH,
) -> dict:
    """
    Build the basic Sparse2Risk validity masks.

    Parameters
    ----------
    lidar_depth:
        Current-scan sparse LiDAR depth in metres.

    reference_depth:
        Accumulated reference depth in metres.

    max_depth:
        Maximum depth included in the experiment.

    Returns
    -------
    dict
        Boolean masks describing which pixels are valid
        in the current LiDAR and reference depth maps.
    """

    # Both depth maps must describe the same image.
    if lidar_depth.shape != reference_depth.shape:
        raise ValueError(
            "LiDAR and reference depth must have the same shape. "
            f"Received {lidar_depth.shape} and "
            f"{reference_depth.shape}."
        )

    # -----------------------------------------------------
    # Current LiDAR-valid pixels
    # -----------------------------------------------------
    #
    # True when:
    #   - the depth is finite,
    #   - greater than zero,
    #   - within our 80 m experimental range.
    #
    current_lidar_valid = (
        np.isfinite(lidar_depth)
        & (lidar_depth > 0.0)
        & (lidar_depth <= max_depth)
    )

    # -----------------------------------------------------
    # Reference-valid pixels
    # -----------------------------------------------------

    reference_valid = (
        np.isfinite(reference_depth)
        & (reference_depth > 0.0)
        & (reference_depth <= max_depth)
    )

    # -----------------------------------------------------
    # Pixels valid in BOTH current LiDAR and reference
    # -----------------------------------------------------

    current_and_reference = (
        current_lidar_valid
        & reference_valid
    )

    # -----------------------------------------------------
    # Reference-valid pixels NOT observed by current LiDAR
    # -----------------------------------------------------
    #
    # These are particularly important for Sparse2Risk.
    #
    non_current_reference = (
        reference_valid
        & ~current_lidar_valid
    )

    # -----------------------------------------------------
    # Current LiDAR pixels without valid reference depth
    # -----------------------------------------------------

    current_lidar_only = (
        current_lidar_valid
        & ~reference_valid
    )

    return {
        "current_lidar_valid": current_lidar_valid,
        "reference_valid": reference_valid,
        "current_and_reference": current_and_reference,
        "non_current_reference": non_current_reference,
        "current_lidar_only": current_lidar_only,
    }