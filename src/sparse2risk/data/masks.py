import numpy as np
from scipy.ndimage import distance_transform_edt

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

def calculate_distance_to_current_lidar(
    current_lidar_valid: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Euclidean pixel distance from every image pixel
    to the nearest valid current-LiDAR sample.

    Current-LiDAR pixels themselves have distance 0.

    Parameters
    ----------
    current_lidar_valid:
        Two-dimensional Boolean mask where True indicates a valid
        current-LiDAR depth measurement.

    Returns
    -------
    np.ndarray
        Float32 array containing distance in image pixels.
    """

    if current_lidar_valid.ndim != 2:
        raise ValueError(
            "Current LiDAR validity mask must be two-dimensional. "
            f"Received shape {current_lidar_valid.shape}."
        )

    current_lidar_valid = current_lidar_valid.astype(
        bool,
        copy=False,
    )

    # A frame should normally contain LiDAR measurements.
    # If none exist, every pixel is effectively infinitely
    # far away from current LiDAR.
    if not current_lidar_valid.any():
        return np.full(
            current_lidar_valid.shape,
            np.inf,
            dtype=np.float32,
        )

    # SciPy measures distance from non-zero pixels to the
    # nearest zero pixel.
    #
    # Therefore we invert the current-LiDAR mask:
    #
    # current LiDAR pixel -> False / zero
    # other pixel         -> True / non-zero
    #
    distance = distance_transform_edt(
        ~current_lidar_valid
    )

    return distance.astype(
        np.float32,
        copy=False,
    )

def build_strict_non_current_reference(
    current_lidar_valid: np.ndarray,
    reference_valid: np.ndarray,
    exclusion_radius: float = 2.0,
) -> np.ndarray:
    """
    Select reference-valid pixels that are farther than the chosen
    pixel radius from every current-LiDAR sample.

    Parameters
    ----------
    current_lidar_valid:
        Boolean mask of valid current-LiDAR pixels.

    reference_valid:
        Boolean mask of valid accumulated-reference pixels.

    exclusion_radius:
        Minimum exclusion radius around current-LiDAR samples,
        measured using Euclidean pixel distance.

        radius 0:
            Excludes only the exact current-LiDAR pixels.

        radius 1:
            Excludes pixels within 1 pixel of current LiDAR.

        radius 2:
            Primary Sparse2Risk strict evaluation setting.

        radius 3:
            Additional sensitivity analysis.

    Returns
    -------
    np.ndarray
        Boolean mask of strict non-current reference pixels.
    """

    if current_lidar_valid.shape != reference_valid.shape:
        raise ValueError(
            "Current-LiDAR and reference-valid masks must "
            "have the same shape."
        )

    if exclusion_radius < 0:
        raise ValueError(
            "Exclusion radius cannot be negative."
        )

    distance_to_lidar = calculate_distance_to_current_lidar(
        current_lidar_valid
    )

    strict_non_current_reference = (
        reference_valid
        & (distance_to_lidar > exclusion_radius)
    )

    return strict_non_current_reference