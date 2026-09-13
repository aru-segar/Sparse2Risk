from pathlib import Path

import numpy as np
from PIL import Image


KITTI_DEPTH_SCALE = 256.0


def load_kitti_depth(path: str | Path) -> np.ndarray:
    """
    Load a KITTI depth PNG and convert stored values to metres.

    KITTI encoding:
        depth_in_metres = stored_value / 256

    Stored value 0 means that no valid depth measurement exists
    at that pixel.

    Parameters
    ----------
    path:
        Path to a KITTI depth PNG file.

    Returns
    -------
    np.ndarray
        A two-dimensional float32 array containing depth in metres.
        Invalid pixels remain 0.0.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"KITTI depth file does not exist: {path}"
        )

    with Image.open(path) as image:
        raw_depth = np.array(image)

    if raw_depth.ndim != 2:
        raise ValueError(
            "KITTI depth image must be single-channel. "
            f"Received shape {raw_depth.shape} from {path}"
        )

    if not np.issubdtype(raw_depth.dtype, np.integer):
        raise TypeError(
            "Expected integer-valued KITTI depth PNG, "
            f"but received dtype {raw_depth.dtype}"
        )

    depth_metres = (
        raw_depth.astype(np.float32) / KITTI_DEPTH_SCALE
    )

    return depth_metres


def valid_depth_mask(
    depth: np.ndarray,
    max_depth: float | None = None,
) -> np.ndarray:
    """
    Create a Boolean mask indicating valid depth pixels.

    Parameters
    ----------
    depth:
        Depth array measured in metres.

    max_depth:
        Optional maximum permitted depth. If None, only positive,
        finite depth is required.

    Returns
    -------
    np.ndarray
        Boolean mask with True at valid pixels.
    """

    mask = np.isfinite(depth) & (depth > 0.0)

    if max_depth is not None:
        mask &= depth <= max_depth

    return mask