import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
from remotezip import RemoteIOError, RemoteZip
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# Sparse2Risk - Resilient Fast KITTI Raw RGB Downloader
# =========================================================
#
# Downloads ONLY the required image_02 members from remote
# KITTI *_sync.zip archives using HTTP Range requests.
#
# Improvements over the first fast downloader:
#   - robust HTTP retry policy;
#   - whole-drive retry after transient SSL/network failures;
#   - keeps already downloaded PNGs;
#   - continues to the next drive after repeated failure;
#   - reports failed drives at the end;
#   - independently verifies presence and image dimensions.
#
# Safe to re-run: existing valid RGB files are skipped.
# =========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manifests"
    / "kitti_full_depth_manifest.csv"
)

FROZEN_SPLIT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "splits"
    / "kitti_sparse2risk_v1.csv"
)

RAW_RGB_ROOT = Path(
    r"C:\FYP\dataset\kitti_raw"
)

BASE_URL = (
    "https://s3.eu-central-1.amazonaws.com/"
    "avg-kitti/raw_data"
)

EXPECTED_CAMERA = "image_02"
EXPECTED_TOTAL_DRIVES = 151
EXPECTED_TOTAL_FRAMES = 46_375

# Controlled batch size. Re-run for the next batch.
MAX_DRIVES_PER_RUN = 25

# Retry settings for unstable internet/TLS connections.
MAX_DRIVE_ATTEMPTS = 5
RETRY_BACKOFF_SECONDS = 5

REMOTE_TIMEOUT_SECONDS = 120
REMOTE_CENTRAL_DIRECTORY_BUFFER = 1024 * 1024


def read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def expected_rgb_path(
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


def build_archive_url(
    drive: str,
):
    if not drive.endswith("_sync"):
        raise ValueError(
            f"Unexpected drive name: {drive}"
        )

    base_drive = drive.removesuffix(
        "_sync"
    )

    return (
        f"{BASE_URL}/"
        f"{base_drive}/"
        f"{drive}.zip"
    )


def required_member_suffix(
    drive: str,
    frame: str,
):
    return (
        f"{drive}/"
        f"{EXPECTED_CAMERA}/"
        f"data/"
        f"{frame}.png"
    )


def verify_png(
    path: Path,
):
    with Image.open(path) as image:
        image.verify()


def image_size(
    path: Path,
):
    with Image.open(path) as image:
        return image.size


def build_resilient_session():
    """
    Create one Requests session with retries for transient
    TLS/connect/read/server failures.
    """

    retry_policy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        other=5,
        backoff_factor=1.5,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=frozenset(
            [
                "HEAD",
                "GET",
            ]
        ),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "Sparse2Risk-KITTI-RGB-Downloader/1.0"
            )
        }
    )

    return session


def missing_frames_for_drive(
    drive,
    required_rows_by_drive,
):
    return [
        row["frame"]
        for row in required_rows_by_drive[
            drive
        ]
        if not expected_rgb_path(
            drive,
            row["frame"],
        ).exists()
    ]


def download_one_drive(
    drive,
    required_rows_by_drive,
):
    """
    Try the full drive up to MAX_DRIVE_ATTEMPTS times.

    Already completed PNGs survive failures and are skipped
    on the next retry.
    """

    url = build_archive_url(
        drive
    )

    for attempt in range(
        1,
        MAX_DRIVE_ATTEMPTS + 1,
    ):

        missing_frames = (
            missing_frames_for_drive(
                drive,
                required_rows_by_drive,
            )
        )

        if not missing_frames:
            return True

        print()
        print(
            f"    Drive attempt "
            f"{attempt}/{MAX_DRIVE_ATTEMPTS}"
        )

        print(
            f"    Missing at start  : "
            f"{len(missing_frames):,}"
        )

        session = (
            build_resilient_session()
        )

        try:
            with RemoteZip(
                url,
                session=session,
                timeout=REMOTE_TIMEOUT_SECONDS,
                initial_buffer_size=(
                    REMOTE_CENTRAL_DIRECTORY_BUFFER
                ),
            ) as remote_zip:

                archive_names = (
                    remote_zip.namelist()
                )

                for index, frame in enumerate(
                    missing_frames,
                    start=1,
                ):

                    # Another retry/partial run may already
                    # have created this file.
                    destination = (
                        expected_rgb_path(
                            drive,
                            frame,
                        )
                    )

                    if destination.exists():
                        continue

                    suffix = (
                        required_member_suffix(
                            drive,
                            frame,
                        )
                    )

                    matches = [
                        name
                        for name in archive_names
                        if (
                            not name.endswith("/")
                            and name.replace(
                                "\\",
                                "/",
                            ).endswith(
                                suffix
                            )
                        )
                    ]

                    if len(matches) != 1:
                        raise RuntimeError(
                            "Expected exactly one ZIP member "
                            f"for {drive} frame {frame}; "
                            f"found {len(matches)}."
                        )

                    destination.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    temporary_path = (
                        destination.with_suffix(
                            ".png.part"
                        )
                    )

                    if temporary_path.exists():
                        temporary_path.unlink()

                    try:
                        with remote_zip.open(
                            matches[0],
                            "r",
                        ) as source:

                            with temporary_path.open(
                                "wb",
                            ) as output:

                                shutil.copyfileobj(
                                    source,
                                    output,
                                    length=1024 * 1024,
                                )

                        verify_png(
                            temporary_path
                        )

                        temporary_path.replace(
                            destination
                        )

                    except Exception:
                        if temporary_path.exists():
                            temporary_path.unlink()
                        raise

                    if (
                        index % 25 == 0
                        or index
                        == len(missing_frames)
                    ):
                        print(
                            f"\r    Downloaded + verified "
                            f"{index:,}/"
                            f"{len(missing_frames):,}",
                            end="",
                            flush=True,
                        )

                print()

        except (
            requests.exceptions.RequestException,
            RemoteIOError,
            OSError,
        ) as error:

            print()
            print(
                "    TRANSIENT NETWORK ERROR"
            )

            print(
                f"    {type(error).__name__}: "
                f"{error}"
            )

            session.close()

            if (
                attempt
                == MAX_DRIVE_ATTEMPTS
            ):
                return False

            wait_seconds = (
                RETRY_BACKOFF_SECONDS
                * attempt
            )

            print(
                f"    Keeping completed PNGs and "
                f"retrying in {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

            continue

        finally:
            session.close()

        # Verify that the drive is fully present.
        missing_after = (
            missing_frames_for_drive(
                drive,
                required_rows_by_drive,
            )
        )

        if not missing_after:
            return True

        print(
            f"    Drive still has "
            f"{len(missing_after):,} missing frames."
        )

    return False


def verify_drive_shapes(
    drive,
    required_rows_by_drive,
):
    """
    Compare raw RGB, current LiDAR and reference-depth image
    dimensions for every required frame in the drive.
    """

    missing = []
    mismatches = []

    for row in required_rows_by_drive[
        drive
    ]:

        frame = row["frame"]

        rgb_path = expected_rgb_path(
            drive,
            frame,
        )

        if not rgb_path.exists():
            missing.append(
                frame
            )
            continue

        rgb_size = image_size(
            rgb_path
        )

        lidar_size = image_size(
            Path(
                row["lidar_path"]
            )
        )

        reference_size = image_size(
            Path(
                row["reference_path"]
            )
        )

        if not (
            rgb_size
            == lidar_size
            == reference_size
        ):
            mismatches.append(
                (
                    frame,
                    rgb_size,
                    lidar_size,
                    reference_size,
                )
            )

    return (
        missing,
        mismatches,
    )


def main():

    print("=" * 82)
    print(
        "Sparse2Risk - Resilient Fast KITTI Raw RGB Downloader"
    )
    print("=" * 82)

    manifest_rows = read_csv(
        MANIFEST_PATH
    )

    frozen_rows = read_csv(
        FROZEN_SPLIT_PATH
    )

    if (
        len(manifest_rows)
        != EXPECTED_TOTAL_FRAMES
    ):
        raise RuntimeError(
            "Manifest no longer contains "
            "46,375 frames."
        )

    required_rows_by_drive = defaultdict(
        list
    )

    for row in manifest_rows:

        if row["camera"] != EXPECTED_CAMERA:
            raise RuntimeError(
                "Unexpected camera in manifest: "
                f"{row['camera']}"
            )

        required_rows_by_drive[
            row["drive"]
        ].append(
            row
        )

    manifest_drives = set(
        required_rows_by_drive
    )

    frozen_drives = {
        row["drive"]
        for row in frozen_rows
    }

    if (
        len(manifest_drives)
        != EXPECTED_TOTAL_DRIVES
    ):
        raise RuntimeError(
            "Manifest no longer contains "
            "151 drives."
        )

    if (
        manifest_drives
        != frozen_drives
    ):
        raise RuntimeError(
            "Frozen split and manifest "
            "drive sets differ."
        )

    # -----------------------------------------------------
    # Current state
    # -----------------------------------------------------

    incomplete_drives = []

    already_present = 0

    for drive in sorted(
        manifest_drives
    ):

        missing = (
            missing_frames_for_drive(
                drive,
                required_rows_by_drive,
            )
        )

        required_count = len(
            required_rows_by_drive[
                drive
            ]
        )

        already_present += (
            required_count
            - len(missing)
        )

        if missing:
            incomplete_drives.append(
                drive
            )

    print()
    print(
        f"Study drives          : "
        f"{len(manifest_drives):,}"
    )

    print(
        f"Study RGB frames      : "
        f"{len(manifest_rows):,}"
    )

    print(
        f"Already present       : "
        f"{already_present:,}"
    )

    print(
        f"Still missing         : "
        f"{EXPECTED_TOTAL_FRAMES - already_present:,}"
    )

    print(
        f"Incomplete drives     : "
        f"{len(incomplete_drives):,}"
    )

    if not incomplete_drives:
        print()
        print(
            "PASS: All required RGB frames "
            "are already present."
        )
        return

    selected_drives = (
        incomplete_drives[
            :MAX_DRIVES_PER_RUN
        ]
    )

    print()
    print(
        f"Processing up to "
        f"{MAX_DRIVES_PER_RUN} incomplete drives."
    )

    failed_drives = []
    completed_this_run = []

    # -----------------------------------------------------
    # Controlled batch
    # -----------------------------------------------------

    for drive_index, drive in enumerate(
        selected_drives,
        start=1,
    ):

        print()
        print("=" * 82)

        print(
            f"[{drive_index}/"
            f"{len(selected_drives)}] "
            f"{drive}"
        )

        print("=" * 82)

        print(
            f"Remote archive        : "
            f"{build_archive_url(drive)}"
        )

        success = download_one_drive(
            drive,
            required_rows_by_drive,
        )

        if not success:
            print()
            print(
                "FAILED AFTER RETRIES - "
                "moving to next drive."
            )

            failed_drives.append(
                drive
            )

            continue

        (
            missing_after,
            shape_mismatches,
        ) = verify_drive_shapes(
            drive,
            required_rows_by_drive,
        )

        print()
        print("Drive verification")
        print("-" * 82)

        print(
            f"Required frames       : "
            f"{len(required_rows_by_drive[drive]):,}"
        )

        print(
            f"Missing after         : "
            f"{len(missing_after):,}"
        )

        print(
            f"Shape mismatches      : "
            f"{len(shape_mismatches):,}"
        )

        if (
            missing_after
            or shape_mismatches
        ):
            print(
                "FAIL: Drive verification failed."
            )

            failed_drives.append(
                drive
            )

            continue

        print(
            "PASS: Drive complete and aligned."
        )

        completed_this_run.append(
            drive
        )

    # -----------------------------------------------------
    # Whole-study status
    # -----------------------------------------------------

    final_present = 0
    final_incomplete_drives = 0

    for drive in manifest_drives:

        missing = (
            missing_frames_for_drive(
                drive,
                required_rows_by_drive,
            )
        )

        final_present += (
            len(
                required_rows_by_drive[
                    drive
                ]
            )
            - len(missing)
        )

        if missing:
            final_incomplete_drives += 1

    final_missing = (
        EXPECTED_TOTAL_FRAMES
        - final_present
    )

    print()
    print("=" * 82)
    print("RUN SUMMARY")
    print("=" * 82)

    print(
        f"Completed this run    : "
        f"{len(completed_this_run):,}"
    )

    print(
        f"Failed this run       : "
        f"{len(failed_drives):,}"
    )

    if failed_drives:
        print()
        print("Failed drives:")
        for drive in failed_drives:
            print(
                f"  {drive}"
            )

    print()
    print(
        f"Present RGB frames    : "
        f"{final_present:,}"
    )

    print(
        f"Missing RGB frames    : "
        f"{final_missing:,}"
    )

    print(
        f"Incomplete drives     : "
        f"{final_incomplete_drives:,}"
    )

    print(
        f"Coverage              : "
        f"{100.0 * final_present / EXPECTED_TOTAL_FRAMES:.2f}%"
    )

    if final_missing == 0:
        print()
        print(
            "PASS: All 46,375 frozen-study RGB "
            "frames are present."
        )
    else:
        print()
        print(
            "Batch complete. Re-run the script "
            "to continue with remaining drives."
        )

    print("=" * 82)


if __name__ == "__main__":
    main()
