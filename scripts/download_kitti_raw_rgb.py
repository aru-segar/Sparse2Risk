import csv
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


# =========================================================
# Sparse2Risk - KITTI Raw RGB Downloader
# =========================================================
#
# Purpose
# -------
# Download the synchronized KITTI raw archives required by
# the frozen Sparse2Risk study, but extract ONLY the exact
# image_02 RGB frames referenced by the study manifest.
#
# The script:
#   1. reads the frozen 151-drive study requirement;
#   2. downloads one *_sync.zip at a time;
#   3. extracts only required image_02/data/*.png files;
#   4. verifies every required frame extracted from that drive;
#   5. deletes the temporary ZIP after successful extraction;
#   6. can be safely re-run and skips already-complete drives.
#
# It does NOT download or extract image_03, Velodyne, OXTS,
# or grayscale camera streams.
#
# Use only after you have accepted KITTI's download terms and
# obtained the official raw-data download access/script.
# =========================================================


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_MANIFEST_PATH = (
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

TEMP_DOWNLOAD_DIR = (
    RAW_RGB_ROOT
    / "_downloads"
)

BASE_URL = (
    "https://s3.eu-central-1.amazonaws.com/"
    "avg-kitti/raw_data"
)

EXPECTED_CAMERA = "image_02"

EXPECTED_TOTAL_DRIVES = 151
EXPECTED_TOTAL_FRAMES = 46_375

DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_ATTEMPTS = 3


# =========================================================
# Helpers
# =========================================================

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
        return list(csv.DictReader(file))


def build_required_frames(manifest_rows):
    """
    Return:
        required_frames_by_drive[drive] = {frame, ...}
    """

    required = defaultdict(set)

    for row in manifest_rows:
        camera = row["camera"]

        if camera != EXPECTED_CAMERA:
            raise ValueError(
                f"Unexpected camera {camera!r}; "
                f"expected only {EXPECTED_CAMERA!r}."
            )

        required[
            row["drive"]
        ].add(
            row["frame"]
        )

    return required


def expected_rgb_path(
    drive: str,
    frame: str,
):
    date = drive[:10]

    return (
        RAW_RGB_ROOT
        / date
        / drive
        / EXPECTED_CAMERA
        / "data"
        / f"{frame}.png"
    )


def missing_required_frames(
    drive,
    required_frames,
):
    return [
        frame
        for frame in sorted(required_frames)
        if not expected_rgb_path(
            drive,
            frame,
        ).exists()
    ]


def get_archive_info(drive: str):
    """
    Frozen manifest drive:
        2011_09_26_drive_0001_sync

    KITTI raw archive:
        2011_09_26_drive_0001/
        2011_09_26_drive_0001_sync.zip
    """

    if not drive.endswith("_sync"):
        raise ValueError(
            f"Expected a synchronized drive name: {drive}"
        )

    base_drive = drive.removesuffix(
        "_sync"
    )

    archive_name = (
        f"{drive}.zip"
    )

    url = (
        f"{BASE_URL}/"
        f"{base_drive}/"
        f"{archive_name}"
    )

    return (
        base_drive,
        archive_name,
        url,
    )


def download_file(
    url: str,
    destination: Path,
):
    """
    Download with resume-safe temporary '.part' handling.

    If a previous partial file exists, it is deleted and the
    current archive is downloaded again from the beginning.
    This is deliberately simple and robust.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    part_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    if part_path.exists():
        part_path.unlink()

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):
        try:
            print(
                f"    Download attempt "
                f"{attempt}/{MAX_DOWNLOAD_ATTEMPTS}"
            )

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Sparse2Risk-KITTI-RGB-Downloader/1.0"
                    )
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:

                content_length = (
                    response.headers.get(
                        "Content-Length"
                    )
                )

                total_bytes = (
                    int(content_length)
                    if content_length
                    else None
                )

                downloaded = 0

                with part_path.open(
                    "wb"
                ) as output_file:

                    while True:
                        chunk = response.read(
                            DOWNLOAD_CHUNK_SIZE
                        )

                        if not chunk:
                            break

                        output_file.write(
                            chunk
                        )

                        downloaded += len(
                            chunk
                        )

                        if total_bytes:
                            percent = (
                                100.0
                                * downloaded
                                / total_bytes
                            )

                            print(
                                f"\r    "
                                f"{downloaded / (1024**2):8.1f} MB "
                                f"/ "
                                f"{total_bytes / (1024**2):8.1f} MB "
                                f"({percent:6.2f}%)",
                                end="",
                                flush=True,
                            )

                        else:
                            print(
                                f"\r    "
                                f"{downloaded / (1024**2):8.1f} MB",
                                end="",
                                flush=True,
                            )

            print()

            part_path.replace(
                destination
            )

            return

        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as error:

            print()
            print(
                f"    Download error: {error}"
            )

            if part_path.exists():
                part_path.unlink()

            if (
                attempt
                == MAX_DOWNLOAD_ATTEMPTS
            ):
                raise

            wait_seconds = (
                5 * attempt
            )

            print(
                f"    Retrying in "
                f"{wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )


def find_required_member(
    archive_names,
    drive,
    frame,
):
    """
    Locate one exact RGB member in the ZIP.

    Expected member suffix:
        <date>/<drive>/image_02/data/<frame>.png

    We match by suffix so the code remains robust to a ZIP
    optionally containing an additional top-level prefix.
    """

    expected_suffix = (
        f"{drive}/"
        f"{EXPECTED_CAMERA}/"
        f"data/"
        f"{frame}.png"
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
                expected_suffix
            )
        )
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        return None

    raise RuntimeError(
        f"Archive contains multiple matches for "
        f"{drive}/{frame}.png: {matches}"
    )


def extract_required_frames(
    archive_path: Path,
    drive: str,
    required_frames,
):
    """
    Extract only the exact required image_02 frames.

    Files are written directly into the canonical raw-RGB
    directory rather than extracting all other sensors.
    """

    destination_dir = (
        RAW_RGB_ROOT
        / drive[:10]
        / drive
        / EXPECTED_CAMERA
        / "data"
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    still_missing = missing_required_frames(
        drive,
        required_frames,
    )

    if not still_missing:
        return 0

    extracted_count = 0

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:

        bad_member = (
            archive.testzip()
        )

        if bad_member is not None:
            raise RuntimeError(
                f"Corrupt ZIP member detected: "
                f"{bad_member}"
            )

        archive_names = (
            archive.namelist()
        )

        for index, frame in enumerate(
            still_missing,
            start=1,
        ):
            member = find_required_member(
                archive_names,
                drive,
                frame,
            )

            if member is None:
                raise FileNotFoundError(
                    f"Required RGB frame not found "
                    f"inside archive: "
                    f"{drive} frame {frame}"
                )

            destination_path = (
                destination_dir
                / f"{frame}.png"
            )

            with archive.open(
                member,
                "r",
            ) as source_file:

                with destination_path.open(
                    "wb",
                ) as destination_file:

                    shutil.copyfileobj(
                        source_file,
                        destination_file,
                    )

            extracted_count += 1

            if (
                index % 250 == 0
                or index == len(
                    still_missing
                )
            ):
                print(
                    f"\r    Extracted "
                    f"{index:,}/"
                    f"{len(still_missing):,} "
                    f"required RGB frames",
                    end="",
                    flush=True,
                )

    print()

    return extracted_count


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 82)
    print(
        "Sparse2Risk - KITTI Raw image_02 Downloader"
    )
    print("=" * 82)

    manifest_rows = read_csv(
        FULL_MANIFEST_PATH
    )

    frozen_rows = read_csv(
        FROZEN_SPLIT_PATH
    )

    required_frames_by_drive = (
        build_required_frames(
            manifest_rows
        )
    )

    frozen_drives = {
        row["drive"]
        for row in frozen_rows
    }

    manifest_drives = set(
        required_frames_by_drive
    )

    # -----------------------------------------------------
    # Safety checks before any network traffic
    # -----------------------------------------------------

    if (
        len(manifest_rows)
        != EXPECTED_TOTAL_FRAMES
    ):
        raise RuntimeError(
            "Manifest frame count changed. "
            "Run the frozen-split validator first."
        )

    if (
        len(manifest_drives)
        != EXPECTED_TOTAL_DRIVES
    ):
        raise RuntimeError(
            "Manifest drive count changed."
        )

    if (
        frozen_drives
        != manifest_drives
    ):
        raise RuntimeError(
            "Frozen split and manifest drive sets differ."
        )

    required_frame_total = sum(
        len(frames)
        for frames
        in required_frames_by_drive.values()
    )

    if (
        required_frame_total
        != EXPECTED_TOTAL_FRAMES
    ):
        raise RuntimeError(
            "Required RGB frame count is not 46,375."
        )

    RAW_RGB_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    TEMP_DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Required drives : "
        f"{len(manifest_drives):,}"
    )

    print(
        f"Required frames : "
        f"{required_frame_total:,}"
    )

    print(
        f"Destination     : "
        f"{RAW_RGB_ROOT}"
    )

    # -----------------------------------------------------
    # Current completion status
    # -----------------------------------------------------

    already_present = 0

    for drive in manifest_drives:
        for frame in (
            required_frames_by_drive[
                drive
            ]
        ):
            if expected_rgb_path(
                drive,
                frame,
            ).exists():
                already_present += 1

    print(
        f"Already present : "
        f"{already_present:,}"
    )

    print(
        f"Still required  : "
        f"{required_frame_total - already_present:,}"
    )

    print()
    print(
        "The downloader will process one synchronized "
        "drive archive at a time."
    )

    print(
        "Only required image_02 PNGs are retained."
    )

    print(
        "Temporary ZIP files are deleted after a drive "
        "passes extraction verification."
    )

    print()

    answer = input(
        "Type DOWNLOAD to begin, or anything else to stop: "
    ).strip()

    if answer != "DOWNLOAD":
        print(
            "Download cancelled. No archives were downloaded."
        )
        return

    # -----------------------------------------------------
    # Download in deterministic date/drive order
    # -----------------------------------------------------

    drives = sorted(
        manifest_drives
    )

    completed_drives = 0

    for drive_index, drive in enumerate(
        drives,
        start=1,
    ):
        required_frames = (
            required_frames_by_drive[
                drive
            ]
        )

        missing_before = (
            missing_required_frames(
                drive,
                required_frames,
            )
        )

        print()
        print("=" * 82)

        print(
            f"[{drive_index:3d}/{len(drives)}] "
            f"{drive}"
        )

        print("=" * 82)

        print(
            f"    Required frames : "
            f"{len(required_frames):,}"
        )

        print(
            f"    Missing now     : "
            f"{len(missing_before):,}"
        )

        if not missing_before:
            print(
                "    COMPLETE - skipping download."
            )

            completed_drives += 1
            continue

        (
            _,
            archive_name,
            url,
        ) = get_archive_info(
            drive
        )

        archive_path = (
            TEMP_DOWNLOAD_DIR
            / archive_name
        )

        print(
            f"    URL             : {url}"
        )

        print(
            f"    Temporary ZIP   : {archive_path}"
        )

        # If a ZIP from an interrupted prior extraction
        # remains and is a valid ZIP, reuse it.
        reuse_existing_zip = False

        if archive_path.exists():
            try:
                with zipfile.ZipFile(
                    archive_path,
                    "r",
                ) as archive:
                    reuse_existing_zip = (
                        archive.testzip()
                        is None
                    )
            except zipfile.BadZipFile:
                reuse_existing_zip = False

        if reuse_existing_zip:
            print(
                "    Reusing existing valid temporary ZIP."
            )
        else:
            if archive_path.exists():
                archive_path.unlink()

            print(
                "    Downloading synchronized raw archive..."
            )

            download_file(
                url,
                archive_path,
            )

        print(
            "    Extracting required image_02 frames..."
        )

        try:
            extract_required_frames(
                archive_path,
                drive,
                required_frames,
            )
        except zipfile.BadZipFile as error:
            raise RuntimeError(
                f"Downloaded archive is not a valid ZIP: "
                f"{archive_path}"
            ) from error

        missing_after = (
            missing_required_frames(
                drive,
                required_frames,
            )
        )

        if missing_after:
            raise RuntimeError(
                f"{drive} still has "
                f"{len(missing_after):,} missing "
                f"required RGB frames after extraction."
            )

        completed_drives += 1

        print(
            "    VERIFIED COMPLETE."
        )

        # Delete archive only after the required frames
        # for this drive have been independently verified.
        if archive_path.exists():
            archive_path.unlink()

            print(
                "    Temporary ZIP deleted."
            )

    # -----------------------------------------------------
    # Final whole-study verification
    # -----------------------------------------------------

    missing_final = []

    for drive in sorted(manifest_drives):
        for frame in sorted(
            required_frames_by_drive[
                drive
            ]
        ):
            path = expected_rgb_path(
                drive,
                frame,
            )

            if not path.exists():
                missing_final.append(
                    (
                        drive,
                        frame,
                        path,
                    )
                )

    print()
    print("=" * 82)
    print("FINAL DOWNLOAD VERIFICATION")
    print("=" * 82)

    print(
        f"Complete drives      : "
        f"{completed_drives:,}/"
        f"{len(drives):,}"
    )

    print(
        f"Required frames      : "
        f"{required_frame_total:,}"
    )

    print(
        f"Missing final frames : "
        f"{len(missing_final):,}"
    )

    if not missing_final:
        print()
        print(
            "PASS: All 46,375 frozen-study image_02 "
            "RGB frames are present."
        )

        print(
            "Run plan_raw_rgb_acquisition.py again "
            "to perform the independent acquisition audit."
        )
    else:
        print()
        print(
            "FAIL: Some required RGB frames are "
            "still missing."
        )

        sys.exit(1)

    print("=" * 82)


if __name__ == "__main__":
    main()
