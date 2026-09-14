import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


# =========================================================
# Sparse2Risk - Final Drive-Disjoint Split Optimizer
# =========================================================
#
# Purpose
# -------
# Create a reproducible drive-level candidate split for the
# final Sparse2Risk study.
#
# Policy
# ------
# 1. All original KITTI "val" drives are fixed to DEV because
#    they were already represented in the 1,000-frame pilot.
#
# 2. Only original KITTI "train" drives are eligible for:
#       TRAIN / CALIBRATION / LOCKED_TEST
#
# 3. Drives are never split across partitions.
#
# 4. CALIBRATION and LOCKED_TEST each target 10% of the
#    pilot-unseen frame pool.
#
# 5. The optimizer uses ONLY dataset metadata:
#       - source split
#       - drive identity
#       - recording date
#       - frame count
#
#    It does NOT use teacher accuracy, Sparse2Risk performance,
#    depth errors, AUSE, Spearman, calibration results, or any
#    other model-derived quantity.
#
# 6. Optimization is lexicographic:
#       Stage 1: minimize CAL/TEST frame-count error.
#       Stage 2: among equally good size solutions, make the
#                LOCKED TEST date distribution as representative
#                as possible.
#       Stage 3: among those solutions, make CALIBRATION date
#                distribution as representative as possible.
#       Stage 4: among those solutions, reduce single-drive
#                dominance in CALIBRATION and LOCKED TEST.
#
# This avoids arbitrary weighted trade-offs between scientific
# priorities.
# =========================================================


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "manifests"
    / "kitti_full_depth_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "split_optimization"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_priority_drive_split_candidate.csv"
)


# ---------------------------------------------------------
# Dataset policy
# ---------------------------------------------------------

DEV_SOURCE_SPLIT = "val"
UNSEEN_SOURCE_SPLIT = "train"


# ---------------------------------------------------------
# Desired allocation of the pilot-unseen pool
# ---------------------------------------------------------

CALIBRATION_FRACTION = 0.10
TEST_FRACTION = 0.10

# This is only a safety range.
# The first optimization stage will try to get much closer
# than this and, if exact targets are feasible, will return
# zero frame-count error.
MAX_TARGET_DEVIATION = 0.10


# ---------------------------------------------------------
# Solver settings
# ---------------------------------------------------------

SOLVER_TIME_LIMIT_SECONDS = 180.0

# Used when freezing an optimum before moving to the next
# lexicographic optimization stage.
OBJECTIVE_TOLERANCE = 1e-6


# =========================================================
# Data loading
# =========================================================

def load_manifest():
    """
    Load the previously generated full KITTI depth manifest.
    """

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


def build_drive_records(rows):
    """
    Convert the frame-level manifest into one record per drive.
    """

    frames_per_drive = Counter()
    source_by_drive = {}

    for row in rows:
        drive = row["drive"]
        source_split = row["source_split"]

        frames_per_drive[drive] += 1

        if drive in source_by_drive:
            if source_by_drive[drive] != source_split:
                raise ValueError(
                    f"Drive {drive} appears in multiple "
                    "source splits."
                )

        source_by_drive[drive] = source_split

    records = []

    for drive in sorted(frames_per_drive):
        records.append(
            {
                "drive": drive,
                "date": drive[:10],
                "frames": frames_per_drive[drive],
                "source_split": source_by_drive[drive],
            }
        )

    return records


# =========================================================
# Summary helpers
# =========================================================

def summarize_split(records):
    """
    Summarise one drive-level partition.
    """

    frames_per_date = Counter()

    for record in records:
        frames_per_date[record["date"]] += record["frames"]

    total_frames = sum(
        record["frames"]
        for record in records
    )

    largest_drive_frames = max(
        (
            record["frames"]
            for record in records
        ),
        default=0,
    )

    largest_drive_share = (
        largest_drive_frames / total_frames
        if total_frames > 0
        else 0.0
    )

    return {
        "frames": total_frames,
        "drives": len(records),
        "dates": len(frames_per_date),
        "frames_per_date": frames_per_date,
        "largest_drive_frames": largest_drive_frames,
        "largest_drive_share": largest_drive_share,
    }


def date_distribution_l1_error(
    summary,
    dates,
    pool_date_share,
):
    """
    L1 distance between a split's date proportions and the
    pilot-unseen pool date proportions.

    Lower is better. Zero would mean identical proportions.
    """

    if summary["frames"] == 0:
        return float("inf")

    error = 0.0

    for date in dates:
        split_share = (
            summary["frames_per_date"].get(date, 0)
            / summary["frames"]
        )

        error += abs(
            split_share
            - pool_date_share[date]
        )

    return error


# =========================================================
# Optimization helper
# =========================================================

def solve_exact_stage(
    objective,
    integrality,
    bounds,
    constraint_rows,
    lower_limits,
    upper_limits,
    stage_name,
):
    """
    Solve one MILP stage and require an optimal solution.
    """

    constraints = LinearConstraint(
        np.vstack(constraint_rows),
        np.array(lower_limits),
        np.array(upper_limits),
    )

    print()
    print("-" * 78)
    print(stage_name)
    print("-" * 78)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "time_limit": SOLVER_TIME_LIMIT_SECONDS,
            "mip_rel_gap": 0.0,
        },
    )

    print(
        f"Solver status : {result.status}"
    )

    print(
        f"Solver message: {result.message}"
    )

    if result.x is None:
        raise RuntimeError(
            f"{stage_name} did not return a feasible solution."
        )

    if result.status != 0:
        raise RuntimeError(
            f"{stage_name} did not prove optimality. "
            "Do not freeze a split from this run."
        )

    objective_value = float(
        np.dot(
            objective,
            result.x,
        )
    )

    print(
        f"Objective     : {objective_value:.8f}"
    )

    return result, objective_value


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 78)
    print(
        "Sparse2Risk - Final Lexicographic Drive Split Optimizer"
    )
    print("=" * 78)

    # -----------------------------------------------------
    # Load and aggregate data
    # -----------------------------------------------------

    rows = load_manifest()

    drive_records = build_drive_records(
        rows
    )

    dev_records = [
        record
        for record in drive_records
        if record["source_split"] == DEV_SOURCE_SPLIT
    ]

    unseen_records = [
        record
        for record in drive_records
        if record["source_split"] == UNSEEN_SOURCE_SPLIT
    ]

    if not dev_records:
        raise RuntimeError(
            "No fixed DEV drives were found."
        )

    if not unseen_records:
        raise RuntimeError(
            "No pilot-unseen drives were found."
        )

    # -----------------------------------------------------
    # Dataset statistics
    # -----------------------------------------------------

    dates = sorted(
        {
            record["date"]
            for record in unseen_records
        }
    )

    n = len(unseen_records)
    d = len(dates)

    drive_frames = np.array(
        [
            record["frames"]
            for record in unseen_records
        ],
        dtype=float,
    )

    unseen_total_frames = int(
        drive_frames.sum()
    )

    calibration_target = round(
        unseen_total_frames
        * CALIBRATION_FRACTION
    )

    test_target = round(
        unseen_total_frames
        * TEST_FRACTION
    )

    train_target = (
        unseen_total_frames
        - calibration_target
        - test_target
    )

    pool_frames_per_date = Counter()

    for record in unseen_records:
        pool_frames_per_date[
            record["date"]
        ] += record["frames"]

    pool_date_share = {
        date: (
            pool_frames_per_date[date]
            / unseen_total_frames
        )
        for date in dates
    }

    dev_summary = summarize_split(
        dev_records
    )

    print()
    print("Fixed DEV population")
    print("-" * 78)

    print(
        f"Drives : {dev_summary['drives']:,}"
    )

    print(
        f"Frames : {dev_summary['frames']:,}"
    )

    print(
        f"Dates  : {dev_summary['dates']:,}"
    )

    print()
    print("Pilot-unseen allocation pool")
    print("-" * 78)

    print(
        f"Drives : {n:,}"
    )

    print(
        f"Frames : {unseen_total_frames:,}"
    )

    print()
    print("Target frame counts")
    print("-" * 78)

    print(
        f"TRAIN        : {train_target:,}"
    )

    print(
        f"CALIBRATION  : {calibration_target:,}"
    )

    print(
        f"LOCKED TEST  : {test_target:,}"
    )

    print()
    print("Pilot-unseen date distribution")
    print("-" * 78)

    for date in dates:
        print(
            f"{date}: "
            f"{pool_frames_per_date[date]:6,d} frames "
            f"({100.0 * pool_date_share[date]:5.2f}%)"
        )

    # =====================================================
    # Variable layout
    # =====================================================
    #
    # Binary variables:
    #
    #   cal_drive[i]
    #   test_drive[i]
    #
    # Continuous variables:
    #
    #   cal_size_positive
    #   cal_size_negative
    #   test_size_positive
    #   test_size_negative
    #
    #   cal_date_positive[j]
    #   cal_date_negative[j]
    #   test_date_positive[j]
    #   test_date_negative[j]
    #
    #   cal_largest_drive
    #   test_largest_drive
    #
    # TRAIN is implicit:
    # any unseen drive selected by neither CAL nor TEST.
    # =====================================================

    cal_drive_start = 0
    test_drive_start = n

    continuous_start = 2 * n

    cal_size_positive = continuous_start
    cal_size_negative = continuous_start + 1
    test_size_positive = continuous_start + 2
    test_size_negative = continuous_start + 3

    cal_date_positive_start = continuous_start + 4
    cal_date_negative_start = (
        cal_date_positive_start + d
    )

    test_date_positive_start = (
        cal_date_negative_start + d
    )

    test_date_negative_start = (
        test_date_positive_start + d
    )

    cal_largest_drive = (
        test_date_negative_start + d
    )

    test_largest_drive = (
        cal_largest_drive + 1
    )

    variable_count = (
        test_largest_drive + 1
    )

    # =====================================================
    # Variable types and bounds
    # =====================================================

    integrality = np.zeros(
        variable_count,
        dtype=int,
    )

    # The first 2*n variables are binary assignments.
    integrality[
        : 2 * n
    ] = 1

    lower_bounds = np.zeros(
        variable_count,
        dtype=float,
    )

    upper_bounds = np.full(
        variable_count,
        np.inf,
        dtype=float,
    )

    upper_bounds[
        : 2 * n
    ] = 1.0

    bounds = Bounds(
        lower_bounds,
        upper_bounds,
    )

    # =====================================================
    # Base constraints
    # =====================================================

    constraint_rows = []
    lower_limits = []
    upper_limits = []

    def add_constraint(
        coefficients,
        minimum=-np.inf,
        maximum=np.inf,
    ):
        constraint_rows.append(
            coefficients
        )

        lower_limits.append(
            minimum
        )

        upper_limits.append(
            maximum
        )

    # -----------------------------------------------------
    # HARD RULE 1:
    # one unseen drive cannot be both CAL and TEST.
    # -----------------------------------------------------

    for index in range(n):
        row = np.zeros(
            variable_count
        )

        row[
            cal_drive_start + index
        ] = 1.0

        row[
            test_drive_start + index
        ] = 1.0

        add_constraint(
            row,
            maximum=1.0,
        )

    # -----------------------------------------------------
    # HARD RULE 2:
    # CAL and TEST must remain reasonably close to 10%.
    # -----------------------------------------------------

    for split_start, target in (
        (
            cal_drive_start,
            calibration_target,
        ),
        (
            test_drive_start,
            test_target,
        ),
    ):
        row = np.zeros(
            variable_count
        )

        row[
            split_start
            : split_start + n
        ] = drive_frames

        add_constraint(
            row,
            minimum=(
                target
                * (
                    1.0
                    - MAX_TARGET_DEVIATION
                )
            ),
            maximum=(
                target
                * (
                    1.0
                    + MAX_TARGET_DEVIATION
                )
            ),
        )

    # -----------------------------------------------------
    # Absolute size-error equations
    #
    # actual - positive + negative = target
    # -----------------------------------------------------

    row = np.zeros(
        variable_count
    )

    row[
        cal_drive_start
        : cal_drive_start + n
    ] = drive_frames

    row[
        cal_size_positive
    ] = -1.0

    row[
        cal_size_negative
    ] = 1.0

    add_constraint(
        row,
        minimum=calibration_target,
        maximum=calibration_target,
    )

    row = np.zeros(
        variable_count
    )

    row[
        test_drive_start
        : test_drive_start + n
    ] = drive_frames

    row[
        test_size_positive
    ] = -1.0

    row[
        test_size_negative
    ] = 1.0

    add_constraint(
        row,
        minimum=test_target,
        maximum=test_target,
    )

    # -----------------------------------------------------
    # Absolute date-distribution residual equations
    #
    # For each date:
    #
    # actual_date_frames
    # - expected_pool_share * split_total_frames
    # - positive + negative
    # = 0
    #
    # Minimizing positive + negative gives absolute error.
    # -----------------------------------------------------

    for date_index, date in enumerate(
        dates
    ):
        expected_share = (
            pool_date_share[date]
        )

        date_member_indices = [
            index
            for index, record
            in enumerate(unseen_records)
            if record["date"] == date
        ]

        # CALIBRATION residual
        row = np.zeros(
            variable_count
        )

        row[
            cal_drive_start
            : cal_drive_start + n
        ] = (
            -expected_share
            * drive_frames
        )

        for index in date_member_indices:
            row[
                cal_drive_start + index
            ] += drive_frames[index]

        row[
            cal_date_positive_start
            + date_index
        ] = -1.0

        row[
            cal_date_negative_start
            + date_index
        ] = 1.0

        add_constraint(
            row,
            minimum=0.0,
            maximum=0.0,
        )

        # LOCKED TEST residual
        row = np.zeros(
            variable_count
        )

        row[
            test_drive_start
            : test_drive_start + n
        ] = (
            -expected_share
            * drive_frames
        )

        for index in date_member_indices:
            row[
                test_drive_start + index
            ] += drive_frames[index]

        row[
            test_date_positive_start
            + date_index
        ] = -1.0

        row[
            test_date_negative_start
            + date_index
        ] = 1.0

        add_constraint(
            row,
            minimum=0.0,
            maximum=0.0,
        )

    # -----------------------------------------------------
    # Largest-selected-drive constraints
    #
    # largest_drive >= selected_i * frames_i
    # -----------------------------------------------------

    for index in range(n):

        # CALIBRATION
        row = np.zeros(
            variable_count
        )

        row[
            cal_drive_start + index
        ] = drive_frames[index]

        row[
            cal_largest_drive
        ] = -1.0

        add_constraint(
            row,
            maximum=0.0,
        )

        # LOCKED TEST
        row = np.zeros(
            variable_count
        )

        row[
            test_drive_start + index
        ] = drive_frames[index]

        row[
            test_largest_drive
        ] = -1.0

        add_constraint(
            row,
            maximum=0.0,
        )

    # =====================================================
    # STAGE 1
    # Minimize CAL + TEST frame-count error.
    # =====================================================

    stage_1_objective = np.zeros(
        variable_count,
        dtype=float,
    )

    for index in (
        cal_size_positive,
        cal_size_negative,
        test_size_positive,
        test_size_negative,
    ):
        stage_1_objective[index] = 1.0

    stage_1_result, stage_1_optimum = (
        solve_exact_stage(
            objective=stage_1_objective,
            integrality=integrality,
            bounds=bounds,
            constraint_rows=constraint_rows,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            stage_name=(
                "STAGE 1 - Minimize CAL/TEST frame-count error"
            ),
        )
    )

    # Size error is measured in integer frame counts.
    # Freeze the exact optimum before moving on.
    stage_1_optimum = round(
        stage_1_optimum
    )

    add_constraint(
        stage_1_objective.copy(),
        maximum=(
            stage_1_optimum
            + OBJECTIVE_TOLERANCE
        ),
    )

    # =====================================================
    # STAGE 2
    # Among optimal-size solutions, make LOCKED TEST
    # date composition as representative as possible.
    # =====================================================

    stage_2_objective = np.zeros(
        variable_count,
        dtype=float,
    )

    stage_2_objective[
        test_date_positive_start
        : test_date_positive_start + d
    ] = 1.0

    stage_2_objective[
        test_date_negative_start
        : test_date_negative_start + d
    ] = 1.0

    stage_2_result, stage_2_optimum = (
        solve_exact_stage(
            objective=stage_2_objective,
            integrality=integrality,
            bounds=bounds,
            constraint_rows=constraint_rows,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            stage_name=(
                "STAGE 2 - Optimize LOCKED TEST date distribution"
            ),
        )
    )

    add_constraint(
        stage_2_objective.copy(),
        maximum=(
            stage_2_optimum
            + OBJECTIVE_TOLERANCE
        ),
    )

    # =====================================================
    # STAGE 3
    # Among equally good test solutions, improve
    # CALIBRATION date composition.
    # =====================================================

    stage_3_objective = np.zeros(
        variable_count,
        dtype=float,
    )

    stage_3_objective[
        cal_date_positive_start
        : cal_date_positive_start + d
    ] = 1.0

    stage_3_objective[
        cal_date_negative_start
        : cal_date_negative_start + d
    ] = 1.0

    stage_3_result, stage_3_optimum = (
        solve_exact_stage(
            objective=stage_3_objective,
            integrality=integrality,
            bounds=bounds,
            constraint_rows=constraint_rows,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            stage_name=(
                "STAGE 3 - Optimize CALIBRATION date distribution"
            ),
        )
    )

    add_constraint(
        stage_3_objective.copy(),
        maximum=(
            stage_3_optimum
            + OBJECTIVE_TOLERANCE
        ),
    )

    # =====================================================
    # STAGE 4
    # Among all solutions tied on the priorities above,
    # reduce single-drive dominance.
    # =====================================================

    stage_4_objective = np.zeros(
        variable_count,
        dtype=float,
    )

    stage_4_objective[
        cal_largest_drive
    ] = 1.0

    stage_4_objective[
        test_largest_drive
    ] = 1.0

    final_result, stage_4_optimum = (
        solve_exact_stage(
            objective=stage_4_objective,
            integrality=integrality,
            bounds=bounds,
            constraint_rows=constraint_rows,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            stage_name=(
                "STAGE 4 - Reduce CAL/TEST single-drive dominance"
            ),
        )
    )

    solution = final_result.x

    # =====================================================
    # Decode final candidate
    # =====================================================

    calibration_records = []
    test_records = []
    train_records = []

    for index, record in enumerate(
        unseen_records
    ):

        calibration_selected = (
            solution[
                cal_drive_start + index
            ]
            > 0.5
        )

        test_selected = (
            solution[
                test_drive_start + index
            ]
            > 0.5
        )

        if calibration_selected:
            calibration_records.append(
                record
            )

        elif test_selected:
            test_records.append(
                record
            )

        else:
            train_records.append(
                record
            )

    partitions = {
        "TRAIN": train_records,
        "DEV": dev_records,
        "CALIBRATION": calibration_records,
        "LOCKED_TEST": test_records,
    }

    summaries = {
        name: summarize_split(records)
        for name, records in partitions.items()
    }

    # =====================================================
    # Report optimization priorities
    # =====================================================

    print()
    print("=" * 78)
    print("LEXICOGRAPHIC OPTIMIZATION RESULT")
    print("=" * 78)

    print(
        f"Stage 1 total size error           : "
        f"{stage_1_optimum:.0f} frames"
    )

    print(
        f"Stage 2 LOCKED TEST date residual  : "
        f"{stage_2_optimum:.6f}"
    )

    print(
        f"Stage 3 CALIBRATION date residual  : "
        f"{stage_3_optimum:.6f}"
    )

    print(
        f"Stage 4 largest-drive objective    : "
        f"{stage_4_optimum:.6f}"
    )

    # =====================================================
    # Partition summary
    # =====================================================

    print()
    print("=" * 78)
    print("FINAL OPTIMIZED SPLIT CANDIDATE")
    print("=" * 78)

    for name in (
        "TRAIN",
        "DEV",
        "CALIBRATION",
        "LOCKED_TEST",
    ):
        summary = summaries[name]

        print(
            f"{name:<13}"
            f"frames={summary['frames']:6,d}  "
            f"drives={summary['drives']:3d}  "
            f"dates={summary['dates']}  "
            f"largest_drive="
            f"{summary['largest_drive_frames']:5,d} "
            f"("
            f"{100.0 * summary['largest_drive_share']:5.1f}%"
            f")"
        )

    print()
    print("Target differences")
    print("-" * 78)

    print(
        f"TRAIN        : "
        f"{summaries['TRAIN']['frames'] - train_target:+,}"
    )

    print(
        f"CALIBRATION  : "
        f"{summaries['CALIBRATION']['frames'] - calibration_target:+,}"
    )

    print(
        f"LOCKED TEST  : "
        f"{summaries['LOCKED_TEST']['frames'] - test_target:+,}"
    )

    # =====================================================
    # Date distributions
    # =====================================================

    calibration_summary = (
        summaries["CALIBRATION"]
    )

    test_summary = (
        summaries["LOCKED_TEST"]
    )

    print()
    print("=" * 78)
    print("DATE DISTRIBUTION")
    print("=" * 78)

    for date in dates:
        pool_percent = (
            100.0
            * pool_date_share[date]
        )

        calibration_frames = (
            calibration_summary[
                "frames_per_date"
            ].get(date, 0)
        )

        test_frames = (
            test_summary[
                "frames_per_date"
            ].get(date, 0)
        )

        calibration_percent = (
            100.0
            * calibration_frames
            / calibration_summary["frames"]
        )

        test_percent = (
            100.0
            * test_frames
            / test_summary["frames"]
        )

        print(
            f"{date}  "
            f"pool={pool_percent:5.2f}%  "
            f"cal={calibration_frames:5,d} "
            f"({calibration_percent:5.2f}%)  "
            f"test={test_frames:5,d} "
            f"({test_percent:5.2f}%)"
        )

    calibration_l1_error = (
        date_distribution_l1_error(
            calibration_summary,
            dates,
            pool_date_share,
        )
    )

    test_l1_error = (
        date_distribution_l1_error(
            test_summary,
            dates,
            pool_date_share,
        )
    )

    print()
    print("Date-distribution L1 error")
    print("-" * 78)

    print(
        f"CALIBRATION : "
        f"{calibration_l1_error:.6f}"
    )

    print(
        f"LOCKED TEST : "
        f"{test_l1_error:.6f}"
    )

    print(
        "Lower is better."
    )

    # =====================================================
    # Drive lists
    # =====================================================

    print()
    print("=" * 78)
    print("CALIBRATION DRIVES")
    print("=" * 78)

    for record in sorted(
        calibration_records,
        key=lambda item: (
            item["date"],
            item["drive"],
        ),
    ):
        print(
            f"{record['frames']:5,d}  "
            f"{record['drive']}"
        )

    print()
    print("=" * 78)
    print("LOCKED TEST DRIVES")
    print("=" * 78)

    for record in sorted(
        test_records,
        key=lambda item: (
            item["date"],
            item["drive"],
        ),
    ):
        print(
            f"{record['frames']:5,d}  "
            f"{record['drive']}"
        )

    # =====================================================
    # Integrity checks
    # =====================================================

    drive_sets = {
        name: {
            record["drive"]
            for record in records
        }
        for name, records in partitions.items()
    }

    partition_names = list(
        drive_sets
    )

    no_overlap = True

    for first_index in range(
        len(partition_names)
    ):
        for second_index in range(
            first_index + 1,
            len(partition_names),
        ):
            first_name = (
                partition_names[first_index]
            )

            second_name = (
                partition_names[second_index]
            )

            if not (
                drive_sets[first_name]
                .isdisjoint(
                    drive_sets[second_name]
                )
            ):
                no_overlap = False

    pilot_seen_outside_dev = len(
        drive_sets["DEV"]
        & (
            drive_sets["TRAIN"]
            | drive_sets["CALIBRATION"]
            | drive_sets["LOCKED_TEST"]
        )
    )

    total_drive_count = sum(
        len(drive_set)
        for drive_set in drive_sets.values()
    )

    total_frame_count = sum(
        summary["frames"]
        for summary in summaries.values()
    )

    all_drives_present = (
        total_drive_count
        == len(drive_records)
    )

    all_frames_present = (
        total_frame_count
        == len(rows)
    )

    exact_calibration_target = (
        summaries["CALIBRATION"]["frames"]
        == calibration_target
    )

    exact_test_target = (
        summaries["LOCKED_TEST"]["frames"]
        == test_target
    )

    print()
    print("=" * 78)
    print("INTEGRITY CHECKS")
    print("=" * 78)

    print(
        f"No drive overlap                    : "
        f"{no_overlap}"
    )

    print(
        f"Pilot-seen drives outside DEV       : "
        f"{pilot_seen_outside_dev}"
    )

    print(
        f"All drives represented              : "
        f"{all_drives_present} "
        f"({total_drive_count}/{len(drive_records)})"
    )

    print(
        f"All frames represented              : "
        f"{all_frames_present} "
        f"({total_frame_count:,}/{len(rows):,})"
    )

    print(
        f"Exact CALIBRATION target            : "
        f"{exact_calibration_target}"
    )

    print(
        f"Exact LOCKED TEST target            : "
        f"{exact_test_target}"
    )

    # =====================================================
    # Save provisional drive-level candidate
    # =====================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    for split_name, records in (
        partitions.items()
    ):
        for record in records:
            output_rows.append(
                {
                    **record,
                    "proposed_split": split_name,
                }
            )

    output_rows.sort(
        key=lambda row: (
            row["proposed_split"],
            row["date"],
            row["drive"],
        )
    )

    fieldnames = [
        "drive",
        "date",
        "frames",
        "source_split",
        "proposed_split",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    print()
    print(
        "Candidate saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # =====================================================
    # Final status
    # =====================================================

    print()

    if (
        no_overlap
        and pilot_seen_outside_dev == 0
        and all_drives_present
        and all_frames_present
        and exact_calibration_target
        and exact_test_target
    ):
        print(
            "PASS: Final optimized candidate satisfies "
            "all partition-integrity checks."
        )

    else:
        print(
            "WARNING: Review the candidate before freezing."
        )

    print()
    print(
        "IMPORTANT: This file is still a candidate."
    )

    print(
        "Do not freeze or commit the final split until "
        "the printed allocation has been reviewed."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
