"""
Leakage-safe grouped dataset splitting by player identity or source video.
Ensures events from the same subject / video never leak across train, validation, and test splits.

Guarantees:
1. Uses player_id grouping when reliable player identity metadata is available.
2. Falls back to video_id grouping when player_id is absent, documenting that player leakage
   across videos may occur if the same athlete appears across different video files.
3. Requires at least 3 independent groups for valid 3-way train/val/test evaluation.
   Throws an explicit ValueError on < 3 groups rather than assigning duplicate groups.
4. Strictly isolates train, validation, and test partitions with pairwise disjoint groups.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import pandas as pd


def detect_group_column(df: pd.DataFrame, requested_col: Optional[str] = None) -> Tuple[str, str]:
    """
    Detects the best grouping column and describes the leakage protection level.
    Returns (actual_group_col, leakage_status_description).
    """
    if requested_col and requested_col in df.columns:
        if requested_col == "player_id":
            return "player_id", "player_grouped (athlete-independent isolation)"
        return requested_col, f"grouped by '{requested_col}'"

    # Prioritize player_id if present with non-empty values
    if "player_id" in df.columns:
        valid_players = df["player_id"].dropna().astype(str).str.strip()
        if valid_players.ne("").any():
            return "player_id", "player_grouped (athlete-independent isolation)"

    # Fallback to source video columns
    for fallback in ["video_id", "source_video", "video_name", "sequence_id"]:
        if fallback in df.columns:
            valid_vals = df[fallback].dropna().astype(str).str.strip()
            if valid_vals.ne("").any():
                desc = (
                    f"video_grouped by '{fallback}' "
                    "(Note: player leakage across videos possible if the same athlete appears in multiple videos without player_id metadata)"
                )
                return fallback, desc

    raise ValueError(
        "No valid grouping column found in dataset. Expected 'player_id' or 'video_id'. "
        f"Available columns: {list(df.columns)}"
    )


def grouped_train_val_test_split(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions dataset into train, validation, and test sets grouped by subject or video.
    Guarantees no group leakage across splits.

    Raises:
        ValueError: If df has fewer than 3 independent groups for 3-way splitting.
    """
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    actual_group_col, _ = detect_group_column(df, group_col)

    # Extract non-empty unique group values
    valid_mask = df[actual_group_col].notna() & df[actual_group_col].astype(str).str.strip().ne("")
    valid_groups = df.loc[valid_mask, actual_group_col].astype(str).str.strip().unique()
    unique_groups = np.array(sorted(list(valid_groups)))

    n_groups = len(unique_groups)

    # Require at least 3 independent groups for a valid 3-way train/validation/test split
    if n_groups < 3:
        raise ValueError(
            f"Insufficient independent groups for valid 3-way train/val/test evaluation: "
            f"found {n_groups} group(s) in '{actual_group_col}' ({list(unique_groups)}). "
            f"At least 3 independent groups are required to isolate train, validation, and test "
            f"partitions without leakage. Cannot evaluate models without at least 3 independent groups."
        )

    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_groups)

    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(1, int(round(n_groups * val_ratio)))
    if n_train + n_val >= n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1
    n_test = n_groups - (n_train + n_val)
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1

    train_groups: Set[str] = set(unique_groups[:n_train])
    val_groups: Set[str] = set(unique_groups[n_train : n_train + n_val])
    test_groups: Set[str] = set(unique_groups[n_train + n_val :])

    # Enforce strict pairwise disjoint invariant
    assert train_groups.isdisjoint(val_groups), "Train and validation groups must be strictly disjoint!"
    assert train_groups.isdisjoint(test_groups), "Train and test groups must be strictly disjoint!"
    assert val_groups.isdisjoint(test_groups), "Validation and test groups must be strictly disjoint!"

    str_col = df[actual_group_col].astype(str).str.strip()
    train_df = df[str_col.isin(train_groups)].copy()
    val_df = df[str_col.isin(val_groups)].copy()
    test_df = df[str_col.isin(test_groups)].copy()

    return train_df, val_df, test_df
