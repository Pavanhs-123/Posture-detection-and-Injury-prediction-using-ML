import pandas as pd
import pytest

from train import split_by_video_groups


def test_split_by_video_groups_keeps_videos_disjoint_and_classes_present():
    rows = []
    for video_index in range(9):
        for label in ("LOW", "HIGH"):
            rows.append({"video_id": f"video-{video_index}", "risk_label": label})
    frame = pd.DataFrame(rows)
    X = frame[["video_id"]]
    y = frame["risk_label"]

    partitions = split_by_video_groups(X, y, frame["video_id"], seed=17)

    video_sets = [set(part_x["video_id"]) for part_x, _ in partitions]
    assert all(video_sets[i].isdisjoint(video_sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(set(part_y) == {"LOW", "HIGH"} for _, part_y in partitions)
    assert [len(part_x) for part_x, _ in partitions] == [12, 2, 4]


def test_split_by_video_groups_rejects_class_with_fewer_than_three_videos():
    frame = pd.DataFrame(
        {
            "video_id": ["video-1", "video-2", "video-3", "video-4"],
            "risk_label": ["LOW", "HIGH", "LOW", "HIGH"],
        }
    )

    with pytest.raises(ValueError, match="Each class needs at least 3 independent videos"):
        split_by_video_groups(frame[["video_id"]], frame["risk_label"], frame["video_id"], seed=42)


def test_split_by_video_groups_rejects_blank_video_ids():
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.Series(["LOW", "HIGH", "LOW"])

    with pytest.raises(ValueError, match="non-empty video_id"):
        split_by_video_groups(X, y, pd.Series(["video-1", "", "video-3"]), seed=42)
