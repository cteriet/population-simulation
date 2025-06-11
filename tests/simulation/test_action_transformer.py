# ./simulation/test_action_transformer.py

import pytest
import polars as pl
import polars.testing as pl_testing

# Import the functions and Hierarchy class from your simulation package
from simulation.action_transformer import transform_actions, transform_actions_to_series
from simulation.hierarchy import (
    ActionHierarchy,
)  # Assuming Hierarchy and ActionHierarchy are in simulation/hierarchy.py
import warnings

# --- Fixtures for Hierarchy Objects ---


@pytest.fixture
def two_level_hierarchy():
    """Creates a simple two-level action hierarchy for testing."""
    # Structure: AllActions -> A1, A2
    # Actionable: A1, A2, NoAction
    edges = [
        ("AllActions", "A1"),
        ("AllActions", "A2"),
    ]
    costs = {
        "A1": 1.0,
        "A2": 2.0,
        "NoAction": 0.0,
    }
    return ActionHierarchy(edges=edges, costs=costs)


@pytest.fixture
def three_level_hierarchy():
    """Creates a three-level action hierarchy for testing."""
    # Structure: Root -> L1A, L1B
    # L1A -> L2A1, L2A2
    # L1B -> L2B1
    # Actionable: L2A1, L2A2, L2B1, NoAction
    edges = [
        ("Root", "L1A"),
        ("Root", "L1B"),
        ("L1A", "L2A1"),
        ("L1A", "L2A2"),
        ("L1B", "L2B1"),
    ]
    costs = {
        "L2A1": 1.0,
        "L2A2": 1.5,
        "L2B1": 2.0,
        "NoAction": 0.0,
    }
    return ActionHierarchy(edges=edges, costs=costs)


@pytest.fixture
def four_level_hierarchy():
    """Creates a deeper four-level action hierarchy for testing."""
    # Structure: GrandRoot -> level1_1, level1_2
    # level1_1 -> level2_1, level2_2
    # level1_2 -> level2_3, level2_4
    # level2_4 -> level3_1, level3_2
    # Actionable: level2_1, level2_2, level2_3, level2_4, level3_1, level3_2, NoAction
    edges = [
        ("GrandRoot", "level1_1"),
        ("GrandRoot", "level1_2"),
        ("level1_1", "level2_1"),
        ("level1_1", "level2_2"),
        ("level1_2", "level2_3"),
        ("level1_2", "level2_4"),
        ("level2_4", "level3_1"),
        ("level2_4", "level3_2"),
    ]
    costs = {
        "level2_1": 1.0,
        "level2_2": 1.1,
        "level2_3": 1.2,
        "level2_4": 1.3,  # This node is also an ancestor of level3_1/level3_2
        "level3_1": 1.4,  # Adjusted costs to make them unique
        "level3_2": 1.5,  # Adjusted costs to make them unique
        "NoAction": 0.0,
    }
    return ActionHierarchy(edges=edges, costs=costs)


@pytest.fixture
def hierarchy_no_noaction_cost():
    """Creates a two-level hierarchy where NoAction is NOT in costs."""
    # Structure: AllActions -> A1, A2
    # Actionable: A1, A2
    edges = [
        ("AllActions", "A1"),
        ("AllActions", "A2"),
    ]
    costs = {
        "A1": 1.0,
        "A2": 2.0,
        # NoAction is intentionally missing from costs
    }
    return ActionHierarchy(edges=edges, costs=costs)


# --- Tests for transform_actions ---


# Renamed from test_transform_actions_two_level to be more general
@pytest.mark.parametrize(
    "hierarchy_fixture, input_action, expected_row, expected_cols",
    [
        # Two-level hierarchy (cols: A_A1, A_A2)
        (
            "two_level_hierarchy",
            "A1",
            [1, 0],
            ["A_A1", "A_A2"],
        ),  # A1 is descendant of A1, not A2
        (
            "two_level_hierarchy",
            "A2",
            [0, 1],
            ["A_A1", "A_A2"],
        ),  # A2 is descendant of A2, not A1
        (
            "two_level_hierarchy",
            "NoAction",
            [0, 0],
            ["A_A1", "A_A2"],
        ),  # NoAction not descendant of A1 or A2 (if isolated)
        (
            "two_level_hierarchy",
            "InvalidAction",
            [0, 0],
            ["A_A1", "A_A2"],
        ),  # Invalid is descendant of nothing
        # Three-level hierarchy (cols: A_L1A, A_L1B, A_L2A1, A_L2A2, A_L2B1)
        (
            "three_level_hierarchy",
            "L2A1",
            [1, 0, 1, 0, 0],
            ["A_L1A", "A_L1B", "A_L2A1", "A_L2A2", "A_L2B1"],
        ),  # L2A1 desc of L1A, L2A1.
        (
            "three_level_hierarchy",
            "L2A2",
            [1, 0, 0, 1, 0],
            ["A_L1A", "A_L1B", "A_L2A1", "A_L2A2", "A_L2B1"],
        ),  # L2A2 desc of L1A, L2A2.
        (
            "three_level_hierarchy",
            "L2B1",
            [0, 1, 0, 0, 1],
            ["A_L1A", "A_L1B", "A_L2A1", "A_L2A2", "A_L2B1"],
        ),  # L2B1 desc of L1B, L2B1.
        (
            "three_level_hierarchy",
            "L1A",
            [0, 0, 0, 0, 0],
            ["A_L1A", "A_L1B", "A_L2A1", "A_L2A2", "A_L2B1"],
        ),  # L1A not actionable, not descendant of included nodes *among actionable*.
        (
            "three_level_hierarchy",
            "NoAction",
            [0, 0, 0, 0, 0],
            ["A_L1A", "A_L1B", "A_L2A1", "A_L2A2", "A_L2B1"],
        ),  # NoAction not descendant of any included nodes
        # Four-level hierarchy (cols: A_level1_1, A_level1_2, A_level2_1, A_level2_2, A_level2_3, A_level2_4, A_level3_1, A_level3_2)
        # Excluded: GrandRoot, NoAction
        (
            "four_level_hierarchy",
            "level3_1",
            [0, 1, 0, 0, 0, 1, 1, 0],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # Ancestors: level3_1, level2_4, level1_2, GrandRoot. Excl: GrandRoot, NoAction.
        (
            "four_level_hierarchy",
            "level3_2",
            [0, 1, 0, 0, 0, 1, 0, 1],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # Ancestors: level3_2, level2_4, level1_2, GrandRoot. Excl: GrandRoot, NoAction.
        (
            "four_level_hierarchy",
            "level2_4",
            [0, 1, 0, 0, 0, 1, 0, 0],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # Ancestors: level2_4, level1_2, GrandRoot. Excl: GrandRoot, NoAction. (Note: level3_1/level3_2 are descendants, but this check is about ancestors of the *chosen* action)
        (
            "four_level_hierarchy",
            "level2_1",
            [1, 0, 1, 0, 0, 0, 0, 0],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # Ancestors: level2_1, level1_1, GrandRoot. Excl: GrandRoot, NoAction.
        (
            "four_level_hierarchy",
            "level1_1",
            [0, 0, 0, 0, 0, 0, 0, 0],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # Not actionable, not descendant of included nodes *among actionable*.
        (
            "four_level_hierarchy",
            "NoAction",
            [0, 0, 0, 0, 0, 0, 0, 0],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # NoAction not descendant of any included nodes
        (
            "four_level_hierarchy",
            "InvalidAction",
            [0, 0, 0, 0, 0, 0, 0, 0],
            [
                "A_level1_1",
                "A_level1_2",
                "A_level2_1",
                "A_level2_2",
                "A_level2_3",
                "A_level2_4",
                "A_level3_1",
                "A_level3_2",
            ],
        ),  # Invalid is descendant of nothing
    ],
)
def test_transform_actions_levels(
    request, hierarchy_fixture, input_action, expected_row, expected_cols
):
    """Tests transform_actions with different hierarchy levels."""
    ah = request.getfixturevalue(hierarchy_fixture)

    actions_series = pl.Series("actions", [input_action])
    expected_df = pl.DataFrame(
        {col: [val] for col, val in zip(expected_cols, expected_row)},
        schema={col: pl.UInt8 for col in expected_cols},
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_df = transform_actions(actions_series, ah)

        if input_action not in ah.costs.keys():
            assert any(issubclass(warn.category, UserWarning) for warn in w)
            assert any(
                f"Input series contains action names not defined in action_hierarchy.costs: ['{input_action}']"
                in str(warn.message)
                for warn in w
            )
        else:
            assert not w

    pl_testing.assert_frame_equal(actual_df, expected_df, check_dtypes=True)


def test_transform_actions_empty_series(two_level_hierarchy):
    """Tests transform_actions with an empty input series."""
    actions_series = pl.Series("actions", [], dtype=pl.String)
    expected_df = pl.DataFrame(schema={"A_A1": pl.UInt8, "A_A2": pl.UInt8})
    actual_df = transform_actions(actions_series, two_level_hierarchy)
    pl_testing.assert_frame_equal(actual_df, expected_df, check_dtypes=True)


# --- Tests for Round Trip Transformation ---


@pytest.mark.parametrize(
    "action_names",
    [
        (["A1", "A2", "NoAction", "A1", "A2"]),  # Two-level actionable nodes
        (["L2A1", "L2A2", "L2B1", "NoAction", "L2A1"]),  # Three-level actionable nodes
        (
            [
                "level2_1",
                "level2_2",
                "level2_3",
                "level2_4",
                "level3_1",
                "level3_2",
                "NoAction",
                "level3_1",
            ]
        ),  # Four-level actionable nodes
        (["A1"]),  # Single action (two-level)
        (["L2A1"]),  # Single action (three-level)
        (["level3_1"]),  # Single action (four-level, leaf)
        (["level2_4"]),  # Single action (four-level, actionable ancestor)
        (["NoAction"]),  # Only NoAction
        ([]),  # Empty input
    ],
)
def test_transform_round_trip(
    two_level_hierarchy, three_level_hierarchy, four_level_hierarchy, action_names
):
    """Tests that transform_actions_to_series is the inverse of transform_actions for valid inputs."""
    hierarchies = [two_level_hierarchy, three_level_hierarchy, four_level_hierarchy]

    for ah in hierarchies:
        valid_action_names = [name for name in action_names if name in ah.costs.keys()]
        actions_series = pl.Series("actions", valid_action_names, dtype=pl.String)

        if actions_series.len() == 0 and not action_names:
            expected_series = pl.Series("chosen_action", [], dtype=pl.String)
        else:
            expected_series = actions_series.alias("chosen_action")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            transformed_df = transform_actions(actions_series, ah)
            reversed_series = transform_actions_to_series(transformed_df, ah)

        pl_testing.assert_series_equal(
            reversed_series, expected_series, check_names=True, check_dtypes=True
        )
