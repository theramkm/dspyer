import pytest

from dspy_transpiler.state import ImmutableState


def test_state_initialization():
    initial_data = {"user_id": 42, "metadata": {"session": "abc"}}
    state = ImmutableState(initial_data)

    # Assert data is deep copied
    assert state.to_dict() == initial_data
    assert state.to_dict() is not initial_data
    assert state.to_dict()["metadata"] is not initial_data["metadata"]


def test_apply_patch_addition_and_update():
    state = ImmutableState({"user_id": 42, "status": "active"})
    patch = {"status": "inactive", "new_key": "hello"}

    new_state = state.apply_patch(patch)

    # Verify updates
    assert new_state.to_dict() == {"user_id": 42, "status": "inactive", "new_key": "hello"}
    # Verify original state remains untouched
    assert state.to_dict() == {"user_id": 42, "status": "active"}


def test_apply_patch_deletion():
    state = ImmutableState({"user_id": 42, "status": "active", "metadata": {"session": "abc"}})
    patch = {"status": None, "metadata": {"session": None}}

    new_state = state.apply_patch(patch)

    assert new_state.to_dict() == {"user_id": 42, "metadata": {}}
    assert "status" not in new_state.to_dict()


def test_apply_patch_deep_merge():
    state = ImmutableState(
        {"user": {"name": "Alice", "preferences": {"theme": "dark", "notifications": True}}}
    )
    patch = {"user": {"preferences": {"theme": "light"}, "age": 30}}

    new_state = state.apply_patch(patch)

    expected = {
        "user": {
            "name": "Alice",
            "age": 30,
            "preferences": {"theme": "light", "notifications": True},
        }
    }
    assert new_state.to_dict() == expected


def test_apply_patch_mismatch_replacement():
    # If target has a dict, but patch has a list/primitive, replace it
    state = ImmutableState({"config": {"nested": "value"}, "tags": ["a", "b"]})
    patch = {"config": "flat_value", "tags": {"new_structure": True}}

    new_state = state.apply_patch(patch)

    assert new_state.to_dict() == {"config": "flat_value", "tags": {"new_structure": True}}


def test_apply_patch_list_replacement():
    # Lists are replaced entirely, not merged element-wise
    state = ImmutableState({"items": [1, 2, 3]})
    patch = {"items": [4, 5]}

    new_state = state.apply_patch(patch)

    assert new_state.to_dict() == {"items": [4, 5]}


def test_state_merge_last_write_wins():
    state1 = ImmutableState({"a": 1, "b": [1, 2], "c": {"nested": "value"}})
    state2 = ImmutableState({"a": 2, "b": [3, 4], "d": 4})

    merged = state1.merge(state2, policy="last_write_wins")
    assert merged.to_dict() == {"a": 2, "b": [3, 4], "c": {"nested": "value"}, "d": 4}


def test_state_merge_combine_lists():
    state1 = ImmutableState({"a": 1, "b": [1, 2], "c": {"nested": "value"}})
    state2 = ImmutableState({"a": 2, "b": [3, 4], "d": 4})

    merged = state1.merge(state2, policy="combine_lists")
    assert merged.to_dict() == {"a": 2, "b": [1, 2, 3, 4], "c": {"nested": "value"}, "d": 4}


def test_state_merge_raise():
    state1 = ImmutableState({"a": 1, "b": [1, 2]})
    state2 = ImmutableState({"a": 2, "b": [1, 2]})

    # Conflict on 'a' should raise ValueError
    with pytest.raises(ValueError) as excinfo:
        state1.merge(state2, policy="raise")
    assert "Conflict detected at key 'a'" in str(excinfo.value)

    # Mismatched lists under 'raise' should also raise ValueError
    state3 = ImmutableState({"a": 1, "b": [3, 4]})
    with pytest.raises(ValueError) as excinfo:
        state1.merge(state3, policy="raise")
    assert "Conflict detected at key 'b'" in str(excinfo.value)
